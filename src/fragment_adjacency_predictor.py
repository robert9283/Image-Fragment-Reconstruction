"""
Model definition for the fragment adjacency predictor.

This module implements the core neural network used for self-supervised image
fragment clustering. It consists of a shared CNN encoder (CNNEncoder) that maps
16×16 fragments to embeddings, a comparison head (ComparisonHead) that predicts
a scalar confidence score for each fragment pair, and the top-level model class
(FragmentAdjacencyPredictor) that combines them into a trainable system.

Classes:
    CNNEncoder               -- Shared convolutional encoder: fragment → embedding.
    ComparisonHead           -- MLP that scores a pair of embeddings.
    FragmentAdjacencyPredictor -- Full model with training, evaluation, and I/O.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from src.model import BaseModel


class CNNEncoder(nn.Module):
    """
    Convolutional encoder that maps a 16×16 RGB fragment to a fixed-size embedding.

    Architecture: three Conv-BN-ReLU blocks with two max-pooling layers, followed
    by a linear projection, ReLU, and dropout. Output shape: (N, embedding_dim).
    """

    def __init__(self, embedding_dim=256, dropout=0.3):
        """
        Args:
            embedding_dim (int): Size of the output embedding vector. Default 256.
            dropout (float): Dropout probability applied after the linear layer. Default 0.3.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 8×8×64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 4×4×128
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input fragments of shape (N, 3, 16, 16).

        Returns:
            torch.Tensor: Embeddings of shape (N, embedding_dim).
        """
        return self.head(self.conv(x))


class ComparisonHead(nn.Module):
    """
    MLP that takes two fragment embeddings and predicts a scalar confidence score.

    The two embeddings are combined into a 4×embedding_dim feature vector using
    element-wise difference, element-wise product, and the two embeddings
    themselves: [a−b, a*b, a, b]. A two-layer MLP with sigmoid output maps this
    to a score in (0, 1).
    """

    def __init__(self, embedding_dim=256):
        """
        Args:
            embedding_dim (int): Size of each input embedding. Default 256.
                The MLP input size is embedding_dim * 4.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),           # diff + product + both embeddings
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        """
        Args:
            a (torch.Tensor): Embeddings for the first fragment in each pair, shape (P, D).
            b (torch.Tensor): Embeddings for the second fragment in each pair, shape (P, D).

        Returns:
            torch.Tensor: Confidence scores of shape (P,), values in (0, 1).
        """
        combined = torch.cat([a - b, a * b, a, b], dim=-1)
        return self.mlp(combined).squeeze(-1)


class FragmentAdjacencyPredictor(BaseModel):
    """
    Siamese CNN with a single comparison head trained on a combined loss:

        L = lambda_adj * WBCE(p_ij, y_adj)  +  lambda_same * BCE(p_ij, y_same)

    Both loss terms act on the same per-pair score p_ij produced by the single
    head. The adjacency term uses weighted BCE (pos_weight_adj = beta * ratio)
    to handle the 52:1 class imbalance; the same-image term uses plain BCE
    (mild 9.6:1 imbalance does not require reweighting).

    Setting lambda_same = 0 recovers pure adjacency prediction.
    Setting lambda_adj = 0 trains solely on same-image membership.

    At inference, get_output returns the p_ij similarity matrix directly.
    """

    def __init__(self, embedding_dim=256, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 pos_weight_adj=1.0, lambda_adj=1.0, lambda_same=0.0):
        """
        Args:
            embedding_dim (int): Size of the fragment embeddings. Default 256.
            dropout (float): Dropout probability in the encoder. Default 0.3.
            lr (float): Adam learning rate. Default 1e-3.
            weight_decay (float): Adam weight decay. Default 1e-4.
            pos_weight_adj (float): Weight applied to adjacent (positive) pairs in
                the adjacency WBCE loss. Typically set to beta * (n_neg / n_pos).
                Default 1.0 (no reweighting).
            lambda_adj (float): Weight on the adjacency loss term. Default 1.0.
            lambda_same (float): Weight on the same-image loss term. Set to 0 to
                disable the same-image objective. Default 0.0.
        """
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder  = CNNEncoder(embedding_dim, dropout).to(self.device)
        self.adj_head = ComparisonHead(embedding_dim).to(self.device)

        self.lambda_adj     = float(lambda_adj)
        self.lambda_same    = float(lambda_same)
        self.pos_weight_adj = float(pos_weight_adj)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.adj_head.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        self.loss_fn = nn.BCELoss(reduction='none')

    def _to_tensor(self, fragments):
        """
        Convert a numpy fragment array to a channel-first float tensor on the model device.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, H, W, C).

        Returns:
            torch.Tensor: Float tensor of shape (N, C, H, W) on self.device.
        """
        x = torch.tensor(fragments, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)                       # (N, H, W, C) → (N, C, H, W)
        return x.to(self.device)

    def _wbce(self, preds, targets, pos_weight):
        """
        Weighted binary cross-entropy loss.

        Computes per-pair BCE losses, multiplies each by its class weight
        (pos_weight for positives, 1.0 for negatives), then takes the mean.

        Args:
            preds (torch.Tensor): Predicted scores of shape (P,), values in (0, 1).
            targets (torch.Tensor): Binary ground-truth labels of shape (P,).
            pos_weight (float): Multiplier applied to the loss of positive pairs.

        Returns:
            torch.Tensor: Scalar weighted mean BCE loss.
        """
        weight = torch.where(targets == 1,
                             torch.tensor(pos_weight, device=self.device),
                             torch.ones((), device=self.device))
        return (self.loss_fn(preds, targets) * weight).mean()

    def train_step(self, fragments, labels, adjacency):
        """
        Run one training iteration on a single batch.

        Encodes all fragments, computes scores for every upper-triangular pair,
        and minimises the combined adjacency + same-image loss.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).
            labels (np.ndarray): Integer source-image index for each fragment,
                shape (N,).
            adjacency (np.ndarray): Binary adjacency matrix of shape (N, N),
                where entry (i, j) = 1 if fragments i and j are spatially adjacent.

        Returns:
            float: Scalar training loss for this iteration.
        """
        self.encoder.train()
        self.adj_head.train()

        x = self._to_tensor(fragments)
        embeddings = self.encoder(x)                     # (N, D)

        # all pairwise combinations
        n = embeddings.shape[0]
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        emb_i = embeddings[idx_i]
        emb_j = embeddings[idx_j]

        preds = self.adj_head(emb_i, emb_j)

        # adjacency loss (weighted BCE)
        adj_targets = torch.tensor(
            adjacency[idx_i.cpu(), idx_j.cpu()],
            dtype=torch.float32,
            device=self.device,
        )
        loss = self.lambda_adj * self._wbce(preds, adj_targets, self.pos_weight_adj)

        # same-image loss (plain BCE, same preds)
        if self.lambda_same > 0:
            labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)
            same_targets = (labels_t[idx_i] == labels_t[idx_j]).float()
            loss = loss + self.lambda_same * self.loss_fn(preds, same_targets).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_output(self, fragments):
        """
        Compute the full pairwise similarity matrix for downstream clustering.

        Runs the encoder and comparison head on all upper-triangular pairs and
        assembles the result into a symmetric (N, N) matrix.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).

        Returns:
            np.ndarray: Symmetric similarity matrix of shape (N, N), values in (0, 1).
        """
        self.encoder.eval()
        self.adj_head.eval()
        with torch.no_grad():
            x = self._to_tensor(fragments)
            embeddings = self.encoder(x)
            n = embeddings.shape[0]
            similarity = torch.zeros(n, n, device=self.device)
            idx_i, idx_j = torch.triu_indices(n, n, offset=1)
            probs = self.adj_head(embeddings[idx_i], embeddings[idx_j])
            similarity[idx_i, idx_j] = probs
            similarity[idx_j, idx_i] = probs
        return similarity.cpu().numpy()

    def _pair_scores(self, fragments):
        """
        Return raw per-pair confidence scores for all upper-triangular pairs.

        Helper used by evaluate_adjacency, evaluate_same_image, and external
        analysis scripts. Runs in eval mode with no gradient computation.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).

        Returns:
            tuple:
                probs (np.ndarray): Scores of shape (P,), where P = N*(N-1)/2.
                idx_i (torch.Tensor): Row indices of the upper-triangular pairs.
                idx_j (torch.Tensor): Column indices of the upper-triangular pairs.
        """
        self.encoder.eval()
        self.adj_head.eval()
        with torch.no_grad():
            x = self._to_tensor(fragments)
            embeddings = self.encoder(x)
            n = embeddings.shape[0]
            idx_i, idx_j = torch.triu_indices(n, n, offset=1)
            probs = self.adj_head(embeddings[idx_i], embeddings[idx_j]).cpu().numpy()
        return probs, idx_i, idx_j

    def evaluate_adjacency(self, fragments, adjacency):
        """
        Compute AUROC and AUPRC for the adjacency prediction task.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).
            adjacency (np.ndarray): Binary ground-truth adjacency matrix, shape (N, N).

        Returns:
            dict: {'auroc': float, 'auprc': float}
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        probs, idx_i, idx_j = self._pair_scores(fragments)
        targets = adjacency[idx_i.cpu(), idx_j.cpu()]
        return {
            'auroc': float(roc_auc_score(targets, probs)),
            'auprc': float(average_precision_score(targets, probs)),
        }

    def evaluate_same_image(self, fragments, labels):
        """
        Compute AUROC and AUPRC for the same-image prediction task.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).
            labels (np.ndarray): Integer source-image index for each fragment, shape (N,).

        Returns:
            dict: {'auroc': float, 'auprc': float}, or None if lambda_same == 0.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        if self.lambda_same == 0:
            return None
        probs, idx_i, idx_j = self._pair_scores(fragments)
        labels = np.asarray(labels)
        targets = (labels[idx_i.cpu().numpy()] == labels[idx_j.cpu().numpy()]).astype(np.float32)
        return {
            'auroc': float(roc_auc_score(targets, probs)),
            'auprc': float(average_precision_score(targets, probs)),
        }

    def save(self, path):
        """
        Save the encoder and comparison head state dicts to disk.

        Args:
            path (str): Destination path without extension. The file is written
                as path + '.pt'.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder':  self.encoder.state_dict(),
            'adj_head': self.adj_head.state_dict(),
        }, path + '.pt')

    def load(self, path):
        """
        Load encoder and comparison head weights from a checkpoint file.

        Supports checkpoints saved under the key 'adj_head' (current) or
        the legacy key 'head' (pre-refactor).

        Args:
            path (str): Path to the checkpoint without extension. Loads path + '.pt'.
        """
        checkpoint = torch.load(path + '.pt', map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        adj_state = checkpoint.get('adj_head', checkpoint.get('head'))
        self.adj_head.load_state_dict(adj_state)
