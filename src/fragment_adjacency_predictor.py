import os
import numpy as np
import torch
import torch.nn as nn
from src.model import BaseModel


class CNNEncoder(nn.Module):

    def __init__(self, embedding_dim=256, dropout=0.3):
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
        return self.head(self.conv(x))


class ComparisonHead(nn.Module):
    """Takes two embeddings and predicts a binary probability."""

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),           # diff + product + both embeddings
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        combined = torch.cat([a - b, a * b, a, b], dim=-1)
        return self.mlp(combined).squeeze(-1)


class FragmentAdjacencyPredictor(BaseModel):
    """
    Siamese CNN with one or two prediction heads:

    - The adjacency head predicts whether two fragments are spatially adjacent
      in their source image. The labels come from the grid structure.
    - The optional same-image head predicts whether two fragments come from
      the same source image. The labels come from `labels` passed to
      train_step. This head is only created when `lambda_same > 0`.

    The total loss is

        L = lambda_adj * WBCE(adjacency)  +  lambda_same * WBCE(same_image),

    where each WBCE is weighted by its own pos_weight (default 1, i.e. plain
    BCE; see results section of the doc for why this choice was made).

    At inference, get_output returns the similarity matrix used by the
    downstream clustering. When the same-image head is active, its
    probabilities are returned, since they are more directly aligned with
    the clustering objective. Otherwise the adjacency-head probabilities
    are returned.
    """

    def __init__(self, embedding_dim=256, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 pos_weight_adj=1.0, lambda_adj=1.0,
                 pos_weight_same=1.0, lambda_same=0.0):
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = CNNEncoder(embedding_dim, dropout).to(self.device)
        self.adj_head = ComparisonHead(embedding_dim).to(self.device)

        self.lambda_adj  = float(lambda_adj)
        self.lambda_same = float(lambda_same)
        self.pos_weight_adj  = float(pos_weight_adj)
        self.pos_weight_same = float(pos_weight_same)

        params = list(self.encoder.parameters()) + list(self.adj_head.parameters())
        if self.lambda_same > 0:
            self.same_head = ComparisonHead(embedding_dim).to(self.device)
            params += list(self.same_head.parameters())
        else:
            self.same_head = None

        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.loss_fn   = nn.BCELoss(reduction='none')

    def _to_tensor(self, fragments):
        x = torch.tensor(fragments, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)                       # (N, H, W, C) → (N, C, H, W)
        return x.to(self.device)

    def _wbce(self, preds, targets, pos_weight):
        """Properly weighted BCE: per-pair losses, weighted, then mean."""
        weight = torch.where(targets == 1,
                             torch.tensor(pos_weight, device=self.device),
                             torch.ones((), device=self.device))
        return (self.loss_fn(preds, targets) * weight).mean()

    def train_step(self, fragments, labels, adjacency):
        self.encoder.train()
        self.adj_head.train()
        if self.same_head is not None:
            self.same_head.train()

        x = self._to_tensor(fragments)
        embeddings = self.encoder(x)                     # (N, D)

        # all pairwise combinations
        n = embeddings.shape[0]
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        emb_i = embeddings[idx_i]
        emb_j = embeddings[idx_j]

        # adjacency loss
        adj_targets = torch.tensor(
            adjacency[idx_i.cpu(), idx_j.cpu()],
            dtype=torch.float32,
            device=self.device,
        )
        adj_preds = self.adj_head(emb_i, emb_j)
        loss = self.lambda_adj * self._wbce(adj_preds, adj_targets, self.pos_weight_adj)

        # same-image loss
        if self.same_head is not None:
            labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)
            same_targets = (labels_t[idx_i] == labels_t[idx_j]).float()
            same_preds   = self.same_head(emb_i, emb_j)
            loss = loss + self.lambda_same * self._wbce(
                same_preds, same_targets, self.pos_weight_same
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_output(self, fragments):
        """
        Returns (N, N) similarity matrix for clustering. Uses the same-image
        head's output when it exists, otherwise the adjacency head's output.
        """
        self.encoder.eval()
        self.adj_head.eval()
        if self.same_head is not None:
            self.same_head.eval()

        with torch.no_grad():
            x = self._to_tensor(fragments)
            embeddings = self.encoder(x)

            n = embeddings.shape[0]
            similarity = torch.zeros(n, n, device=self.device)
            idx_i, idx_j = torch.triu_indices(n, n, offset=1)
            emb_i = embeddings[idx_i]
            emb_j = embeddings[idx_j]

            head = self.same_head if self.same_head is not None else self.adj_head
            probs = head(emb_i, emb_j)

            similarity[idx_i, idx_j] = probs
            similarity[idx_j, idx_i] = probs

        return similarity.cpu().numpy()

    def _pair_scores(self, fragments, head):
        """Return per-pair confidence scores from a given head, plus the upper-
        triangular index pairs used. Eval mode, no grad."""
        self.encoder.eval()
        head.eval()
        with torch.no_grad():
            x = self._to_tensor(fragments)
            embeddings = self.encoder(x)
            n = embeddings.shape[0]
            idx_i, idx_j = torch.triu_indices(n, n, offset=1)
            probs = head(embeddings[idx_i], embeddings[idx_j]).cpu().numpy()
        return probs, idx_i, idx_j

    def evaluate_adjacency(self, fragments, adjacency):
        """
        Threshold-independent adjacency metrics: AUROC and AUPRC, computed on
        the adjacency head's outputs.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        probs, idx_i, idx_j = self._pair_scores(fragments, self.adj_head)
        targets = adjacency[idx_i.cpu(), idx_j.cpu()]
        return {
            'auroc': float(roc_auc_score(targets, probs)),
            'auprc': float(average_precision_score(targets, probs)),
        }

    def evaluate_same_image(self, fragments, labels):
        """
        Threshold-independent same-image metrics: AUROC and AUPRC, computed on
        the same-image head's outputs. Only valid when the same-image head is
        active (lambda_same > 0); returns None if the head was never created.

        labels: (N,) array of source-image indices, one per fragment.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        if self.same_head is None:
            return None
        probs, idx_i, idx_j = self._pair_scores(fragments, self.same_head)
        labels = np.asarray(labels)
        targets = (labels[idx_i.cpu().numpy()] == labels[idx_j.cpu().numpy()]).astype(np.float32)
        return {
            'auroc': float(roc_auc_score(targets, probs)),
            'auprc': float(average_precision_score(targets, probs)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'encoder':  self.encoder.state_dict(),
            'adj_head': self.adj_head.state_dict(),
        }
        if self.same_head is not None:
            state['same_head'] = self.same_head.state_dict()
        torch.save(state, path + '.pt')

    def load(self, path):
        checkpoint = torch.load(path + '.pt', map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        # backward compat: old checkpoints stored adjacency head under 'head'
        adj_state = checkpoint.get('adj_head', checkpoint.get('head'))
        self.adj_head.load_state_dict(adj_state)
        if 'same_head' in checkpoint and self.same_head is not None:
            self.same_head.load_state_dict(checkpoint['same_head'])
