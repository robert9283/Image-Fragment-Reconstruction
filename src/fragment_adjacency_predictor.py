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
        """Returns (N, N) similarity matrix p_ij for downstream clustering."""
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
        """Per-pair scores p_ij from the head. Eval mode, no grad."""
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
        """AUROC and AUPRC for adjacency prediction."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        probs, idx_i, idx_j = self._pair_scores(fragments)
        targets = adjacency[idx_i.cpu(), idx_j.cpu()]
        return {
            'auroc': float(roc_auc_score(targets, probs)),
            'auprc': float(average_precision_score(targets, probs)),
        }

    def evaluate_same_image(self, fragments, labels):
        """AUROC and AUPRC for same-image prediction. Returns None if lambda_same == 0."""
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder':  self.encoder.state_dict(),
            'adj_head': self.adj_head.state_dict(),
        }, path + '.pt')

    def load(self, path):
        checkpoint = torch.load(path + '.pt', map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        adj_state = checkpoint.get('adj_head', checkpoint.get('head'))
        self.adj_head.load_state_dict(adj_state)
