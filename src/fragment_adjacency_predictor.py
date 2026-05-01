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
    """Takes two embeddings and predicts adjacency probability."""

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

    def __init__(self, embedding_dim=256, dropout=0.3, lr=1e-3, weight_decay=1e-4):
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = CNNEncoder(embedding_dim, dropout).to(self.device)
        self.head    = ComparisonHead(embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.BCELoss()

    def _to_tensor(self, fragments):
        x = torch.tensor(fragments, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)                       # (N, H, W, C) → (N, C, H, W)
        return x.to(self.device)

    def train_step(self, fragments, labels, adjacency):
        self.encoder.train()
        self.head.train()

        x = self._to_tensor(fragments)
        embeddings = self.encoder(x)                     # (N, D)

        # all pairwise combinations
        n = embeddings.shape[0]
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        emb_i = embeddings[idx_i]
        emb_j = embeddings[idx_j]

        preds  = self.head(emb_i, emb_j)
        targets = torch.tensor(
            adjacency[idx_i.cpu(), idx_j.cpu()],
            dtype=torch.float32,
            device=self.device,
        )

        # weighted loss to handle class imbalance (few positives vs many negatives)
        n_pos = targets.sum().item()
        n_neg = len(targets) - n_pos
        weight = torch.where(targets == 1,
                             torch.tensor(n_neg / (n_pos + 1e-6), device=self.device),
                             torch.ones(len(targets), device=self.device))
        loss = (self.loss_fn(preds, targets) * weight).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_output(self, fragments):
        """Returns (N, N) similarity matrix for spectral clustering."""
        self.encoder.eval()
        self.head.eval()

        with torch.no_grad():
            x = self._to_tensor(fragments)
            embeddings = self.encoder(x)

            n = embeddings.shape[0]
            similarity = torch.zeros(n, n, device=self.device)
            idx_i, idx_j = torch.triu_indices(n, n, offset=1)
            emb_i = embeddings[idx_i]
            emb_j = embeddings[idx_j]
            probs = self.head(emb_i, emb_j)

            similarity[idx_i, idx_j] = probs
            similarity[idx_j, idx_i] = probs             # symmetric

        return similarity.cpu().numpy()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'head':    self.head.state_dict(),
        }, path + '.pt')

    def load(self, path):
        checkpoint = torch.load(path + '.pt', map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.head.load_state_dict(checkpoint['head'])
