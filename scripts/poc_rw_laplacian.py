#!/usr/bin/env python3
"""
Proof-of-concept: random-walk Laplacian vs symmetric Laplacian for spectral clustering.

Loads the adj_only_seed_1 checkpoint (no code changes to src/), applies both
clustering variants to N_BATCHES validation batches and reports mean ARI.

Usage (from project root):
    source venv/bin/activate
    python scripts/poc_rw_laplacian.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))

import numpy as np
import yaml
from sklearn.manifold import SpectralEmbedding

from data import Imagenet64
from src.fragments import extract_fragments, GRID
from src.clustering import _balanced_kmeans, compute_metrics

# ── config ────────────────────────────────────────────────────────────────────

RUN_DIR    = 'runs/adj_only_seed_1'
N_BATCHES  = 50      # number of validation batches to average over
N_CLUSTERS = 10
N_PER_CLUSTER = GRID * GRID   # 16

# ── load model ────────────────────────────────────────────────────────────────

with open(os.path.join(RUN_DIR, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

# inline load_model to avoid importing main.py which has side effects
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

ratio = ((cfg['n_images'] * GRID**2) * (cfg['n_images'] * GRID**2 - 1) // 2
         - cfg['n_images'] * 2 * GRID * (GRID - 1)) / (cfg['n_images'] * 2 * GRID * (GRID - 1))

model = FragmentAdjacencyPredictor(
    pos_weight_adj = cfg.get('beta', 0.01923) * ratio,
    lambda_adj     = cfg.get('lambda_adj', 1.0),
    lambda_same    = cfg.get('lambda_same', 0.0),
)
model.load(os.path.join(RUN_DIR, 'model'))
print(f"Loaded checkpoint from {RUN_DIR}")

# ── data generator ────────────────────────────────────────────────────────────

ds      = Imagenet64(cfg['data_path'])
datagen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)

# ── clustering helpers ────────────────────────────────────────────────────────

def spectral_embed(W, k):
    return SpectralEmbedding(
        n_components=k, affinity='precomputed', random_state=42
    ).fit_transform(W)


def cluster_sym(W):
    """Standard symmetric-Laplacian spectral clustering."""
    emb = spectral_embed(W, N_CLUSTERS)
    return _balanced_kmeans(emb, k=N_CLUSTERS, size=N_PER_CLUSTER)


def cluster_rw(W):
    """Random-walk Laplacian: rescale embedding rows by 1/sqrt(degree)."""
    emb = spectral_embed(W, N_CLUSTERS)
    degrees = W.sum(axis=1)
    emb_rw = emb / np.sqrt(degrees[:, None])
    return _balanced_kmeans(emb_rw, k=N_CLUSTERS, size=N_PER_CLUSTER)

# ── evaluation loop ───────────────────────────────────────────────────────────

ari_sym_list, ari_rw_list = [], []

for i in range(N_BATCHES):
    images, _ = next(datagen)
    fragments, labels = extract_fragments(np.array(images))
    labels = np.array(labels)

    W = model.get_output(fragments)   # (N, N) affinity matrix

    pred_sym = cluster_sym(W)
    pred_rw  = cluster_rw(W)

    ari_sym_list.append(compute_metrics(pred_sym, labels)['ari'])
    ari_rw_list.append(compute_metrics(pred_rw,  labels)['ari'])

    if (i + 1) % 10 == 0:
        print(f"  batch {i+1:3d}/{N_BATCHES}  "
              f"sym={np.mean(ari_sym_list):.4f}  "
              f"rw={np.mean(ari_rw_list):.4f}")

# ── results ───────────────────────────────────────────────────────────────────

ari_sym = np.array(ari_sym_list)
ari_rw  = np.array(ari_rw_list)

print(f"\n{'Method':<8}  {'Mean ARI':>9}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
print("-" * 45)
print(f"{'L_sym':<8}  {ari_sym.mean():>9.4f}  {ari_sym.std():>7.4f}  "
      f"{ari_sym.min():>7.4f}  {ari_sym.max():>7.4f}")
print(f"{'L_rw':<8}  {ari_rw.mean():>9.4f}  {ari_rw.std():>7.4f}  "
      f"{ari_rw.min():>7.4f}  {ari_rw.max():>7.4f}")
print(f"\nDelta (rw - sym): {ari_rw.mean() - ari_sym.mean():+.4f}")

# paired t-test
from scipy import stats
t, p = stats.ttest_rel(ari_rw, ari_sym)
print(f"Paired t-test:    t={t:.3f}  p={p:.4f}")
