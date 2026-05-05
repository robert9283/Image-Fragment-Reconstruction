"""
Submission script 1: evaluate the model on N_SAMPLES test batches and report metrics.

This script measures two things on the held-out test split:

  1. Adjacency prediction  — how well the model scores spatially adjacent
     fragment pairs higher than non-adjacent ones (AUROC, AUPRC, F1 at
     threshold 0.5).

  2. Fragment clustering   — how accurately balanced spectral clustering
     reconstructs the original source-image groupings (ARI, NMI, purity,
     each reported as mean ± std over N_SAMPLES batches).

Configuration is read from the config.yaml snapshot stored alongside the
checkpoint. Results are printed to stdout and saved as test_metrics.json in
the run directory.

Usage:
    python src/script1_metrics.py

Override paths via environment variables if needed:
    DATA_PATH=path/to/data CHECKPOINT_PATH=runs/my_run/model python src/script1_metrics.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import yaml
import numpy as np
from sklearn.metrics import f1_score
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID
from src.clustering import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

# ── edit these two paths ──────────────────────────────────────────────────────
DATA_PATH       = os.environ.get('DATA_PATH',
                    os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data'))
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH',
                    os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'model'))
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 1000

# config is read from the run directory (parent of checkpoint path)
RUN_DIR = os.path.dirname(os.path.realpath(CHECKPOINT_PATH))
with open(os.path.join(RUN_DIR, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

N_IMAGES = cfg.get('n_images', 10)

dataset = Imagenet64(DATA_PATH)
gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds='test', augmentation=False)

n_pos   = N_IMAGES * 2 * GRID * (GRID - 1)
n_pairs = (N_IMAGES * GRID**2) * (N_IMAGES * GRID**2 - 1) // 2
ratio   = (n_pairs - n_pos) / n_pos

model = FragmentAdjacencyPredictor(
    pos_weight_adj = cfg.get('beta', 0.01923) * ratio,
    lambda_adj     = cfg.get('lambda_adj',  1.0),
    lambda_same    = cfg.get('lambda_same', 0.0),
)
model.load(CHECKPOINT_PATH)
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

adjacency_results  = {'auroc': [], 'auprc': [], 'f1': []}
clustering_results = {'ari': [], 'nmi': [], 'purity': []}

for i in range(N_SAMPLES):
    images, _ = next(gen)
    fragments, true_labels = extract_fragments(np.array(images))
    adjacency = build_adjacency(n_images=N_IMAGES)

    # ── adjacency prediction ──────────────────────────────────────────────────
    adj_metrics = model.evaluate_adjacency(fragments, adjacency)
    probs, idx_i, idx_j = model._pair_scores(fragments)
    targets = adjacency[idx_i.cpu().numpy(), idx_j.cpu().numpy()]
    preds   = (np.array(probs) >= 0.5).astype(int)
    adjacency_results['auroc'].append(adj_metrics['auroc'])
    adjacency_results['auprc'].append(adj_metrics['auprc'])
    adjacency_results['f1'].append(float(f1_score(targets, preds, zero_division=0)))

    # ── fragment clustering ───────────────────────────────────────────────────
    similarity   = model.get_output(fragments)
    pred_labels  = cluster(similarity, n_per_cluster=GRID * GRID)
    cl_metrics   = compute_metrics(pred_labels, true_labels)
    for key, val in cl_metrics.items():
        clustering_results[key].append(val)

    if (i + 1) % 100 == 0:
        print(f"[{i + 1}/{N_SAMPLES}]  ARI={np.mean(clustering_results['ari']):.3f}  F1={np.mean(adjacency_results['f1']):.3f}")

# ── assemble and save results ─────────────────────────────────────────────────
results = {
    'adjacency_prediction': {
        'auroc': round(float(np.mean(adjacency_results['auroc'])), 4),
        'auprc': round(float(np.mean(adjacency_results['auprc'])), 4),
        'f1':    round(float(np.mean(adjacency_results['f1'])),    4),
    },
    'fragment_clustering': {
        'ari_mean':    round(float(np.mean(clustering_results['ari'])),    4),
        'ari_std':     round(float(np.std(clustering_results['ari'])),     4),
        'nmi_mean':    round(float(np.mean(clustering_results['nmi'])),    4),
        'nmi_std':     round(float(np.std(clustering_results['nmi'])),     4),
        'purity_mean': round(float(np.mean(clustering_results['purity'])), 4),
        'purity_std':  round(float(np.std(clustering_results['purity'])),  4),
    }
}

out_path = os.path.join(RUN_DIR, 'test_metrics.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n── Results ──────────────────────────────────────────")
print(json.dumps(results, indent=2))
print(f"\nSaved: {out_path}")
