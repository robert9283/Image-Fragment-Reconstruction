"""
Submission script 1: collect metrics over N_SAMPLES samples.
Edit DATA_PATH and CHECKPOINT_PATH, then run directly.
Saves results to results.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency
from src.clustering import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

# ── edit these two paths ──────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'model')
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 1000
N_IMAGES  = 10

dataset = Imagenet64(DATA_PATH)
gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds='test', augmentation=False)

model = FragmentAdjacencyPredictor()
model.load(CHECKPOINT_PATH)
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

adjacency_results  = {'precision': [], 'recall': [], 'f1': []}
clustering_results = {'ari': [], 'nmi': [], 'purity': []}

for i in range(N_SAMPLES):
    images, _ = next(gen)
    fragments, true_labels = extract_fragments(np.array(images))
    adjacency = build_adjacency(n_images=N_IMAGES)

    # ── adjacency prediction ──────────────────────────────────────────────────
    adj_metrics = model.evaluate_adjacency(fragments, adjacency)
    for key, val in adj_metrics.items():
        adjacency_results[key].append(val)

    # ── fragment clustering ───────────────────────────────────────────────────
    similarity   = model.get_output(fragments)
    pred_labels  = cluster(similarity)
    cl_metrics   = compute_metrics(pred_labels, true_labels)
    for key, val in cl_metrics.items():
        clustering_results[key].append(val)

    if (i + 1) % 100 == 0:
        print(f"[{i + 1}/{N_SAMPLES}]  ARI={np.mean(clustering_results['ari']):.3f}  F1={np.mean(adjacency_results['f1']):.3f}")

# ── assemble and save results ─────────────────────────────────────────────────
results = {
    'adjacency_prediction': {
        'precision': round(float(np.mean(adjacency_results['precision'])), 4),
        'recall':    round(float(np.mean(adjacency_results['recall'])),    4),
        'f1':        round(float(np.mean(adjacency_results['f1'])),        4),
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

out_path = os.path.join(os.path.dirname(__file__), '..', 'results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n── Results ──────────────────────────────────────────")
print(json.dumps(results, indent=2))
print(f"\nSaved: {out_path}")
