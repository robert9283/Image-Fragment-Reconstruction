"""
Submission script 1: collect metrics over N_SAMPLES samples.
Edit DATA_PATH and CHECKPOINT_PATH, then run directly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from data import Imagenet64
from src.fragments import extract_fragments
from src.evaluate import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

# ── edit these two paths ──────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model')
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 1000
N_IMAGES  = 10

dataset = Imagenet64(DATA_PATH)
gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds='test', augmentation=False)

model = FragmentAdjacencyPredictor()
model.load(CHECKPOINT_PATH)
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

results = {'ari': [], 'nmi': [], 'purity': []}

for i in range(N_SAMPLES):
    images, _ = next(gen)
    fragments, true_labels = extract_fragments(np.array(images))

    model_output = model.get_output(fragments)
    pred_labels  = cluster(model_output)
    metrics      = compute_metrics(pred_labels, true_labels)
    for key, val in metrics.items():
        results[key].append(val)

    if (i + 1) % 100 == 0:
        print(f"[{i + 1}/{N_SAMPLES}]  ARI={np.mean(results['ari']):.3f}")

print("\n── Results ──────────────────────────────────────────")
for metric, values in results.items():
    print(f"  {metric.upper():7s}  mean={np.mean(values):.4f}  std={np.std(values):.4f}")
