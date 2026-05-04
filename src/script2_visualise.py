"""
Submission script 2: visualise clustering on a single sample.
Edit DATA_PATH and CHECKPOINT_PATH, then run directly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from data import Imagenet64
from src.fragments import extract_fragments, GRID, FRAGMENT_SIZE
from src.clustering import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

# ── edit these two paths ──────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'model')
# ─────────────────────────────────────────────────────────────────────────────

N_IMAGES = 10

dataset = Imagenet64(DATA_PATH)
gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds='test', augmentation=False)
images, _ = next(gen)
fragments, true_labels = extract_fragments(np.array(images))

model = FragmentAdjacencyPredictor()
model.load(CHECKPOINT_PATH)
model_output = model.get_output(fragments)
pred_labels  = cluster(model_output)
metrics      = compute_metrics(pred_labels, true_labels)

print(f"ARI={metrics['ari']:.3f}  NMI={metrics['nmi']:.3f}  purity={metrics['purity']:.3f}")


def fragments_to_image(frags):
    """frags: (16, 16, 16, 3) in row-major grid order → (64, 64, 3)"""
    return (frags.reshape(GRID, GRID, FRAGMENT_SIZE, FRAGMENT_SIZE, 3)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(64, 64, 3))


def align_labels(true_labels, pred_labels, k=N_IMAGES):
    """Hungarian matching: map predicted cluster ids to true image ids."""
    cost = np.zeros((k, k))
    for t in range(k):
        for p in range(k):
            cost[t, p] = -np.sum((true_labels == t) & (pred_labels == p))
    _, col_ind = linear_sum_assignment(cost)
    mapping = {pred: true for true, pred in enumerate(col_ind)}
    return np.array([mapping[p] for p in pred_labels])


aligned = align_labels(true_labels, pred_labels)

fig, axes = plt.subplots(2, N_IMAGES, figsize=(20, 4))
fig.suptitle(f'Top: ground truth    Bottom: predicted clusters    ARI={metrics["ari"]:.3f}', fontsize=11)

for img_idx in range(N_IMAGES):
    true_frags = fragments[true_labels == img_idx]
    axes[0, img_idx].imshow(np.clip(fragments_to_image(true_frags), 0, 1))
    axes[0, img_idx].axis('off')
    axes[0, img_idx].set_title(f'img {img_idx}', fontsize=8)

    pred_frags = fragments[aligned == img_idx]
    if len(pred_frags) == GRID * GRID:
        axes[1, img_idx].imshow(np.clip(fragments_to_image(pred_frags), 0, 1))
    else:
        axes[1, img_idx].text(0.5, 0.5, f'{len(pred_frags)} frags',
                              ha='center', va='center', transform=axes[1, img_idx].transAxes)
    axes[1, img_idx].axis('off')
    axes[1, img_idx].set_title(f'pred {img_idx}', fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'clustering_visualisation.png')
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
