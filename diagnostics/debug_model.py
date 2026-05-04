"""
Debug script: show predictions of an untrained model on one batch.
Visualises the predicted similarity matrix vs ground truth adjacency,
and highlights the top predicted pairs vs true adjacent pairs.
Run from the project root:
    python diagnostics/debug_model.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from debug.utils import load_batch, plot_matrix, GRID, N_IMAGES
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'model')

images, fragments, labels, adjacency = load_batch(augmentation=False)

model = FragmentAdjacencyPredictor()
if os.path.exists(CHECKPOINT + '.pt'):
    model.load(CHECKPOINT)
    model_label = 'trained'
    print("Loaded checkpoint.")
else:
    model_label = 'untrained'
    print("No checkpoint found — using untrained model.")
similarity = model.get_output(fragments)

print(f"similarity shape: {similarity.shape}")
print(f"similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
print(f"similarity mean:  {similarity.mean():.3f}")

# ── 1. ground truth vs predicted similarity matrices side by side ─────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, matrix, title in zip(axes,
                              [adjacency, similarity],
                              ['Ground truth adjacency', f'Predicted similarity ({model_label})']):
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('fragment index')
    ax.set_ylabel('fragment index')
    for i in range(1, N_IMAGES):
        ax.axhline(i * GRID * GRID - 0.5, color='white', linewidth=0.8)
        ax.axvline(i * GRID * GRID - 0.5, color='white', linewidth=0.8)
    plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_model_matrices.png'), dpi=150)
plt.show()
plt.close()

# ── 2. top-10 predicted pairs vs whether they are truly adjacent ──────────────
n = similarity.shape[0]
rows, cols = np.triu_indices(n, k=1)
scores     = similarity[rows, cols]
top_idx    = np.argsort(scores)[::-1][:10]

print(f"\nTop 10 predicted pairs ({model_label} model):")
print(f"{'rank':>4}  {'frag_i':>6}  {'frag_j':>6}  {'score':>6}  {'src_i':>5}  {'src_j':>5}  {'truly adjacent':>14}")
for rank, idx in enumerate(top_idx):
    i, j    = rows[idx], cols[idx]
    score   = scores[idx]
    truly_adj = bool(adjacency[i, j])
    same_src  = labels[i] == labels[j]
    print(f"{rank+1:>4}  {i:>6}  {j:>6}  {score:>6.3f}  {labels[i]:>5}  {labels[j]:>5}  {str(truly_adj):>14}  {'(same image)' if same_src else ''}")

# ── 3. visualise the top-6 predicted pairs ───────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle(f'Top 6 predicted pairs ({model_label}) — green border = truly adjacent', fontsize=11)
for col, idx in enumerate(top_idx[:6]):
    i, j      = rows[idx], cols[idx]
    truly_adj = bool(adjacency[i, j])
    color     = 'green' if truly_adj else 'red'
    for row, frag_idx in enumerate([i, j]):
        ax = axes[row, col]
        ax.imshow(np.clip(fragments[frag_idx], 0, 1))
        ax.set_title(f'frag {frag_idx} (src {labels[frag_idx]})', fontsize=7)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_model_top_pairs.png'), dpi=150)
plt.show()
plt.close()

print("\nSaved: out_model_matrices.png, out_model_top_pairs.png")
