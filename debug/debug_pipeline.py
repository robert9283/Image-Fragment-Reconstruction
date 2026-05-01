"""
Debug script: visualise one training round — original images, their fragments,
and the ground truth adjacency matrix. Run from the project root:
    python debug/debug_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from debug.utils import load_batch, plot_matrix, GRID, FRAGMENT_SIZE, N_IMAGES

images, fragments, labels, adjacency = load_batch(augmentation=True)

print(f"images shape:    {images.shape}")
print(f"fragments shape: {fragments.shape}")
print(f"labels shape:    {labels.shape}")
print(f"adjacency shape: {adjacency.shape}")
print(f"adjacent pairs:  {int(adjacency.sum() // 2)}")

# ── original images ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_IMAGES, figsize=(20, 2.5))
fig.suptitle('Original images (augmented)', fontsize=11)
for i, ax in enumerate(axes):
    ax.imshow(np.clip(images[i], 0, 1))
    ax.set_title(f'img {i}', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_originals.png'), dpi=150)
plt.show()
plt.close()

# ── fragments for one image ───────────────────────────────────────────────────
IMG_IDX = 0
fig, axes = plt.subplots(GRID, GRID, figsize=(6, 6))
fig.suptitle(f'16 fragments of image {IMG_IDX}', fontsize=11)
for row in range(GRID):
    for col in range(GRID):
        frag_idx = IMG_IDX * GRID * GRID + row * GRID + col
        axes[row, col].imshow(np.clip(fragments[frag_idx], 0, 1))
        axes[row, col].set_title(f'({row},{col})', fontsize=7)
        axes[row, col].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_fragments.png'), dpi=150)
plt.show()
plt.close()

# ── shuffled fragment pile ────────────────────────────────────────────────────
shuffled_idx = np.random.permutation(len(fragments))
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Mixed fragment pile (shuffled, first 32) — true source in title', fontsize=11)
for i, ax in enumerate(axes.flat):
    idx = shuffled_idx[i]
    ax.imshow(np.clip(fragments[idx], 0, 1))
    ax.set_title(f'src={labels[idx]}', fontsize=7)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_shuffled_pile.png'), dpi=150)
plt.show()
plt.close()

# ── ground truth adjacency matrix ────────────────────────────────────────────
plot_matrix(adjacency, 'Ground truth adjacency matrix\nwhite lines separate source images',
            os.path.join(os.path.dirname(__file__), 'out_adjacency.png'))

print("Saved: out_originals.png, out_fragments.png, out_shuffled_pile.png, out_adjacency.png")
