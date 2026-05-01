"""
Debug script: visualise one training round — original images, their fragments,
and the adjacency matrix. Run from the project root:
    python debug/debug_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID, FRAGMENT_SIZE

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data')
N_IMAGES  = 10

# ── 1. load one batch ─────────────────────────────────────────────────────────
dataset = Imagenet64(DATA_PATH)
gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds='train', augmentation=True)
images, _ = next(gen)
images = np.array(images)

fragments, labels = extract_fragments(images)
adjacency         = build_adjacency(n_images=N_IMAGES)

print(f"images shape:    {images.shape}")
print(f"fragments shape: {fragments.shape}")
print(f"labels shape:    {labels.shape}")
print(f"adjacency shape: {adjacency.shape}")
print(f"adjacent pairs:  {int(adjacency.sum() // 2)}")

# ── 2. original images ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_IMAGES, figsize=(20, 2.5))
fig.suptitle('Original images (augmented)', fontsize=11)
for i, ax in enumerate(axes):
    ax.imshow(np.clip(images[i], 0, 1))
    ax.set_title(f'img {i}', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_originals.png'), dpi=150)
plt.show()

# ── 3. fragments for one image ────────────────────────────────────────────────
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

# ── 4. shuffled fragment pile (first 32 fragments after shuffle) ──────────────
shuffled_idx = np.random.permutation(len(fragments))
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Mixed fragment pile (shuffled, first 32) — true source shown in title', fontsize=11)
for i, ax in enumerate(axes.flat):
    idx = shuffled_idx[i]
    ax.imshow(np.clip(fragments[idx], 0, 1))
    ax.set_title(f'src={labels[idx]}', fontsize=7)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_shuffled_pile.png'), dpi=150)
plt.show()

# ── 5. adjacency matrix heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(adjacency, cmap='Blues', aspect='auto')
ax.set_title('Adjacency matrix (160×160)\nwhite lines separate source images', fontsize=11)
ax.set_xlabel('fragment index')
ax.set_ylabel('fragment index')
for i in range(1, N_IMAGES):
    ax.axhline(i * GRID * GRID - 0.5, color='white', linewidth=0.8)
    ax.axvline(i * GRID * GRID - 0.5, color='white', linewidth=0.8)
plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'out_adjacency.png'), dpi=150)
plt.show()

print("Saved: out_originals.png, out_fragments.png, out_shuffled_pile.png, out_adjacency.png")
