"""Shared utilities for debug scripts."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID, FRAGMENT_SIZE

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data')
N_IMAGES  = 10


def load_batch(ds='train', augmentation=True):
    """Returns (images, fragments, labels, adjacency) for one batch."""
    dataset = Imagenet64(DATA_PATH)
    gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds=ds, augmentation=augmentation)
    images, _ = next(gen)
    images    = np.array(images)
    fragments, labels = extract_fragments(images)
    adjacency         = build_adjacency(n_images=N_IMAGES)
    return images, fragments, labels, adjacency


def plot_matrix(matrix, title, filepath, n_images=N_IMAGES):
    """Plot a 160×160 matrix as a heatmap with image boundary lines."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('fragment index')
    ax.set_ylabel('fragment index')
    for i in range(1, n_images):
        ax.axhline(i * GRID * GRID - 0.5, color='white', linewidth=0.8)
        ax.axvline(i * GRID * GRID - 0.5, color='white', linewidth=0.8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.show()
    plt.close()


def save(name, fig=None):
    """Save figure to debug output folder."""
    path = os.path.join(os.path.dirname(__file__), f'out_{name}.png')
    if fig:
        fig.savefig(path, dpi=150)
    return path
