"""
Shared utilities for the diagnostics package.

Provides a data-loading helper and matplotlib convenience functions used by
debug_model.py, debug_pipeline.py, and plot_training.py.

Constants:
    DATA_PATH -- Path to the ImageNet-64 dataset directory.
    N_IMAGES  -- Default batch size in images (10).
"""
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
    """
    Draw one batch from the ImageNet-64 dataset and extract fragments.

    Args:
        ds (str): Dataset split to use — 'train' or 'test'. Default 'train'.
        augmentation (bool): Whether to apply data augmentation. Default True.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            images    -- Source images, shape (N_IMAGES, 64, 64, 3).
            fragments -- Extracted patches, shape (N_IMAGES*16, 16, 16, 3).
            labels    -- Source-image index per fragment, shape (N_IMAGES*16,).
            adjacency -- Binary adjacency matrix, shape (N_IMAGES*16, N_IMAGES*16).
    """
    dataset = Imagenet64(DATA_PATH)
    gen     = dataset.datagen_cls(batch_size=N_IMAGES, ds=ds, augmentation=augmentation)
    images, _ = next(gen)
    images    = np.array(images)
    fragments, labels = extract_fragments(images)
    adjacency         = build_adjacency(n_images=N_IMAGES)
    return images, fragments, labels, adjacency


def plot_matrix(matrix, title, filepath, n_images=N_IMAGES):
    """
    Plot a square matrix as a heatmap and save to disk.

    Draws white dividing lines at every GRID*GRID boundary to visually
    separate the per-image blocks. Useful for inspecting adjacency and
    similarity matrices.

    Args:
        matrix (np.ndarray): Square matrix to visualise, shape (N, N).
        title (str): Figure title string.
        filepath (str): Absolute path where the PNG will be saved.
        n_images (int): Number of source images; controls where boundary
            lines are drawn. Default N_IMAGES (10).
    """
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
    """
    Save a figure to the diagnostics output folder as out_{name}.png.

    Args:
        name (str): Output filename stem; the file will be saved as
            diagnostics/out_{name}.png.
        fig (matplotlib.figure.Figure | None): Figure to save. If None,
            only the path is returned without saving.

    Returns:
        str: Absolute path of the output file.
    """
    path = os.path.join(os.path.dirname(__file__), f'out_{name}.png')
    if fig:
        fig.savefig(path, dpi=150)
    return path
