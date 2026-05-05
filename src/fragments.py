"""
Fragment extraction and adjacency label construction.

Each 64×64 image is divided into a 4×4 grid of 16×16 patches called fragments.
This module provides utilities to perform that slicing and to build the binary
adjacency matrix that marks which fragment pairs are spatially neighbouring
within the same source image.

Constants:
    GRID          -- Number of fragments per row/column (4).
    FRAGMENT_SIZE -- Pixel size of each fragment (16).

Functions:
    extract_fragments -- Slice a batch of images into fragments and return labels.
    build_adjacency   -- Build the ground-truth adjacency matrix for a batch.
"""
import numpy as np

GRID = 4
FRAGMENT_SIZE = 16  # 64 // 4


def extract_fragments(images):
    """
    Slice a batch of images into 16×16 fragments and return source-image labels.

    Each image is divided into a GRID×GRID grid of non-overlapping patches.
    The fragments are returned in row-major order: all 16 fragments of image 0
    first, then all 16 of image 1, and so on.

    Args:
        images (np.ndarray): Batch of images with shape (N, 64, 64, 3),
            values expected in [0, 1].

    Returns:
        tuple:
            fragments (np.ndarray): Array of shape (N*16, 16, 16, 3) containing
                all extracted patches.
            labels (np.ndarray): Integer array of shape (N*16,) where each entry
                is the index of the source image the fragment came from.
    """
    fragments, labels = [], []
    for i, img in enumerate(images):
        for row in range(GRID):
            for col in range(GRID):
                r, c = row * FRAGMENT_SIZE, col * FRAGMENT_SIZE
                fragments.append(img[r:r + FRAGMENT_SIZE, c:c + FRAGMENT_SIZE])
                labels.append(i)
    return np.array(fragments), np.array(labels)


def build_adjacency(n_images, grid=GRID):
    """
    Build the ground-truth binary adjacency matrix for a batch of images.

    Two fragments are considered adjacent if they share an edge within the same
    source image (horizontal or vertical neighbours only; no diagonals).
    Cross-image pairs are always 0.

    Args:
        n_images (int): Number of images in the batch.
        grid (int): Grid size; each image has grid*grid fragments. Default GRID (4).

    Returns:
        np.ndarray: Symmetric float32 matrix of shape
            (n_images*grid*grid, n_images*grid*grid), where entry (i, j) = 1
            if fragments i and j are spatially adjacent, 0 otherwise.
    """
    n = n_images * grid * grid
    adj = np.zeros((n, n), dtype=np.float32)
    for img_idx in range(n_images):
        base = img_idx * grid * grid
        for row in range(grid):
            for col in range(grid):
                idx = base + row * grid + col
                if col + 1 < grid:
                    neighbor = base + row * grid + (col + 1)
                    adj[idx, neighbor] = adj[neighbor, idx] = 1
                if row + 1 < grid:
                    neighbor = base + (row + 1) * grid + col
                    adj[idx, neighbor] = adj[neighbor, idx] = 1
    return adj
