import numpy as np

GRID = 4
FRAGMENT_SIZE = 16  # 64 // 4


def extract_fragments(images):
    """
    images: (N, 64, 64, 3) float32 array
    Returns:
        fragments: (N*16, 16, 16, 3)
        labels:    (N*16,) source image index per fragment
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
    Returns (n_images*grid*grid, n_images*grid*grid) binary matrix.
    Entry (i, j) = 1 iff fragments i and j are spatially adjacent
    within the same source image.
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
