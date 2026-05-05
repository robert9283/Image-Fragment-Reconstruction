"""
Submission script 2: visualise the clustering result on a single test batch.

Loads the best checkpoint, draws one batch of N_IMAGES test images, runs the
model and balanced spectral clustering, then produces a two-row figure:

  Row 1 — ground-truth: fragments reassembled into their original source images.
  Row 2 — predicted:    fragments reassembled according to the model's clusters,
                        after Hungarian matching of cluster IDs to source IDs.

The figure is saved as clustering_visualisation.png in the src/ directory.
Metrics (ARI, NMI, purity) for this single batch are printed to stdout.

Usage:
    python src/script2_visualise.py

Override paths via environment variables if needed:
    DATA_PATH=path/to/data CHECKPOINT_PATH=runs/my_run/model python src/script2_visualise.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
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
DATA_PATH       = os.environ.get('DATA_PATH',
                    os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data'))
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH',
                    os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'model'))
# ─────────────────────────────────────────────────────────────────────────────


def load_model_from_checkpoint(checkpoint_path):
    """
    Read config.yaml from the run directory and instantiate a loaded model.

    Args:
        checkpoint_path (str): Path to the checkpoint without extension.

    Returns:
        tuple[FragmentAdjacencyPredictor, dict]: Loaded model and its config.
    """
    run_dir = os.path.dirname(os.path.realpath(checkpoint_path))
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)

    n_images = cfg.get('n_images', 10)
    n_pos    = n_images * 2 * GRID * (GRID - 1)
    n_pairs  = (n_images * GRID**2) * (n_images * GRID**2 - 1) // 2
    ratio    = (n_pairs - n_pos) / n_pos

    model = FragmentAdjacencyPredictor(
        pos_weight_adj = cfg.get('beta', 0.01923) * ratio,
        lambda_adj     = cfg.get('lambda_adj',  1.0),
        lambda_same    = cfg.get('lambda_same', 0.0),
    )
    model.load(checkpoint_path)
    return model, cfg


def run_clustering(model, cfg, data_path):
    """
    Draw one test batch and run the model and balanced spectral clustering.

    Args:
        model (FragmentAdjacencyPredictor): Loaded model.
        cfg (dict): Configuration dictionary (uses 'n_images').
        data_path (str): Path to the ImageNet-64 dataset directory.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
            fragments   -- (N, 16, 16, 3) fragment array.
            true_labels -- (N,) ground-truth source-image indices.
            pred_labels -- (N,) cluster assignments from balanced spectral clustering.
            metrics     -- {'ari': float, 'nmi': float, 'purity': float}.
    """
    n_images = cfg.get('n_images', 10)
    dataset  = Imagenet64(data_path)
    gen      = dataset.datagen_cls(batch_size=n_images, ds='test', augmentation=False)
    images, _ = next(gen)
    fragments, true_labels = extract_fragments(np.array(images))

    # balanced spectral clustering: enforce exactly GRID*GRID fragments per cluster
    model_output = model.get_output(fragments)
    pred_labels  = cluster(model_output, n_per_cluster=GRID * GRID)
    metrics      = compute_metrics(pred_labels, true_labels)
    return fragments, true_labels, pred_labels, metrics


def fragments_to_image(frags):
    """
    Reassemble 16 fragments in row-major order into a single 64×64 image.

    Args:
        frags (np.ndarray): Array of shape (16, 16, 16, 3) containing the
            GRID*GRID fragments of one image in row-major (top-left to
            bottom-right) order.

    Returns:
        np.ndarray: Reconstructed image of shape (64, 64, 3).
    """
    return (frags.reshape(GRID, GRID, FRAGMENT_SIZE, FRAGMENT_SIZE, 3)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(64, 64, 3))


def align_labels(true_labels, pred_labels, k=10):
    """
    Map predicted cluster IDs to true source-image IDs via Hungarian matching.

    Builds a cost matrix where cost[t, p] is the negative overlap between
    true class t and predicted cluster p, then finds the assignment that
    maximises total overlap.

    Args:
        true_labels (np.ndarray): Ground-truth source-image indices, shape (N,).
        pred_labels (np.ndarray): Predicted cluster assignments, shape (N,).
        k (int): Number of clusters / source images. Default 10.

    Returns:
        np.ndarray: Remapped predicted labels of shape (N,), aligned so that
            cluster IDs correspond to the best-matching true source-image IDs.
    """
    cost = np.zeros((k, k))
    for t in range(k):
        for p in range(k):
            cost[t, p] = -np.sum((true_labels == t) & (pred_labels == p))
    _, col_ind = linear_sum_assignment(cost)
    mapping = {pred: true for true, pred in enumerate(col_ind)}
    return np.array([mapping[p] for p in pred_labels])


def make_figure(fragments, true_labels, aligned_labels, metrics, n_images, out_path):
    """
    Produce and save the two-row visualisation figure.

    Row 1 shows each source image reconstructed from its ground-truth fragments.
    Row 2 shows the same reconstruction using the model's predicted cluster
    assignments (after Hungarian label alignment). Clusters that ended up with
    a non-standard number of fragments display a count instead of an image.

    Args:
        fragments (np.ndarray): All fragments, shape (N, 16, 16, 3).
        true_labels (np.ndarray): Ground-truth source-image indices, shape (N,).
        aligned_labels (np.ndarray): Hungarian-aligned predicted labels, shape (N,).
        metrics (dict): {'ari': float, 'nmi': float, 'purity': float}.
        n_images (int): Number of source images (= number of columns in the figure).
        out_path (str): File path where the PNG will be saved.
    """
    fig, axes = plt.subplots(2, n_images, figsize=(20, 4))
    fig.suptitle(
        f'Top: ground truth    Bottom: predicted clusters    ARI={metrics["ari"]:.3f}',
        fontsize=11,
    )

    for img_idx in range(n_images):
        # top row: ground-truth reconstruction
        true_frags = fragments[true_labels == img_idx]
        axes[0, img_idx].imshow(np.clip(fragments_to_image(true_frags), 0, 1))
        axes[0, img_idx].axis('off')
        axes[0, img_idx].set_title(f'img {img_idx}', fontsize=8)

        # bottom row: predicted reconstruction (fall back to text if wrong count)
        pred_frags = fragments[aligned_labels == img_idx]
        if len(pred_frags) == GRID * GRID:
            axes[1, img_idx].imshow(np.clip(fragments_to_image(pred_frags), 0, 1))
        else:
            axes[1, img_idx].text(0.5, 0.5, f'{len(pred_frags)} frags',
                                  ha='center', va='center',
                                  transform=axes[1, img_idx].transAxes)
        axes[1, img_idx].axis('off')
        axes[1, img_idx].set_title(f'pred {img_idx}', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    """
    Load the model, cluster one test batch, and save the visualisation figure.
    """
    model, cfg = load_model_from_checkpoint(CHECKPOINT_PATH)
    fragments, true_labels, pred_labels, metrics = run_clustering(model, cfg, DATA_PATH)

    print(f"ARI={metrics['ari']:.3f}  NMI={metrics['nmi']:.3f}  purity={metrics['purity']:.3f}")

    aligned  = align_labels(true_labels, pred_labels, k=cfg.get('n_images', 10))
    out_path = os.path.join(os.path.dirname(__file__), 'clustering_visualisation.png')
    make_figure(fragments, true_labels, aligned, metrics, cfg.get('n_images', 10), out_path)


if __name__ == '__main__':
    main()
