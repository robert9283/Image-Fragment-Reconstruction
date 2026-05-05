"""
Failure modes analysis. Auto-selects the run with the highest best_ari from
results.jsonl, loads its checkpoint, runs on a fresh validation batch, and
writes:

    doc/failure_modes.png         — six misclassified fragments visualised
    doc/fig_failure_modes.tex     — \\input-able tex with hyperparameters
                                     in the caption

Run from the project root:
    python scripts/failure_modes.py
"""
import os
import sys
import json
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'to_share', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from data import Imagenet64
from src.fragments import extract_fragments, GRID
from src.clustering import cluster
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

DOC_DIR    = os.path.join(PROJECT_ROOT, 'doc')
FIG_DIR    = os.path.join(DOC_DIR, 'figures')
PNG_PATH   = os.path.join(FIG_DIR, 'failure_modes.png')
TEX_PATH   = os.path.join(FIG_DIR, 'fig_failure_modes.tex')
RESULTS    = os.path.join(PROJECT_ROOT, 'results.jsonl')


def n_neg_over_n_pos(n_images, grid):
    """
    Compute the ratio of non-adjacent to adjacent fragment pairs.

    Used to scale the positive-class weight in the weighted BCE loss so that
    the loss contribution from rare adjacent pairs balances that of the
    majority non-adjacent pairs.

    Args:
        n_images (int): Number of images in the batch.
        grid (int): Grid size; each image contributes grid*grid fragments.

    Returns:
        float: (n_total_pairs - n_adjacent) / n_adjacent.
    """
    n_pos = n_images * (2 * grid * (grid - 1))
    n_pairs = (n_images * grid * grid) * (n_images * grid * grid - 1) // 2
    return (n_pairs - n_pos) / n_pos


def find_best_run():
    """
    Find the run with the highest best_ari that has a checkpoint on disk.

    Reads all entries from results.jsonl, filters to those whose model.pt
    file exists under runs/{run_name}/, and returns the one with the
    maximum best_ari value.

    Returns:
        dict: The results.jsonl entry for the best run.

    Raises:
        RuntimeError: If no run with a saved checkpoint is found.
    """
    with open(RESULTS) as f:
        runs = [json.loads(line) for line in f]

    def has_checkpoint(r):
        return os.path.exists(os.path.join(PROJECT_ROOT, 'runs', r['run'], 'model.pt'))

    runs_with_ckpt = [r for r in runs if has_checkpoint(r)]
    if not runs_with_ckpt:
        raise RuntimeError("no runs in results.jsonl have a saved checkpoint on disk")
    return max(runs_with_ckpt, key=lambda r: r['best_ari'])


def load_model(run_summary):
    """
    Load the FragmentAdjacencyPredictor from a run's checkpoint directory.

    Reads the config.yaml stored alongside the checkpoint to reconstruct the
    exact hyperparameters used during training, then loads the saved weights.

    Args:
        run_summary (dict): A results.jsonl entry containing at least a 'run'
            key matching a subdirectory of runs/.

    Returns:
        tuple[FragmentAdjacencyPredictor, dict]: Loaded model and its config.
    """
    run_dir = os.path.join(PROJECT_ROOT, 'runs', run_summary['run'])
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    beta = float(cfg.get('beta', 0.01923))
    ratio = n_neg_over_n_pos(cfg['n_images'], GRID)
    model = FragmentAdjacencyPredictor(
        pos_weight_adj = beta * ratio,
        lambda_adj     = float(cfg.get('lambda_adj',  1.0)),
        lambda_same    = float(cfg.get('lambda_same', 0.0)),
    )
    model.load(os.path.join(run_dir, 'model'))
    return model, cfg


def hungarian_match(pred, true, k=10):
    """
    Map predicted cluster IDs to true source IDs via Hungarian assignment.

    Builds a k×k cost matrix where cost[i, j] is the negative overlap between
    predicted cluster i and true class j, then finds the assignment maximising
    total overlap.

    Args:
        pred (np.ndarray): Predicted cluster assignments, shape (N,).
        true (np.ndarray): Ground-truth source-image indices, shape (N,).
        k (int): Number of clusters / source images. Default 10.

    Returns:
        np.ndarray: Remapped predicted labels of shape (N,), aligned so that
            each predicted cluster ID corresponds to the best-matching true
            source-image ID.
    """
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = -np.sum((pred == i) & (true == j))
    row, col = linear_sum_assignment(cost)
    mapping = dict(zip(row, col))
    return np.array([mapping[p] for p in pred])


def select_failures(misclassified, labels, n=6):
    """
    Select up to n misclassified fragments from distinct source images.

    Iterates over the misclassified indices and greedily picks one fragment
    per source image for visual diversity. If fewer than n distinct source
    images are represented, fills the remainder with any remaining
    misclassified fragments.

    Args:
        misclassified (np.ndarray): Indices of incorrectly clustered fragments.
        labels (np.ndarray): Ground-truth source-image index per fragment.
        n (int): Maximum number of fragments to select. Default 6.

    Returns:
        list[int]: Selected fragment indices (length ≤ n).
    """
    chosen, seen = [], set()
    for idx in misclassified:
        if labels[idx] not in seen:
            chosen.append(idx); seen.add(labels[idx])
            if len(chosen) == n:
                return chosen
    # not enough distinct sources — fill with remaining indices
    for idx in misclassified:
        if idx not in chosen:
            chosen.append(idx)
            if len(chosen) == n:
                break
    return chosen


def style_axis(ax, color):
    """
    Apply a coloured border to a matplotlib axis and remove tick marks.

    Used to visually distinguish the three columns of the failure-modes figure
    (red for misclassified fragment, green for true source, orange for
    predicted source).

    Args:
        ax (matplotlib.axes.Axes): The axis to style.
        color (str): Border colour string (e.g. 'red', 'green', 'orange').
    """
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])


def make_figure(images, fragments, labels, pred, pred_matched, run_summary, cfg):
    """
    Produce and save the failure-modes figure.

    Each row shows one misclassified fragment alongside its true source image
    (green border) and the source image the model wrongly assigned it to
    (orange border). Selects up to 4 failures from distinct source images for
    visual diversity.

    Args:
        images (np.ndarray): Batch of source images, shape (n_images, 64, 64, 3).
        fragments (np.ndarray): All extracted fragments, shape (N, 16, 16, 3).
        labels (np.ndarray): Ground-truth source-image index per fragment, shape (N,).
        pred (np.ndarray): Raw predicted cluster assignments, shape (N,).
        pred_matched (np.ndarray): Hungarian-aligned predicted labels, shape (N,).
        run_summary (dict): Results entry for the best run (used for printing).
        cfg (dict): Run config (not directly used in the figure body).

    Returns:
        int: Total number of misclassified fragments in this batch.
    """
    misclassified = np.where(pred_matched != labels)[0]
    print(f"Misclassified: {len(misclassified)} / {len(labels)} fragments")

    selected = select_failures(misclassified, labels, n=4)
    n_show = len(selected)

    fig, axes = plt.subplots(n_show, 3, figsize=(8, 2.4 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, frag_idx in enumerate(selected):
        true_src    = int(labels[frag_idx])
        pred_src    = int(pred_matched[frag_idx])

        # col 0: the misclassified fragment
        ax = axes[row, 0]
        ax.imshow(np.clip(fragments[frag_idx], 0, 1))
        ax.set_title(f'misclassified fragment', fontsize=8)
        style_axis(ax, 'red')

        # col 1: the true source image
        ax = axes[row, 1]
        ax.imshow(np.clip(images[true_src], 0, 1))
        ax.set_title(f'true source image (source {true_src})', fontsize=8)
        style_axis(ax, 'green')

        # col 2: the predicted source image (after Hungarian matching of
        # cluster IDs to source IDs, this is the source the model thought
        # the fragment came from)
        ax = axes[row, 2]
        ax.imshow(np.clip(images[pred_src], 0, 1))
        ax.set_title(f'predicted source image (source {pred_src})', fontsize=8)
        style_axis(ax, 'orange')

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PNG_PATH}")
    return len(misclassified)


def write_tex(run_summary, cfg, n_misclassified, total):
    """
    Write an \input-able LaTeX figure environment for the failure-modes figure.

    The generated .tex file includes a caption with hyperparameters, the
    best ARI, and the misclassification count. It references the PNG by
    filename so that the LaTeX document must include it via \graphicspath or
    an equivalent.

    Args:
        run_summary (dict): Results entry for the best run (provides run name
            and best_ari).
        cfg (dict): Run config (provides beta, lambda_adj, lambda_same).
        n_misclassified (int): Number of incorrectly clustered fragments.
        total (int): Total number of fragments in the batch.
    """
    safe_name = run_summary['run'].replace('_', r'\_')
    caption = (
        f"Selected misclassified fragments from the best checkpoint "
        f"(\\texttt{{{safe_name}}}, "
        f"best ARI = {run_summary['best_ari']:.3f}, "
        f"$\\beta = {cfg.get('beta', 0.01923)}$, "
        f"$\\lambda_{{\\mathrm{{adj}}}} = {cfg.get('lambda_adj', 1.0)}$, "
        f"$\\lambda_{{\\mathrm{{same}}}} = {cfg.get('lambda_same', 0.0)}$). "
        f"On the validation batch shown, "
        f"{n_misclassified} of {total} fragments were assigned to the wrong "
        f"cluster after Hungarian matching of predicted to true labels. "
        f"Each row shows: the misclassified fragment (left, red), the true "
        f"source image (middle, green), and the source image the model "
        f"assigned it to (right, orange)."
    )
    tex = (
        "% Auto-generated by scripts/failure_modes.py.\n"
        "% Re-run that script after a new best checkpoint to refresh.\n"
        "\\begin{figure}[h!]\n"
        "\\centering\n"
        "\\includegraphics[width=0.85\\textwidth]{failure_modes.png}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{fig:failure-modes}\n"
        "\\end{figure}\n"
    )
    with open(TEX_PATH, 'w') as f:
        f.write(tex)
    print(f"Saved: {TEX_PATH}")


def main():
    """
    Auto-select the best checkpoint, run clustering on one test batch,
    and write the failure-modes figure and LaTeX snippet to doc/figures/.
    """
    run = find_best_run()
    print(f"Best run: {run['run']}  (best ARI = {run['best_ari']})")
    model, cfg = load_model(run)

    np.random.seed(42)
    ds  = Imagenet64(cfg['data_path'])
    gen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)
    images, _ = next(gen)
    images = np.array(images)
    fragments, labels = extract_fragments(images)

    similarity = model.get_output(fragments)
    pred = cluster(similarity, n_per_cluster=GRID * GRID)
    pred_matched = hungarian_match(pred, labels)
    print(f"ARI on this batch: "
          f"{(pred_matched == labels).sum() / len(labels):.3f} cluster accuracy")

    n_miss = make_figure(images, fragments, labels, pred, pred_matched, run, cfg)
    write_tex(run, cfg, n_miss, len(labels))


if __name__ == '__main__':
    main()
