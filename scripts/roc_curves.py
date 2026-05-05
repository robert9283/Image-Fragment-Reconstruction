"""
Threshold-free curves (ROC and Precision-Recall) for the adjacency-prediction
head of the current best checkpoint. Auto-selects the run with the highest
best_ari from results.jsonl whose checkpoint is on disk, runs it on 10 fresh
validation batches, and writes:

    doc/roc_curves.png        — two-panel figure (ROC on the left, PR on the right)
    doc/fig_roc_curves.tex    — \\input-able tex with AUROC/AUPRC and run name
                                in the caption

Run from the project root:
    python scripts/roc_curves.py
"""
import os
import sys
import json
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'to_share', 'src'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

DOC_DIR  = os.path.join(PROJECT_ROOT, 'doc')
FIG_DIR  = os.path.join(DOC_DIR, 'figures')
PNG_PATH = os.path.join(FIG_DIR, 'roc_curves.png')
TEX_PATH = os.path.join(FIG_DIR, 'fig_roc_curves.tex')
RESULTS  = os.path.join(PROJECT_ROOT, 'results.jsonl')

N_BATCHES = 10


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

    Reads config.yaml stored alongside the checkpoint to reconstruct the exact
    hyperparameters used during training, then loads the saved weights.

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


def collect_scores(model, cfg, n_batches):
    """
    Collect per-pair adjacency scores and ground-truth labels from the model.

    Runs the adjacency head on n_batches fresh validation batches. For each
    batch, all upper-triangular fragment pairs are scored; results are
    concatenated across batches.

    Args:
        model (FragmentAdjacencyPredictor): Loaded model in eval mode.
        cfg (dict): Run config (uses 'data_path', 'n_images').
        n_batches (int): Number of validation batches to aggregate.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            probs   -- Predicted adjacency scores, shape (n_pairs,).
            targets -- Ground-truth adjacency labels (0 or 1), shape (n_pairs,).
    """
    np.random.seed(42)
    ds  = Imagenet64(cfg['data_path'])
    gen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)

    all_probs, all_targets = [], []
    model.encoder.eval(); model.adj_head.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            images, _ = next(gen)
            fragments, _ = extract_fragments(np.array(images))
            adjacency = build_adjacency(n_images=cfg['n_images'])

            x = model._to_tensor(fragments)
            embeddings = model.encoder(x)
            n = embeddings.shape[0]
            idx_i, idx_j = torch.triu_indices(n, n, offset=1)
            probs = model.adj_head(embeddings[idx_i], embeddings[idx_j]).cpu().numpy()
            targets = adjacency[idx_i.cpu(), idx_j.cpu()]

            all_probs.append(probs)
            all_targets.append(targets)

    return np.concatenate(all_probs), np.concatenate(all_targets)


def make_figure(probs, targets, run_summary):
    """
    Plot ROC and Precision-Recall curves and save the figure to doc/figures/.

    The left panel shows the ROC curve with the random diagonal baseline; the
    right panel shows the Precision-Recall curve with the random (positive-rate)
    baseline. Both AUROC and AUPRC are annotated in the legend.

    Args:
        probs (np.ndarray): Predicted adjacency scores, shape (n_pairs,).
        targets (np.ndarray): Ground-truth adjacency labels, shape (n_pairs,).
        run_summary (dict): Results entry for the best run (used in the title).

    Returns:
        tuple[float, float, float]: auroc, auprc, pos_rate.
    """
    auroc = roc_auc_score(targets, probs)
    auprc = average_precision_score(targets, probs)
    pos_rate = targets.mean()

    fpr, tpr, _ = roc_curve(targets, probs)
    precision, recall, _ = precision_recall_curve(targets, probs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color='steelblue', linewidth=1.5,
            label=f'model (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linewidth=0.6, linestyle='--',
            label='random (AUROC = 0.5)')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Precision-Recall
    ax = axes[1]
    ax.plot(recall, precision, color='darkorange', linewidth=1.5,
            label=f'model (AUPRC = {auprc:.3f})')
    ax.axhline(pos_rate, color='gray', linewidth=0.6, linestyle='--',
               label=f'random (AUPRC = {pos_rate:.3f})')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision--Recall')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PNG_PATH}")
    return auroc, auprc, pos_rate


def write_tex(run_summary, cfg, auroc, auprc, pos_rate, n_batches, n_pairs):
    """
    Write an \input-able LaTeX figure environment for the ROC/PR-curve figure.

    The caption includes hyperparameters, AUROC, AUPRC, the number of batches
    aggregated, and the total pair count.

    Args:
        run_summary (dict): Results entry for the best run (provides run name).
        cfg (dict): Run config (provides beta, lambda_adj, lambda_same).
        auroc (float): Area under the ROC curve.
        auprc (float): Area under the Precision-Recall curve.
        pos_rate (float): Fraction of positive (adjacent) pairs; used as the
            random AUPRC baseline.
        n_batches (int): Number of validation batches aggregated.
        n_pairs (int): Total number of fragment pairs evaluated.
    """
    safe_name = run_summary['run'].replace('_', r'\_')
    caption = (
        f"Threshold-free adjacency-prediction curves for the best checkpoint "
        f"(\\texttt{{{safe_name}}}, "
        f"$\\beta = {cfg.get('beta', 0.01923)}$, "
        f"$\\lambda_{{\\mathrm{{adj}}}} = {cfg.get('lambda_adj', 1.0)}$, "
        f"$\\lambda_{{\\mathrm{{same}}}} = {cfg.get('lambda_same', 0.0)}$). "
        f"Curves aggregated over {n_batches} fresh validation batches "
        f"({n_pairs:,} pairs in total, with positive rate {pos_rate:.4f}). "
        f"AUROC = {auroc:.3f} and AUPRC = {auprc:.3f}; the AUPRC random baseline "
        f"is the positive rate ($\\approx {pos_rate:.3f}$, dashed grey line)."
    )
    tex = (
        "% Auto-generated by scripts/roc_curves.py.\n"
        "% Re-run that script after a new best checkpoint to refresh.\n"
        "\\begin{figure}[h!]\n"
        "\\centering\n"
        "\\includegraphics[width=0.95\\textwidth]{roc_curves.png}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{fig:roc-curves}\n"
        "\\end{figure}\n"
    )
    with open(TEX_PATH, 'w') as f:
        f.write(tex)
    print(f"Saved: {TEX_PATH}")


def main():
    """
    Auto-select the best checkpoint, collect adjacency scores over N_BATCHES
    validation batches, and write the ROC/PR figure and LaTeX snippet to
    doc/figures/.
    """
    run = find_best_run()
    print(f"Best run: {run['run']}  (best ARI = {run['best_ari']})")
    model, cfg = load_model(run)

    probs, targets = collect_scores(model, cfg, N_BATCHES)
    print(f"Aggregated {len(probs):,} pairs from {N_BATCHES} validation batches "
          f"(positive rate = {targets.mean():.4f})")

    auroc, auprc, pos_rate = make_figure(probs, targets, run)
    print(f"AUROC = {auroc:.4f}   AUPRC = {auprc:.4f}")

    write_tex(run, cfg, auroc, auprc, pos_rate, N_BATCHES, len(probs))


if __name__ == '__main__':
    main()
