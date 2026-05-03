"""
Distribution of the model's per-pair confidence scores, broken down by
the true class of each pair (adjacent / same-image but not adjacent /
cross-image). Auto-selects the best checkpoint from results.jsonl,
runs it on 10 fresh validation batches, and writes:

    doc/weight_distribution.png       — three overlaid histograms
    doc/fig_weight_distribution.tex   — \\input-able figure environment

Run from the project root:
    python scripts/weight_distribution.py
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

from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

DOC_DIR  = os.path.join(PROJECT_ROOT, 'doc')
PNG_PATH = os.path.join(DOC_DIR, 'weight_distribution.png')
TEX_PATH = os.path.join(DOC_DIR, 'fig_weight_distribution.tex')
RESULTS  = os.path.join(PROJECT_ROOT, 'results.jsonl')

N_BATCHES = 10
N_BINS    = 60


def n_neg_over_n_pos(n_images, grid):
    n_pos = n_images * (2 * grid * (grid - 1))
    n_pairs = (n_images * grid * grid) * (n_images * grid * grid - 1) // 2
    return (n_pairs - n_pos) / n_pos


def find_best_run():
    with open(RESULTS) as f:
        runs = [json.loads(line) for line in f]
    runs_with_ckpt = [r for r in runs
                      if os.path.exists(os.path.join(PROJECT_ROOT, 'runs', r['run'], 'model.pt'))]
    return max(runs_with_ckpt, key=lambda r: r['best_ari'])


def load_model(run_summary):
    run_dir = os.path.join(PROJECT_ROOT, 'runs', run_summary['run'])
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    beta = float(cfg.get('beta', 0.01923))
    ratio = n_neg_over_n_pos(cfg['n_images'], GRID)
    model = FragmentAdjacencyPredictor(
        pos_weight_adj  = beta * ratio,
        lambda_adj      = float(cfg.get('lambda_adj',  1.0)),
        pos_weight_same = float(cfg.get('pos_weight_same', 1.0)),
        lambda_same     = float(cfg.get('lambda_same', 0.0)),
    )
    model.load(os.path.join(run_dir, 'model'))
    return model, cfg


def categorise_pairs(labels, adjacency, idx_i, idx_j):
    """Return three boolean masks (adjacent, same_not_adj, cross) covering
    every (i, j) pair."""
    same_image = labels[idx_i] == labels[idx_j]
    adjacent = adjacency[idx_i, idx_j].astype(bool)
    same_not_adj = same_image & (~adjacent)
    cross = ~same_image
    return adjacent, same_not_adj, cross


def main():
    run = find_best_run()
    print(f"Best run: {run['run']}  (best ARI = {run['best_ari']})")
    model, cfg = load_model(run)

    np.random.seed(42)
    ds  = Imagenet64(cfg['data_path'])
    gen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)

    adj_scores, same_scores, cross_scores = [], [], []
    print(f"Running on {N_BATCHES} validation batches...")
    for _ in range(N_BATCHES):
        images, _ = next(gen)
        fragments, labels = extract_fragments(np.array(images))
        adjacency = build_adjacency(n_images=cfg['n_images'])
        sim = model.get_output(fragments)

        n = sim.shape[0]
        idx_i, idx_j = np.triu_indices(n, k=1)
        scores = sim[idx_i, idx_j]
        m_adj, m_same, m_cross = categorise_pairs(labels, adjacency, idx_i, idx_j)

        adj_scores.append(scores[m_adj])
        same_scores.append(scores[m_same])
        cross_scores.append(scores[m_cross])

    adj_scores   = np.concatenate(adj_scores)
    same_scores  = np.concatenate(same_scores)
    cross_scores = np.concatenate(cross_scores)
    print(f"  pairs: adjacent={len(adj_scores)},  "
          f"same-image (not adjacent)={len(same_scores)},  "
          f"cross-image={len(cross_scores)}")

    # plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, N_BINS + 1)

    # cross-image as grey filled (the "background")
    ax.hist(cross_scores, bins=bins, density=False, color='gray', alpha=0.4,
            label=f'cross-image  ({len(cross_scores):,} pairs)')

    # same-image-not-adjacent as orange step
    ax.hist(same_scores, bins=bins, density=False, color='darkorange', alpha=0.7,
            histtype='stepfilled', label=f'same image, not adjacent  ({len(same_scores):,})')

    # adjacent as blue step
    ax.hist(adj_scores, bins=bins, density=False, color='steelblue', alpha=0.85,
            histtype='stepfilled',
            label=f'adjacent  ({len(adj_scores):,})')

    ax.set_yscale('log')
    ax.set_xlabel('predicted confidence  $p_{ij}$')
    ax.set_ylabel('count (log scale)')
    ax.set_title(f"Distribution of pair-confidence scores  "
                 f"(run: {run['run']})")
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PNG_PATH}")

    # summary numbers
    print(f"\nMedian / 75th / 95th percentile per class:")
    for name, s in [('adjacent', adj_scores),
                    ('same-image (not adj)', same_scores),
                    ('cross-image', cross_scores)]:
        print(f"  {name:24s} : "
              f"median={np.median(s):.3f}, "
              f"75%={np.percentile(s, 75):.3f}, "
              f"95%={np.percentile(s, 95):.3f}")

    # write tex
    safe_name = run['run'].replace('_', r'\_')
    caption = (
        f"Distribution of the model's per-pair confidence scores "
        f"$p_{{ij}}$ on {N_BATCHES} fresh validation batches "
        f"(\\texttt{{{safe_name}}}, "
        f"$\\beta = {model.pos_weight_adj/52:.4f}$, "
        f"$\\lambda_{{\\mathrm{{adj}}}} = {model.lambda_adj}$, "
        f"$\\lambda_{{\\mathrm{{same}}}} = {model.lambda_same}$). "
        f"Pairs are split by their true class: adjacent (blue, "
        f"{len(adj_scores):,} pairs), "
        f"same source image but not adjacent (orange, {len(same_scores):,}), "
        f"and cross-image (grey, {len(cross_scores):,}). "
        f"y-axis is logarithmic to make the rare classes visible."
    )
    tex = (
        "% Auto-generated by scripts/weight_distribution.py.\n"
        "\\begin{figure}[h!]\n"
        "\\centering\n"
        "\\includegraphics[width=0.95\\textwidth]{weight_distribution.png}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{fig:weight-distribution}\n"
        "\\end{figure}\n"
    )
    with open(TEX_PATH, 'w') as f:
        f.write(tex)
    print(f"Saved: {TEX_PATH}")


if __name__ == '__main__':
    main()
