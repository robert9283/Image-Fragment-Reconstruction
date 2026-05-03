"""
Top-k sparsification of the affinity matrix.

For each node, keep its top-k strongest edges and zero the rest. Two
symmetry rules are tried:

  - "or"  (standard k-NN graph): keep edge (i,j) if j is in i's top-k OR
                                  i is in j's top-k. Each node has >= k
                                  edges. Connectivity is guaranteed.
  - "and" (mutual k-NN graph):    keep edge (i,j) only if j is in i's
                                  top-k AND i is in j's top-k. Each node
                                  has at most k edges. More aggressive.

The model is not retrained; this is a post-hoc filter on the affinity
matrix before balanced spectral clustering.

Run from the project root:
    python scripts/sweep_topk.py
"""
import os
import sys
import json
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'to_share', 'src'))

import numpy as np
import statistics

from data import Imagenet64
from src.fragments import extract_fragments, GRID
from src.clustering import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

RESULTS    = os.path.join(PROJECT_ROOT, 'results.jsonl')
N_BATCHES  = 20
KS         = [5, 10, 15, 20, 25, 30, 40, 60]


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


def topk_mask(sim, k):
    """Per-row top-k mask (boolean, n x n). Diagonal excluded."""
    n = sim.shape[0]
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)               # never pick self
    # argpartition gives indices of top-k per row in arbitrary order
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    mask[rows, idx.ravel()] = True
    return mask


def main():
    run = find_best_run()
    print(f"Best run: {run['run']}  (best ARI = {run['best_ari']})")
    model, cfg = load_model(run)

    np.random.seed(42)
    ds  = Imagenet64(cfg['data_path'])
    gen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)

    print(f"\nCollecting {N_BATCHES} validation batches...")
    pool = []
    for _ in range(N_BATCHES):
        images, _ = next(gen)
        fragments, labels = extract_fragments(np.array(images))
        sim = model.get_output(fragments)
        pool.append((sim, labels))

    # baseline (no sparsification, k = full)
    print(f"\nBaseline (no sparsification): computing...")
    base_aris = []
    for sim, labels in pool:
        s = sim.copy(); np.fill_diagonal(s, 0)
        preds = cluster(s, n_per_cluster=GRID*GRID)
        base_aris.append(compute_metrics(preds, labels)['ari'])
    base_mean = statistics.mean(base_aris)
    print(f"  baseline mean ARI = {base_mean:.4f}")

    # sweep
    print(f"\n{'rule':>6}  {'k':>3}  {'mean ARI':>9}  {'cross-batch std':>15}  "
          f"{'mean delta vs baseline':>22}")
    per_config_aris = {}
    for rule in ['or', 'and']:
        for k in KS:
            aris = []
            for sim, labels in pool:
                mask = topk_mask(sim, k)
                if rule == 'or':
                    keep = mask | mask.T
                else:
                    keep = mask & mask.T
                transformed = sim * keep
                np.fill_diagonal(transformed, 0)
                # safety: if any row is all-zero, the spectral embedding will
                # complain. Add a tiny epsilon to avoid that pathology.
                if (transformed.sum(axis=1) == 0).any():
                    transformed = transformed + 1e-9
                    np.fill_diagonal(transformed, 0)
                preds = cluster(transformed, n_per_cluster=GRID*GRID)
                aris.append(compute_metrics(preds, labels)['ari'])
            per_config_aris[(rule, k)] = aris
            mean_ari = statistics.mean(aris)
            std_ari  = statistics.stdev(aris)
            deltas   = [a - b for a, b in zip(aris, base_aris)]
            mean_d   = statistics.mean(deltas)
            print(f"{rule:>6}  {k:>3}  {mean_ari:>9.4f}  {std_ari:>15.4f}  {mean_d:>+22.4f}")

    # paired comparison vs baseline
    print(f"\nPaired comparison vs full-dense baseline:")
    print(f"{'rule':>6}  {'k':>3}  {'mean delta':>10}  {'SE(delta)':>10}  {'t':>6}  signif?")
    for (rule, k), aris in per_config_aris.items():
        deltas = [a - b for a, b in zip(aris, base_aris)]
        mean_d = statistics.mean(deltas)
        std_d  = statistics.stdev(deltas)
        se_d   = std_d / (len(deltas) ** 0.5)
        t      = mean_d / se_d if se_d > 0 else float('nan')
        sig    = '***' if abs(t) > 3 else '**' if abs(t) > 2 else '*' if abs(t) > 1.5 else ''
        print(f"{rule:>6}  {k:>3}  {mean_d:>+10.4f}  {se_d:>10.4f}  {t:>+6.2f}  {sig}")

    best = max(per_config_aris.items(), key=lambda kv: statistics.mean(kv[1]))
    rule, k = best[0]
    mean = statistics.mean(best[1])
    print(f"\nBest configuration: rule={rule}, k={k}, ARI = {mean:.4f}  "
          f"(delta = {mean - base_mean:+.4f})")


if __name__ == '__main__':
    main()
