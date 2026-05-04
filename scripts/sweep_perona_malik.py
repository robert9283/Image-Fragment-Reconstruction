"""
Perona-Malik diffusion filter applied to the affinity matrix as a post-
processing step before balanced spectral clustering. The model is not
retrained; this is a pure post-hoc filter.

Algorithm (one outer pass):
    For each node i (visited in random order):
        For inner_step in 1..INNER_STEPS:
            For each edge w_ij of node i, update via Perona-Malik
            diffusion across i's other edges:

                w_ij' = w_ij + LAMBDA * (
                    sum_k g(|w_ik - w_ij|) * (w_ik - w_ij)
                  / sum_k g(|w_ik - w_ij|)
                )

            with g(s) = exp(-(s/K)^2). Edges of similar weight pull each
            other toward a common value (smoothing); edges of very
            different weight have negligible influence (edge-preserving).
    Symmetrise: W = (W + W.T) / 2

The procedure is repeated OUTER_PASSES times. INNER_STEPS, OUTER_PASSES,
LAMBDA, and the sweep over K are all configured at the top of this file.

Run from the project root:
    python scripts/sweep_perona_malik.py
"""
import os
import sys
import json
import yaml

# =====================================================================
# Knobs — change these to experiment.
# =====================================================================
N_BATCHES    = 20         # validation batches for paired evaluation
OUTER_PASSES = 50         # number of full passes through all nodes
INNER_STEPS  = 2          # P-M update steps applied to each node's edges
LAMBDA       = 0.25       # step size
KS           = [0.10]     # diffusivity scales swept
SEED         = 42         # for reproducibility of the random node order
MODE         = 'sharpening'   # 'smoothing' (standard P-M) or
                              # 'sharpening' (anti-diffusion: flip the
                              # update sign so similar edges push each
                              # other AWAY from the local mean)
CLIP         = True        # clip each step's output to [0, 1] for stability
# =====================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'to_share', 'src'))

import numpy as np
import statistics

from data import Imagenet64
from src.fragments import extract_fragments, GRID
from src.clustering import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

RESULTS = os.path.join(PROJECT_ROOT, 'results.jsonl')


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


def perona_malik_update(W, K, rng):
    """One outer pass: visit every node in random order, apply
    INNER_STEPS Perona-Malik updates to that node's edge vector,
    then symmetrise."""
    n = W.shape[0]
    order = rng.permutation(n)

    for i in order:
        # f[k] = current edge weight from node i to node k
        f = W[i].copy()
        f[i] = 0.0  # diagonal stays 0; never includes self

        for _ in range(INNER_STEPS):
            # diff[j, k] = f[k] - f[j]   shape (n, n)
            diff = f[None, :] - f[:, None]
            weight = np.exp(-(diff / K) ** 2)
            # exclude self (j == i and k == i) from the averaging
            weight[:, i] = 0.0
            weight[i, :] = 0.0
            np.fill_diagonal(weight, 0.0)

            denom = weight.sum(axis=1) + 1e-12
            update = (weight * diff).sum(axis=1) / denom
            sign = +1 if MODE == 'smoothing' else -1
            f = f + sign * LAMBDA * update
            if CLIP:
                f = np.clip(f, 0.0, 1.0)
            f[i] = 0.0  # keep diagonal entry zero

        W[i] = f

    # symmetrise; updates were applied per-row asymmetrically
    return 0.5 * (W + W.T)


def apply_filter(sim, K, seed):
    """Apply OUTER_PASSES of Perona-Malik to a fresh copy of sim."""
    rng = np.random.default_rng(seed)
    W = sim.copy()
    np.fill_diagonal(W, 0.0)
    for _ in range(OUTER_PASSES):
        W = perona_malik_update(W, K, rng)
        np.fill_diagonal(W, 0.0)
    # clip back into [0, 1] for numerical sanity
    return np.clip(W, 0.0, 1.0)


def main():
    print(f"Knobs: mode={MODE}, outer_passes={OUTER_PASSES}, "
          f"inner_steps={INNER_STEPS}, lambda={LAMBDA}, K sweep={KS}")
    run = find_best_run()
    print(f"Best run: {run['run']}  (best ARI = {run['best_ari']})\n")
    model, cfg = load_model(run)

    np.random.seed(SEED)
    ds  = Imagenet64(cfg['data_path'])
    gen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)

    print(f"Collecting {N_BATCHES} validation batches...")
    pool = []
    for _ in range(N_BATCHES):
        images, _ = next(gen)
        fragments, labels = extract_fragments(np.array(images))
        sim = model.get_output(fragments)
        pool.append((sim, labels))

    # baseline (no filter)
    base_aris = []
    for sim, labels in pool:
        s = sim.copy(); np.fill_diagonal(s, 0)
        preds = cluster(s, n_per_cluster=GRID*GRID)
        base_aris.append(compute_metrics(preds, labels)['ari'])
    base_mean = statistics.mean(base_aris)
    print(f"Baseline (no filter) mean ARI = {base_mean:.4f}\n")

    # sweep
    print(f"{'K':>6}  {'mean ARI':>9}  {'std':>7}  {'mean delta':>10}")
    per_K_aris = {}
    for K in KS:
        aris = []
        for batch_idx, (sim, labels) in enumerate(pool):
            filtered = apply_filter(sim, K, seed=SEED + batch_idx)
            preds = cluster(filtered, n_per_cluster=GRID*GRID)
            aris.append(compute_metrics(preds, labels)['ari'])
        per_K_aris[K] = aris
        m = statistics.mean(aris)
        s = statistics.stdev(aris)
        d = m - base_mean
        print(f"{K:>6.2f}  {m:>9.4f}  {s:>7.4f}  {d:>+10.4f}")

    # paired comparison
    print(f"\nPaired comparison vs unfiltered baseline (n={N_BATCHES} batches):")
    print(f"{'K':>6}  {'mean delta':>10}  {'SE(delta)':>10}  {'t':>6}  signif?")
    for K, aris in per_K_aris.items():
        deltas = [a - b for a, b in zip(aris, base_aris)]
        mean_d = statistics.mean(deltas)
        std_d  = statistics.stdev(deltas)
        se_d   = std_d / (len(deltas) ** 0.5)
        t      = mean_d / se_d if se_d > 0 else float('nan')
        sig    = '***' if abs(t) > 3 else '**' if abs(t) > 2 else '*' if abs(t) > 1.5 else ''
        print(f"{K:>6.2f}  {mean_d:>+10.4f}  {se_d:>10.4f}  {t:>+6.2f}  {sig}")

    best_K = max(per_K_aris, key=lambda K: statistics.mean(per_K_aris[K]))
    best_mean = statistics.mean(per_K_aris[best_K])
    print(f"\nBest K = {best_K}: ARI = {best_mean:.4f}  (delta = {best_mean - base_mean:+.4f})")


if __name__ == '__main__':
    main()
