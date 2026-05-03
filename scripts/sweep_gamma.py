"""
Power-transformation experiment: try various exponents gamma on the
affinity matrix before spectral clustering. The model is not retrained;
this is purely a post-hoc test of whether sharpening the edge weights
improves the downstream clustering.

For each gamma, the affinity matrix is replaced by

    w_ij' = w_ij ** gamma

before being passed to balanced spectral clustering. Larger gamma
suppresses weak/moderate edges more aggressively; gamma=1 is the
baseline (no transformation).

Run from the project root:
    python scripts/sweep_gamma.py
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
GAMMAS     = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]


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


def main():
    run = find_best_run()
    print(f"Best run: {run['run']}  (best ARI = {run['best_ari']})")
    model, cfg = load_model(run)

    np.random.seed(42)
    ds  = Imagenet64(cfg['data_path'])
    gen = ds.datagen_cls(batch_size=cfg['n_images'], ds='test', augmentation=False)

    # collect a fixed pool of (similarity, labels) pairs once, so every gamma
    # is tested on exactly the same batches — a paired comparison
    print(f"\nCollecting {N_BATCHES} validation batches...")
    pool = []
    for _ in range(N_BATCHES):
        images, _ = next(gen)
        fragments, labels = extract_fragments(np.array(images))
        sim = model.get_output(fragments)            # symmetric, with zero diagonal
        pool.append((sim, labels))
    print(f"  done. similarity matrix range across pool: "
          f"[{min(s.min() for s, _ in pool):.3f}, {max(s.max() for s, _ in pool):.3f}]")

    # sweep — keep per-batch ARIs so we can compute paired differences
    print(f"\n{'gamma':>6}  {'mean ARI':>9}  {'cross-batch std':>15}  {'mean NMI':>9}")
    per_gamma_aris = {}
    rows = []
    for g in GAMMAS:
        aris, nmis = [], []
        for sim, labels in pool:
            transformed = np.clip(sim, 0, 1) ** g
            np.fill_diagonal(transformed, 0)
            preds = cluster(transformed, n_per_cluster=GRID*GRID)
            m = compute_metrics(preds, labels)
            aris.append(m['ari']); nmis.append(m['nmi'])
        per_gamma_aris[g] = aris
        mean_ari = statistics.mean(aris)
        std_ari  = statistics.stdev(aris) if len(aris) > 1 else 0
        mean_nmi = statistics.mean(nmis)
        print(f"{g:>6.1f}  {mean_ari:>9.4f}  {std_ari:>15.4f}  {mean_nmi:>9.4f}")
        rows.append((g, mean_ari, std_ari, mean_nmi))

    # paired comparison: for each gamma, look at the per-batch difference vs gamma=1
    print(f"\nPaired comparison vs gamma=1.0 (same batches, so deltas are paired):")
    print(f"{'gamma':>6}  {'mean delta':>10}  {'SE(delta)':>10}  {'t':>6}  signif?")
    base = per_gamma_aris[1.0]
    for g, aris in per_gamma_aris.items():
        if g == 1.0:
            continue
        deltas = [a - b for a, b in zip(aris, base)]
        mean_d = statistics.mean(deltas)
        std_d  = statistics.stdev(deltas)
        se_d   = std_d / (len(deltas) ** 0.5)
        t      = mean_d / se_d if se_d > 0 else float('nan')
        sig    = '***' if abs(t) > 3 else '**' if abs(t) > 2 else '*' if abs(t) > 1.5 else ''
        print(f"{g:>6.1f}  {mean_d:>+10.4f}  {se_d:>10.4f}  {t:>+6.2f}  {sig}")

    best = max(rows, key=lambda r: r[1])
    baseline = next(r for r in rows if r[0] == 1.0)
    print(f"\nBaseline (gamma=1.0): ARI = {baseline[1]:.4f}")
    print(f"Best   gamma = {best[0]}: ARI = {best[1]:.4f}  (delta = {best[1] - baseline[1]:+.4f})")


if __name__ == '__main__':
    main()
