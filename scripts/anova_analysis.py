"""
Mean +/- SE per configuration group, plus pairwise Welch's t-tests, on the
seeded ANOVA-style runs in results.jsonl.

Three groups:
    adj_only_seed_*    (adjacency-only, lambda_same=0)
    noise_seed_*       (multi-task, lambda_same=1)
    same_only_seed_*   (same-image-only, lambda_adj=0)

Run from the project root:
    python scripts/anova_analysis.py
"""
import json
import os
import statistics
from itertools import combinations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results.jsonl')


def welch_t(x, y):
    """Two-sided Welch's t-test from raw samples. Returns (t, df, p)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan'), float('nan'), float('nan')
    mx, my = statistics.mean(x), statistics.mean(y)
    vx, vy = statistics.variance(x), statistics.variance(y)
    se = (vx / nx + vy / ny) ** 0.5
    if se == 0:
        return float('inf'), float('inf'), 0.0
    t = (mx - my) / se
    df = (vx / nx + vy / ny) ** 2 / (
        (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    )
    # two-sided p-value via t-distribution survival function
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), df)
    except ImportError:
        # crude normal approximation if scipy isn't installed
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, df, p


def group_runs(runs, prefix):
    return [r for r in runs if r['run'].startswith(prefix)]


def summarise(runs, label):
    aris   = [r['best_ari']               for r in runs]
    aurocs = [r.get('final_auroc') for r in runs if r.get('final_auroc') is not None]
    n = len(aris)
    if n == 0:
        return None
    mean = statistics.mean(aris)
    std  = statistics.stdev(aris) if n > 1 else float('nan')
    se   = std / (n ** 0.5)         if n > 1 else float('nan')
    return {
        'label': label, 'n': n, 'mean': mean, 'std': std, 'se': se,
        'aris': aris, 'aurocs': aurocs,
    }


def main():
    with open(RESULTS_PATH) as f:
        runs = [json.loads(line) for line in f]

    groups = {
        'adj_only (lambda_same=0)':  group_runs(runs, 'adj_only_seed_'),
        'multitask (lambda_same=1)': group_runs(runs, 'noise_seed_'),
        'same_only (lambda_adj=0)':  group_runs(runs, 'same_only_seed_'),
    }

    summaries = {label: summarise(g, label) for label, g in groups.items()}
    summaries = {k: v for k, v in summaries.items() if v is not None}

    if not summaries:
        print("No seeded ANOVA runs found in results.jsonl yet.")
        return

    # per-group summary
    print("=" * 78)
    print("Per-configuration summary (best ARI across seeds)")
    print("=" * 78)
    print(f"{'group':30s}  {'n':>3s}  {'mean':>7s}  {'std':>7s}  {'SE':>7s}  range")
    for s in summaries.values():
        rng = f"[{min(s['aris']):.3f}, {max(s['aris']):.3f}]"
        std = f"{s['std']:.4f}" if s['n'] > 1 else 'n/a'
        se  = f"{s['se']:.4f}"  if s['n'] > 1 else 'n/a'
        print(f"{s['label']:30s}  {s['n']:>3d}  {s['mean']:>7.4f}  "
              f"{std:>7s}  {se:>7s}  {rng}")

    # pairwise tests
    print()
    print("=" * 78)
    print("Pairwise Welch's t-tests on best ARI (two-sided, significance: p<0.05)")
    print("=" * 78)
    print(f"{'pair':55s}  {'mean diff':>10s}  {'t':>7s}  {'df':>5s}  {'p':>8s}  signif?")
    pairs = list(combinations(summaries.values(), 2))
    for a, b in pairs:
        if a['n'] < 2 or b['n'] < 2:
            continue
        diff = a['mean'] - b['mean']
        t, df, p = welch_t(a['aris'], b['aris'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        pair_label = f"{a['label']:25.25s} vs {b['label']:25.25s}"
        print(f"{pair_label:55s}  {diff:+10.4f}  {t:>7.2f}  {df:>5.1f}  {p:>8.4f}  {sig}")

    # AUROC sanity check
    print()
    print("=" * 78)
    print("AUROC stability across the same groups (sanity check)")
    print("=" * 78)
    print(f"{'group':30s}  {'n':>3s}  {'mean':>7s}  {'std':>7s}")
    for s in summaries.values():
        if s['aurocs']:
            mean = statistics.mean(s['aurocs'])
            std  = statistics.stdev(s['aurocs']) if len(s['aurocs']) > 1 else float('nan')
            std_s = f"{std:.4f}" if len(s['aurocs']) > 1 else 'n/a'
            print(f"{s['label']:30s}  {len(s['aurocs']):>3d}  {mean:>7.4f}  {std_s:>7s}")


if __name__ == '__main__':
    main()
