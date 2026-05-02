"""
Print a markdown comparison table of all runs in results.jsonl.
Optionally overlay the training curves of every run on a single figure.

Run from the project root:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --plot
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results.jsonl')
RUNS_DIR     = os.path.join(PROJECT_ROOT, 'runs')


def load_results():
    if not os.path.exists(RESULTS_PATH):
        print(f"No results.jsonl yet at {RESULTS_PATH}.")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return [json.loads(line) for line in f]


def print_table(results):
    cols = [
        ('run',                 'Run'),
        ('beta',                'beta'),
        ('best_ari',            'Best ARI'),
        ('iter_at_best',        'Iter@best'),
        ('final_auroc',         'Final AUROC'),
        ('final_auprc',         'Final AUPRC'),
        ('final_ari',           'Final ARI'),
        ('duration_min',        'Min'),
        ('notes',               'Notes'),
    ]
    widths = {key: max(len(label), *(len(str(r.get(key, ''))) for r in results))
              for key, label in cols}

    header = '| ' + ' | '.join(label.ljust(widths[key]) for key, label in cols) + ' |'
    sep    = '|-' + '-|-'.join('-' * widths[key] for key, _ in cols) + '-|'
    print(header)
    print(sep)
    for r in results:
        row = '| ' + ' | '.join(str(r.get(key, '')).ljust(widths[key]) for key, _ in cols) + ' |'
        print(row)


def plot_curves(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Training curves across runs', fontsize=12)

    for r in results:
        log = os.path.join(RUNS_DIR, r['run'], 'training_log.jsonl')
        if not os.path.exists(log):
            continue
        with open(log) as f:
            entries = [json.loads(line) for line in f]
        iters  = [e['iteration']      for e in entries]
        ari    = [e['ari']            for e in entries]
        auprc  = [e.get('auprc', e.get('f1')) for e in entries]
        axes[0].plot(iters, ari,   label=r['run'])
        axes[1].plot(iters, auprc, label=r['run'])

    axes[0].set_ylabel('ARI');  axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8)
    axes[0].set_title('Adjusted Rand Index'); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('AUPRC'); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)
    axes[1].set_title('Adjacency AUPRC (threshold-free)'); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Iteration')

    out = os.path.join(PROJECT_ROOT, 'runs_comparison.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='also save runs_comparison.png')
    args = parser.parse_args()

    results = load_results()
    print_table(results)
    if args.plot:
        plot_curves(results)


if __name__ == '__main__':
    main()
