"""
Plot training curves from training_log.jsonl.
Run from the project root:
    python diagnostics/plot_training.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'training_log.jsonl')

if not os.path.exists(LOG_PATH):
    print(f"No training log found at {LOG_PATH}. Run main.py first.")
    sys.exit(1)

with open(LOG_PATH) as f:
    entries = [json.loads(line) for line in f]

iterations = [e['iteration']  for e in entries]
loss       = [e['loss']       for e in entries]
auroc      = [e['auroc']      for e in entries]
auprc      = [e['auprc']      for e in entries]
ari        = [e['ari']        for e in entries]
nmi        = [e['nmi']        for e in entries]
purity     = [e['purity']     for e in entries]

has_timing   = 'timestamp' in entries[0]
x_values     = [e['timestamp'] / 60 for e in entries] if has_timing else iterations
x_label      = 'Time (minutes)' if has_timing else 'Iteration'

# ── training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Training curves', fontsize=13)

axes[0].plot(x_values, loss, color='steelblue')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_values, auroc, label='AUROC', color='darkorange')
axes[1].plot(x_values, auprc, label='AUPRC', color='green',  linestyle='--')
axes[1].axhline(0.019, color='gray', linewidth=0.6, linestyle=':',
                label='AUPRC random baseline (1.9%)')
axes[1].set_ylabel('Score')
axes[1].set_title('Adjacency prediction (threshold-free)')
axes[1].legend()
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x_values, ari,    label='ARI',    color='purple')
axes[2].plot(x_values, nmi,    label='NMI',    color='teal',   linestyle='--')
axes[2].plot(x_values, purity, label='Purity', color='brown',  linestyle='--')
axes[2].set_ylabel('Score')
axes[2].set_xlabel(x_label)
axes[2].set_title('Fragment clustering')
axes[2].legend()
axes[2].set_ylim(0, 1)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'out_training_curves.png')
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

# ── timing summary ────────────────────────────────────────────────────────────
if has_timing:
    train_times = [e['train_time_s'] for e in entries]
    eval_times  = [e['eval_time_s']  for e in entries]
    total_train = sum(train_times)
    total_eval  = sum(eval_times)
    overhead    = 100 * total_eval / (total_train + total_eval)
    total_mins  = x_values[-1]
    print(f"\nTiming summary:")
    print(f"  Total training time: {total_mins:.1f} minutes")
    print(f"  Avg train step:      {np.mean(train_times) / (entries[1]['iteration'] - entries[0]['iteration']) * 1000:.1f} ms")
    print(f"  Avg eval step:       {np.mean(eval_times) * 1000:.1f} ms")
    print(f"  Eval overhead:       {overhead:.1f}% of total time")

print(f"\nFinal   — loss={loss[-1]:.4f}  AUROC={auroc[-1]:.3f}  AUPRC={auprc[-1]:.3f}  ARI={ari[-1]:.3f}")
print(f"Best ARI={max(ari):.3f} at iteration {iterations[ari.index(max(ari))]}")
