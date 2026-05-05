#!/bin/bash
# Full pipeline: train → evaluate → visualise
# Run from the project root: bash run.sh
#
# Both evaluation scripts default to runs/latest/model (set by training).
# Override with: CHECKPOINT_PATH=runs/<run_name>/model bash run.sh

set -e
cd "$(dirname "$0")"

source venv/bin/activate

# ── TRAINING ──────────────────────────────────────────────────────────────────
# Reads config.yaml; ~25 minutes on Apple M-series CPU.
# Key knobs: max_iterations (25000), patience (10), eval_every (250).
python main.py

# ── EVALUATION ────────────────────────────────────────────────────────────────
# Collects ARI/NMI/purity/F1 over 1000 test batches.
# Saves results to runs/latest/test_metrics.json.
python src/script1_metrics.py

# ── VISUALISATION ─────────────────────────────────────────────────────────────
# Clusters a single batch and saves src/clustering_visualisation.png.
python src/script2_visualise.py
