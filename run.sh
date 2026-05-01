#!/bin/bash
# Full pipeline: train → evaluate → visualise
# Run from the project root: bash run.sh

set -e
cd "$(dirname "$0")"

source venv/bin/activate

# ── TRAINING ──────────────────────────────────────────────────────────────────
# Current config (config.yaml) runs ~45 minutes.
# For a full training run (~14 hours), change config.yaml:
#   max_iterations: 500000
#   eval_every:     1000
#   patience:       20
python main.py

# ── EVALUATION ────────────────────────────────────────────────────────────────
# Runs the trained model over 1000 validation samples and reports ARI/NMI/purity.
python src/script1_metrics.py

# ── VISUALISATION ─────────────────────────────────────────────────────────────
# Runs the trained model on a single sample and saves clustering_visualisation.png.
python src/script2_visualise.py
