#!/bin/bash
# ANOVA-style overnight experiment: 5 seeds each of two configurations
# we want to compare against the existing 5-seed multi-task (noise_seed_*)
# baseline.
#
# Configurations:
#   - adj_only:  lambda_adj=1, lambda_same=0  (adjacency-only baseline)
#   - same_only: lambda_adj=0, lambda_same=1  (same-image-only ablation)
#
# (multi-task at lambda_same=1 already has 5 seeds in noise_seed_0..4)
#
# Total: 10 new runs at ~25 min each = ~4-5 hours.
#
# Run from the project root:
#     bash scripts/run_overnight.sh
#
# On macOS, prefix with `caffeinate -i` to keep the laptop awake.

set -e
cd "$(dirname "$0")/.."

bash scripts/run_seed_sweep.sh adj_only 1.0 0.0 0.01923 "adjacency-only, ANOVA replicate (lambda_same=0)"

bash scripts/run_seed_sweep.sh same_only 0.0 1.0 0.01923 "same-image-only, ANOVA replicate (lambda_adj=0)"

echo
echo "================================================================"
echo "[$(date)] all 10 ANOVA runs finished"
echo "================================================================"
python scripts/compare_runs.py
echo
echo "Detailed ANOVA analysis:"
python scripts/anova_analysis.py
