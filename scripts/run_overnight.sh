#!/bin/bash
# Run a small multi-task sweep back-to-back. All runs use plain BCE on the
# adjacency head (beta = 1/52 = 0.01923, since the wbce sweep showed the
# tilt is a second-order knob). What varies is the weight lambda_same on
# the auxiliary same-image prediction head.
#
# - lambda_same = 0   (plain_bce_baseline): adjacency-only, our current best.
#                     This run reproduces the existing result and gives a
#                     fresh seed for the noise floor.
# - lambda_same = 0.5 (multitask_05): mild same-image auxiliary loss.
# - lambda_same = 1.0 (multitask_10): equal-weight multi-task.
# - lambda_same = 2.0 (multitask_20): same-image dominates the loss.
#
# Run from the project root:
#     bash scripts/run_overnight.sh
#
# On macOS, prefix with `caffeinate -i` to keep the laptop awake.

set -e
cd "$(dirname "$0")/.."

source venv/bin/activate

run_one() {
    local run_name="$1"
    local lambda_same="$2"
    local notes="$3"

    echo
    echo "================================================================"
    echo "[$(date)] starting $run_name  (lambda_same=$lambda_same)"
    echo "================================================================"

    python - "$run_name" "$lambda_same" "$notes" <<'PY'
import sys, yaml
run_name, lambda_same, notes = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']     = run_name
cfg['beta']         = 0.01923   # plain BCE on adjacency
cfg['lambda_adj']   = 1.0
cfg['lambda_same']  = lambda_same
cfg['notes']        = notes
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    python main.py > "runs/${run_name}.log" 2>&1
    echo "[$(date)] $run_name finished"
}

run_one plain_bce_baseline 0.0 "adjacency-only baseline (lambda_same=0)"
run_one multitask_05       0.5 "multi-task, mild same-image weight"
run_one multitask_10       1.0 "multi-task, equal weights"
run_one multitask_20       2.0 "multi-task, same-image dominates"

echo
echo "================================================================"
echo "[$(date)] all four runs finished"
python scripts/compare_runs.py
