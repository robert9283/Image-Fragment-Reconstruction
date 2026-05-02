#!/bin/bash
# Two remaining multi-task data points (multitask_01 already finished
# in a previous overnight session, ARI=0.489):
#
# - multitask_15: lambda_adj=1, lambda_same=1.5.  Between the equal-weight
#                 setting that gave us our best ARI (0.570 at 1.0) and the
#                 same-image-dominates setting we skipped at 2.0. Tells us
#                 whether ARI continues to climb past 1.0 or peaks there.
# - same_only:    lambda_adj=0, lambda_same=1.   Drops the adjacency loss
#                 entirely and trains only on same-image prediction. Tells
#                 us whether the adjacency pretext is doing real
#                 representation work, or whether same-image alone is
#                 sufficient. Also tests whether the encoder collapses to
#                 a within-image-degenerate solution without the adjacency
#                 regulariser.
#
# All runs use plain BCE on both heads (beta = 1/52 on adjacency,
# pos_weight_same = 1 on same-image).
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
    local lambda_adj="$2"
    local lambda_same="$3"
    local notes="$4"

    echo
    echo "================================================================"
    echo "[$(date)] starting $run_name  (lambda_adj=$lambda_adj  lambda_same=$lambda_same)"
    echo "================================================================"

    python - "$run_name" "$lambda_adj" "$lambda_same" "$notes" <<'PY'
import sys, yaml
run_name, lambda_adj, lambda_same, notes = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']     = run_name
cfg['beta']         = 0.01923   # plain BCE on adjacency
cfg['lambda_adj']   = lambda_adj
cfg['lambda_same']  = lambda_same
cfg['notes']        = notes
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    python main.py > "runs/${run_name}.log" 2>&1
    echo "[$(date)] $run_name finished"
}

run_one multitask_15 1.0 1.5 "multi-task, same-image weighted higher than adjacency"
run_one same_only    0.0 1.0 "same-image only (lambda_adj=0); ablation of the adjacency head"

echo
echo "================================================================"
echo "[$(date)] both runs finished"
python scripts/compare_runs.py
