#!/bin/bash
# Run 5 seeds of a given configuration, suitable as one ANOVA "group".
#
# Usage (from project root):
#     bash scripts/run_seed_sweep.sh <base_name> <lambda_adj> <lambda_same> <beta> "<notes>"
#
# Each individual run is named "<base_name>_seed_<n>" for n in 0..4 and writes
# its outputs to runs/<base_name>_seed_<n>/. Per-seed metrics are appended to
# results.jsonl as usual.

set -e
cd "$(dirname "$0")/.."

if [ "$#" -ne 5 ]; then
    echo "usage: bash scripts/run_seed_sweep.sh <base_name> <lambda_adj> <lambda_same> <beta> \"<notes>\""
    exit 1
fi

BASE_NAME="$1"
LAMBDA_ADJ="$2"
LAMBDA_SAME="$3"
BETA="$4"
NOTES="$5"

source venv/bin/activate

for seed in 0 1 2 3 4; do
    run_name="${BASE_NAME}_seed_${seed}"

    # resume support: skip a seed only if it has BOTH a saved checkpoint AND
    # a matching summary in results.jsonl. The combination guards against
    # partial state (runs that were killed mid-training save a checkpoint but
    # never write the summary): those get rerun from scratch on resume.
    has_ckpt=false
    has_summary=false
    [ -f "runs/${run_name}/model.pt" ] && has_ckpt=true
    grep -q "\"run\": \"${run_name}\"" results.jsonl 2>/dev/null && has_summary=true
    if $has_ckpt && $has_summary; then
        echo "[$(date)] skipping ${run_name} (already finished)"
        continue
    fi
    if $has_ckpt && ! $has_summary; then
        echo "[$(date)] removing partial run ${run_name} (checkpoint without summary)"
        rm -rf "runs/${run_name}"
    fi

    echo
    echo "================================================================"
    echo "[$(date)] starting ${run_name}"
    echo "  lambda_adj=${LAMBDA_ADJ}  lambda_same=${LAMBDA_SAME}  beta=${BETA}  seed=${seed}"
    echo "================================================================"

    python - "$run_name" "$seed" "$LAMBDA_ADJ" "$LAMBDA_SAME" "$BETA" "$NOTES" <<'PY'
import sys, yaml
run_name, seed, l_adj, l_same, beta, notes = sys.argv[1:]
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']           = run_name
cfg['seed']               = int(seed)
cfg['model']              = 'fragment-adjacency-predictor'
cfg['n_images']           = 10
cfg['max_iterations']     = 25000
cfg['eval_every']         = 250
cfg['patience']           = 10
cfg['beta']               = float(beta)
cfg['lambda_adj']         = float(l_adj)
cfg['lambda_same']        = float(l_same)
cfg['n_eval_batches']     = 20
cfg['balanced_clustering']= True
cfg['notes']              = notes
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    python main.py > "runs/${run_name}.log" 2>&1
    echo "[$(date)] ${run_name} finished"
done
