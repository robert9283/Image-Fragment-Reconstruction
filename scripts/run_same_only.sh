#!/bin/bash
# Run the missing same_only ablation: train only on same-image prediction
# (lambda_adj = 0, lambda_same = 1.0). Adjacency head receives no gradient.
#
# This isolates the contribution of the adjacency pretext task. If ARI
# matches multitask_10 (~0.570), the adjacency loss adds nothing and the
# simpler same-image-only approach wins. If ARI drops sharply, the
# adjacency loss is doing real representation work (potentially preventing
# encoder collapse where all fragments of one image map to the same
# embedding).
#
# Run from the project root:
#     bash scripts/run_same_only.sh
#
# On macOS, prefix with `caffeinate -i` to keep the laptop awake.

set -e
cd "$(dirname "$0")/.."

source venv/bin/activate

RUN_NAME=same_only
NOTES="same-image only (lambda_adj=0); ablation of the adjacency head"

echo
echo "================================================================"
echo "[$(date)] starting $RUN_NAME"
echo "================================================================"

python - "$RUN_NAME" "$NOTES" <<'PY'
import sys, yaml
run_name, notes = sys.argv[1], sys.argv[2]
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']    = run_name
cfg['beta']        = 0.01923
cfg['lambda_adj']  = 0.0
cfg['lambda_same'] = 1.0
cfg['notes']       = notes
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

# overwrite an empty run dir if one is left over from a previous interrupted attempt
rm -rf "runs/${RUN_NAME}"

python main.py > "runs/${RUN_NAME}.log" 2>&1
echo "[$(date)] $RUN_NAME finished"

echo
echo "Comparison:"
python scripts/compare_runs.py
