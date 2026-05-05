#!/bin/bash
# Quick smoke-test for the single-head refactor.
# Runs 200 iterations with lambda_same=1.0 to exercise the multi-task code path.
# Should finish in ~2-3 minutes on CPU.
# Usage: bash scripts/test_single_head.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

python3 - <<'PY'
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']            = 'test_single_head_full'
cfg['model']               = 'fragment-adjacency-predictor'
cfg['n_images']            = 10
cfg['max_iterations']      = 25000
cfg['eval_every']          = 250
cfg['patience']            = 10
cfg['beta']                = 0.01923
cfg['lambda_adj']          = 1.0
cfg['lambda_same']         = 1.0
cfg['n_eval_batches']      = 20
cfg['balanced_clustering'] = True
cfg['seed']                = 0
cfg['notes']               = 'single-head refactor validation'
cfg.pop('pos_weight_same', None)
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "Running smoke test (200 iters, lambda_same=1.0)..."
python main.py
echo
echo "Smoke test passed. Check runs/test_single_head/ for outputs."
