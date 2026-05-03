#!/bin/bash
# Multi-seed runs of the best configuration (multitask_10) to estimate
# the run-to-run noise floor more rigorously.
#
# All five runs use identical hyperparameters; only the random seed
# differs. The std of best ARI across the five runs is our estimate of
# sigma_seed: the variance attributable to model initialisation, batch
# order, and stochastic augmentation, with everything else held fixed.
#
# Configuration: multi-task with equal weights, plain BCE on both heads,
# balanced spectral clustering, 20-batch eval averaging.
#
# Run from the project root:
#     bash scripts/run_overnight.sh
#
# On macOS, prefix with `caffeinate -i` to keep the laptop awake.
# Total runtime: ~5 * 25 min = ~2 hours.

set -e
cd "$(dirname "$0")/.."

source venv/bin/activate

run_one() {
    local seed="$1"

    echo
    echo "================================================================"
    echo "[$(date)] starting noise_seed_${seed}"
    echo "================================================================"

    python - "$seed" <<'PY'
import sys, yaml
seed = int(sys.argv[1])
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']           = f'noise_seed_{seed}'
cfg['seed']               = seed
cfg['model']              = 'fragment-adjacency-predictor'
cfg['n_images']           = 10
cfg['max_iterations']     = 25000
cfg['eval_every']         = 250
cfg['patience']           = 10
cfg['beta']               = 0.01923   # plain BCE on adjacency
cfg['lambda_adj']         = 1.0
cfg['lambda_same']        = 1.0       # equal-weight multi-task (the best config)
cfg['pos_weight_same']    = 1.0
cfg['n_eval_batches']     = 20
cfg['balanced_clustering']= True
cfg['notes']              = f'multi-seed noise estimation, seed={seed}'
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    python main.py > "runs/noise_seed_${seed}.log" 2>&1
    echo "[$(date)] noise_seed_${seed} finished"
}

for s in 0 1 2 3 4; do
    run_one "$s"
done

echo
echo "================================================================"
echo "[$(date)] all five noise-floor runs finished"
echo "================================================================"
python scripts/compare_runs.py
echo
echo "Quick summary:"
python - <<'PY'
import json, statistics
rows = [json.loads(l) for l in open('results.jsonl') if 'noise_seed_' in l]
aris = [r['best_ari'] for r in rows]
print(f'  n        = {len(aris)}')
print(f'  ARI      = {aris}')
print(f'  mean     = {statistics.mean(aris):.4f}')
print(f'  std      = {statistics.stdev(aris):.4f}  (this is sigma_seed)')
print(f'  range    = [{min(aris):.4f}, {max(aris):.4f}]')
PY
