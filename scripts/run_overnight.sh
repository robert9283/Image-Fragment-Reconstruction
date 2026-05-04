#!/bin/bash
# Full experiment rerun after single-head refactor.
#
# Re-runs every configuration that has lambda_same > 0 (behaviour changed).
# lambda_same=0 runs (beta sweep, adj_only ANOVA) are unaffected and skipped.
#
# Configurations rerun:
#   lambda sweep  : multitask_01/05/10/15, same_only          (5 single runs)
#   ANOVA groups  : noise_seed_0..4, same_only_seed_0..4      (10 seed runs)
#
# Total: ~15 runs x ~25 min ~= 6 hours.
# On macOS: caffeinate -i bash scripts/run_overnight.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

# helpers

remove_run() {
    local name="$1"
    rm -rf "runs/${name}"
    python3 - "$name" <<'PY'
import sys, json
name = sys.argv[1]
with open("results.jsonl") as f:
    lines = [l for l in f if json.loads(l)["run"] != name]
with open("results.jsonl", "w") as f:
    f.writelines(lines)
PY
    echo "  removed: $name"
}

run_single() {
    local run_name="$1" lambda_adj="$2" lambda_same="$3" beta="$4" notes="$5"
    echo
    echo "================================================================"
    echo "[$(date)] starting ${run_name}"
    echo "  lambda_adj=${lambda_adj}  lambda_same=${lambda_same}  beta=${beta}"
    echo "================================================================"
    python3 - "$run_name" "$lambda_adj" "$lambda_same" "$beta" "$notes" <<'PY'
import sys, yaml
run_name, l_adj, l_same, beta, notes = sys.argv[1:]
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name']           = run_name
cfg['model']              = 'fragment-adjacency-predictor'
cfg['n_images']           = 10
cfg['max_iterations']     = 25000
cfg['eval_every']         = 250
cfg['patience']           = 10
cfg['beta']               = float(beta)
cfg['lambda_adj']         = float(l_adj)
cfg['lambda_same']        = float(l_same)
cfg['n_eval_batches']     = 20
cfg['balanced_clustering'] = True
cfg['notes']              = notes
cfg.pop('seed', None)
cfg.pop('pos_weight_same', None)
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
    python main.py > "runs/${run_name}.log" 2>&1
    echo "[$(date)] ${run_name} finished"
}

# 1. remove stale runs with lambda_same > 0
echo "Removing stale multi-task and same-only runs..."
for run in multitask_01 multitask_05 multitask_10 multitask_15 same_only \
           noise_seed_0 noise_seed_1 noise_seed_2 noise_seed_3 noise_seed_4 \
           same_only_seed_0 same_only_seed_1 same_only_seed_2 \
           same_only_seed_3 same_only_seed_4; do
    remove_run "$run"
done

# 2. lambda sweep (single seed)
# plain_bce_baseline (lambda_same=0) unchanged -- not rerun.
run_single "multitask_01" 1.0 0.1 0.01923 "multi-task, lambda_same=0.1"
run_single "multitask_05" 1.0 0.5 0.01923 "multi-task, lambda_same=0.5"
run_single "multitask_10" 1.0 1.0 0.01923 "multi-task, lambda_same=1.0"
run_single "multitask_15" 1.0 1.5 0.01923 "multi-task, lambda_same=1.5"
run_single "same_only"    0.0 1.0 0.01923 "same-image-only (lambda_adj=0)"

# 3. ANOVA seed sweeps
# adj_only_seed_* (lambda_same=0) unchanged -- not rerun.
bash scripts/run_seed_sweep.sh noise     1.0 1.0 0.01923 "multi-task ANOVA replicate (lambda_same=1)"
bash scripts/run_seed_sweep.sh same_only 0.0 1.0 0.01923 "same-only ANOVA replicate (lambda_adj=0)"

# 4. analysis
echo
echo "================================================================"
echo "[$(date)] all runs finished"
echo "================================================================"
python scripts/compare_runs.py
echo
echo "Detailed ANOVA analysis:"
python scripts/anova_analysis.py
