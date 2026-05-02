#!/bin/bash
# Wait for a given PID to finish, then update config.yaml and launch main.py.
#
# Usage:
#     bash scripts/run_after.sh <pid_to_wait_for> <run_name> <beta> "<notes>"
#
# All four arguments are required. The script polls for the PID's exit, sets
# the run_name / beta / notes in config.yaml in place, and then runs main.py
# with stdout/stderr redirected to runs/<run_name>.log.

set -e
cd "$(dirname "$0")/.."

PID_TO_WAIT="$1"
RUN_NAME="$2"
BETA="$3"
NOTES="$4"

if [ -z "$PID_TO_WAIT" ] || [ -z "$RUN_NAME" ] || [ -z "$BETA" ] || [ -z "$NOTES" ]; then
    echo "usage: bash scripts/run_after.sh <pid> <run_name> <beta> \"<notes>\""
    exit 1
fi

echo "[$(date)] waiting for PID $PID_TO_WAIT to finish..."
while kill -0 "$PID_TO_WAIT" 2>/dev/null; do
    sleep 30
done
echo "[$(date)] PID $PID_TO_WAIT finished, preparing $RUN_NAME (beta=$BETA)"

# update config.yaml in place using python (avoids fragile sed on yaml)
python - "$RUN_NAME" "$BETA" "$NOTES" <<'PY'
import sys, yaml
run_name, beta, notes = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['run_name'] = run_name
cfg['beta']     = beta
cfg['notes']    = notes
with open('config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'config.yaml updated: run_name={run_name}, beta={beta}')
PY

source venv/bin/activate
echo "[$(date)] launching main.py for $RUN_NAME"
python main.py > "runs/${RUN_NAME}.log" 2>&1
echo "[$(date)] $RUN_NAME finished"
