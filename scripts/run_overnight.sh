#!/bin/bash
# Run all pending experiments from the plan DB.
# Ctrl-C stops cleanly after the current run finishes.
#
# First-time setup:
#   python scripts/plan.py init    # populate the DB
#   python scripts/plan.py list    # review the plan
#
# Then run:
#   caffeinate -i bash scripts/run_overnight.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

python scripts/plan.py run-all
