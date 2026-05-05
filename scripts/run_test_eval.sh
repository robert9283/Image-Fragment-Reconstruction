#!/bin/bash
# Run the final test evaluation on a trained checkpoint.
# Saves results to runs/<run_name>/test_metrics.json.
#
# Usage:
#   bash scripts/run_test_eval.sh <run_name>
#
# Example:
#   bash scripts/run_test_eval.sh multitask_10_seed_2

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_test_eval.sh <run_name>"
    echo "Available runs:"
    ls runs/ | grep -v latest
    exit 1
fi

RUN_DIR="runs/$1"

if [ ! -f "$RUN_DIR/model.pt" ]; then
    echo "Error: no checkpoint found at $RUN_DIR/model.pt"
    exit 1
fi

echo "Evaluating: $RUN_DIR"
CHECKPOINT_PATH="$RUN_DIR/model" python src/script1_metrics.py
