#!/bin/bash
# Generate all debug plots and compile the PDF report.
# Run from the project root: bash debug/generate_report.sh

set -e
cd "$(dirname "$0")/.."

source venv/bin/activate

python debug/debug_pipeline.py
python debug/debug_model.py
python debug/plot_training.py
python debug/generate_report.py

# refresh the failure-modes figure used in the approach document, so it
# always reflects the current best checkpoint in results.jsonl
python scripts/failure_modes.py

echo "Report ready: debug/report.pdf"
echo "Approach doc figure refreshed: doc/failure_modes.png, doc/fig_failure_modes.tex"
