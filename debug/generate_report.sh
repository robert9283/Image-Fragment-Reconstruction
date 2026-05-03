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

# refresh the figures used in the approach document so they always reflect
# the current best checkpoint in results.jsonl
python scripts/failure_modes.py
python scripts/roc_curves.py
python scripts/weight_distribution.py

echo "Report ready: debug/report.pdf"
echo "Approach doc figures refreshed:"
echo "  doc/failure_modes.png, doc/fig_failure_modes.tex"
echo "  doc/roc_curves.png, doc/fig_roc_curves.tex"
echo "  doc/weight_distribution.png, doc/fig_weight_distribution.tex"
