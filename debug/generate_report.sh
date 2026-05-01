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

echo "Report ready: debug/report.pdf"
