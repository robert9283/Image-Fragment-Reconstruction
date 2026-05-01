#!/bin/bash
set -e

cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode summary.tex
biber summary
pdflatex -interaction=nonstopmode summary.tex
pdflatex -interaction=nonstopmode summary.tex

rm -f summary.aux summary.bbl summary.bcf summary.blg summary.log summary.out summary.run.xml summary.toc

echo "Done: summary.pdf"
