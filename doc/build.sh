#!/bin/bash
set -e

cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
rm -f report.aux report.log report.out report.toc
echo "Done: report.pdf"

for doc in report_long; do
    pdflatex -interaction=nonstopmode ${doc}.tex
    biber ${doc}
    pdflatex -interaction=nonstopmode ${doc}.tex
    pdflatex -interaction=nonstopmode ${doc}.tex
    rm -f ${doc}.aux ${doc}.bbl ${doc}.bcf ${doc}.blg ${doc}.log ${doc}.out ${doc}.run.xml ${doc}.toc
    echo "Done: ${doc}.pdf"
done
