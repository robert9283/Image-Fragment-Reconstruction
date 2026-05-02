#!/bin/bash
set -e

cd "$(dirname "$0")"

for doc in summary approach; do
    pdflatex -interaction=nonstopmode ${doc}.tex
    biber ${doc}
    pdflatex -interaction=nonstopmode ${doc}.tex
    pdflatex -interaction=nonstopmode ${doc}.tex
    rm -f ${doc}.aux ${doc}.bbl ${doc}.bcf ${doc}.blg ${doc}.log ${doc}.out ${doc}.run.xml ${doc}.toc
    echo "Done: ${doc}.pdf"
done
