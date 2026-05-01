"""
Generate a PDF report bundling all debug plots.
Run from the project root:
    python debug/generate_report.py
"""
import os
import sys
import subprocess

DEBUG_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_PATH  = os.path.join(DEBUG_DIR, 'report.tex')

PLOTS = [
    ('Data Pipeline',
     'out_originals.png',
     'Ten augmented source images as yielded by the data generator.'),
    ('Data Pipeline',
     'out_fragments.png',
     '4$\\times$4 grid of 16$\\times$16 fragments extracted from image 0.'),
    ('Data Pipeline',
     'out_shuffled_pile.png',
     'First 32 fragments from the shuffled pile. Source image index shown in title.'),
    ('Data Pipeline',
     'out_adjacency.png',
     'Ground truth adjacency matrix (160$\\times$160). White lines separate source images. '
     'Blue entries indicate spatially adjacent fragment pairs.'),
    ('Model Output',
     'out_model_matrices.png',
     'Left: ground truth adjacency matrix. Right: predicted similarity matrix from the '
     'trained model. The trained model concentrates high similarity values within '
     'source-image blocks.'),
    ('Model Output',
     'out_model_top_pairs.png',
     'Top 6 fragment pairs predicted as most similar by the trained model. '
     'Green border = truly adjacent. All pairs originate from the same source image.'),
    ('Training',
     'out_training_curves.png',
     'Training curves over time. Top: loss. Middle: adjacency prediction metrics '
     '(F1, precision, recall). Bottom: fragment clustering metrics (ARI, NMI, purity).'),
]

def write_tex():
    lines = [
        r'\documentclass[12pt,a4paper]{article}',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage[margin=2cm]{geometry}',
        r'\usepackage{graphicx}',
        r'\usepackage{booktabs}',
        r'\usepackage{hyperref}',
        r'\usepackage{parskip}',
        r'\title{Debug Report}',
        r'\date{\today}',
        r'\begin{document}',
        r'\maketitle',
        r'\tableofcontents',
        r'\newpage',
    ]

    current_section = None
    for section, filename, caption in PLOTS:
        filepath = os.path.join(DEBUG_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  Skipping {filename} — not found")
            continue

        if section != current_section:
            lines.append(f'\n\\section{{{section}}}')
            current_section = section

        rel_path = os.path.relpath(filepath, DEBUG_DIR).replace('\\', '/')
        lines += [
            r'\begin{figure}[h!]',
            r'\centering',
            f'\\includegraphics[width=\\textwidth]{{{rel_path}}}',
            f'\\caption{{{caption}}}',
            r'\end{figure}',
            r'\clearpage',
        ]

    lines.append(r'\end{document}')

    with open(TEX_PATH, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written: {TEX_PATH}")


def compile_tex():
    for _ in range(2):
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', DEBUG_DIR, TEX_PATH],
            capture_output=True, text=True
        )
    pdf_path = TEX_PATH.replace('.tex', '.pdf')
    if os.path.exists(pdf_path):
        print(f"Compiled: {pdf_path}")
    else:
        print("Compilation failed. Check report.log for details.")
        print(result.stdout[-1000:])

    for ext in ['aux', 'log', 'out', 'toc']:
        aux = TEX_PATH.replace('.tex', f'.{ext}')
        if os.path.exists(aux):
            os.remove(aux)


if __name__ == '__main__':
    write_tex()
    compile_tex()
