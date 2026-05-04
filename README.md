# Image Fragment Reconstruction

Self-supervised model that groups mixed image fragments back to their
original source images.

**Reports:**
- [`doc/report.pdf`](doc/report.pdf) — 3-page submission report (architecture, results, challenges)
- [`doc/report_long.pdf`](doc/report_long.pdf) — detailed write-up covering design decisions, experiments, and statistical analysis

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `to_share/data/` folder must contain the ImageNet-64 dataset
(provided separately).

## Quick start

Train the best configuration (multi-task adjacency + same-image, plain
BCE on both heads) and evaluate it:

```bash
# 1. Train one model — ~25 minutes on a MacBook Air M-series CPU.
#    All hyperparameters are in config.yaml; the defaults match the
#    best configuration documented in the report.
python main.py

# 2. Generate the per-run debug report and refresh the figures used
#    in the approach document. Reads from runs/latest/ automatically.
bash diagnostics/generate_report.sh
```

After training, the per-run outputs (config snapshot, training log,
checkpoint) live in `runs/<run_name>/`, and a one-line summary is
appended to `results.jsonl`. A `runs/latest` symlink points to the
most recent run.

## Submission scripts

The two scripts the case study asks for live in `src/`. Both load the
best checkpoint and run on the test split.

```bash
python src/script1_metrics.py    # per-sample metrics over 1000 samples
python src/script2_visualise.py  # cluster visualisation on a single sample
```

Both scripts default to `runs/latest/model` and read their config from
the same directory. Override via environment variables:

```bash
CHECKPOINT_PATH=runs/my_run/model python src/script1_metrics.py
```

## Reproducing the results in the report

The `scripts/` folder contains the experiments referenced in
`doc/report_long.pdf`:

| Script | Purpose |
|---|---|
| `scripts/plan.py`           | Defines the 5-condition ANOVA run plan and launches missing runs. |
| `scripts/run_overnight.sh`  | Launches the full multi-seed ANOVA experiment (~4 h). |
| `scripts/run_test_eval.sh`  | Evaluates a named run on the test split; saves `test_metrics.json`. |
| `scripts/anova_r.R`         | Welch one-way ANOVA + planned contrasts + Cohen's d; outputs LaTeX tables. |
| `scripts/compare_runs.py`   | Markdown comparison table across all runs. |
| `scripts/failure_modes.py`  | Generates the failure-modes figure for the report. |
| `scripts/roc_curves.py`     | Generates the ROC / PR curves figure for the report. |
| `scripts/weight_distribution.py` | Generates the confidence-score distribution figure. |

`diagnostics/generate_report.sh` runs the debug pipeline plus the three
report-figure scripts, so the figures in `doc/report_long.pdf` always
reflect the current best checkpoint in `results.jsonl`.

## Project structure

```
config.yaml              # hyperparameters; one source of truth
main.py                  # training entry point
requirements.txt
README.md
todo.md                  # outstanding work
src/
  model.py               # abstract base class for all models
  fragments.py           # fragmentation + adjacency label construction
  fragment_adjacency_predictor.py   # the model: Siamese CNN + two heads
  clustering.py          # spectral clustering, balanced variant, metrics
  script1_metrics.py     # submission script 1
  script2_visualise.py   # submission script 2
scripts/
  *                      # experiment runners and analysis utilities
diagnostics/
  debug_pipeline.py      # sanity check on the data pipeline
  debug_model.py         # adjacency-prediction inspection on best checkpoint
  plot_training.py       # training curves from runs/latest/training_log.jsonl
  generate_report.py     # bundles all diagnostic plots into report.pdf
  generate_report.sh     # full diagnostic pipeline + report-figure refresh
doc/
  report.tex / .pdf      # 3-page submission report
  report_long.tex / .pdf # detailed write-up
  figures/               # TikZ and PNG figures, included via \input / \includegraphics
  references.bib
  build.sh               # compiles report_long.pdf (pdflatex + biber)
runs/                    # per-run outputs (config + log + checkpoint),
                         # one subfolder per run, runs/latest -> newest
to_share/
  src/data.py            # data loader and augmentation (provided)
  data/                  # ImageNet-64 dataset (provided)
results.jsonl            # one-line summary per run, version-controlled
```

## Configuration

All hyperparameters live in `config.yaml`. Key knobs:

| Key | Default | Description |
|---|---|---|
| `model`               | `fragment-adjacency-predictor` | Model to load |
| `data_path`           | `to_share/data` | Dataset folder |
| `n_images`            | `10`         | Images per training sample |
| `max_iterations`      | `25000`      | Cap on training iterations |
| `eval_every`          | `250`        | Evaluation frequency |
| `patience`            | `10`         | Early-stopping patience (in eval steps) |
| `n_eval_batches`      | `20`         | Validation batches averaged per eval |
| `balanced_clustering` | `true`       | Hungarian-balanced spectral clustering |
| `beta`                | `0.01923`    | WBCE tilt parameter on adjacency (= plain BCE) |
| `lambda_adj`          | `1.0`        | Weight on the adjacency loss |
| `lambda_same`         | `1.0`        | Weight on the same-image loss; `0` disables the head |
| `pos_weight_same`     | `1.0`        | Weight on same-image positives (= plain BCE) |
| `seed`                | unset        | If set, calls `torch.manual_seed` and `np.random.seed` |
| `run_name`            | unset        | Auto-generated from timestamp if blank |
| `notes`               | empty        | Free-form description, written into `results.jsonl` |

## How a fresh submission would re-create the report

```bash
# 1. Set up the environment.
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train. (~25 min)
python main.py

# 3. Refresh the report figures (uses runs/latest/ automatically).
bash diagnostics/generate_report.sh

# 4. Compile the detailed write-up PDF.
cd doc && bash build.sh

# 5. (Optional) reproduce the multi-seed ANOVA experiment used in the
#    "Multi-seed comparison" section of the report. ~4 hours.
caffeinate -i bash scripts/run_overnight.sh    # macOS
# or
bash scripts/run_overnight.sh                  # Linux
```
