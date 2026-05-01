# Image Fragment Reconstruction

Self-supervised model that groups mixed image fragments back to their original source images.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
config.yaml              # hyperparameters and paths
main.py                  # training entry point
requirements.txt
src/
  model.py               # abstract base class for all models
  fragments.py           # fragmentation and adjacency label construction
  evaluate.py            # clustering and metrics (ARI, NMI, purity)
  script1_metrics.py     # submission script 1: metrics over 1000 samples
  script2_visualise.py   # submission script 2: visualisation on a single sample
to_share/
  src/data.py            # data loading and augmentation (provided)
  data/                  # ImageNet64 dataset (provided)
checkpoints/             # saved model weights
debug/
  debug_pipeline.py      # sanity check for the data pipeline
doc/                     # approach documents and report
```

## Training

Edit `config.yaml` if needed, then:

```bash
python main.py
```

Training uses early stopping on validation ARI. The best checkpoint is saved automatically to `checkpoints/`.

## Evaluation

Edit the two paths at the top of each script, then run:

**Script 1** — metrics over 1000 samples:
```bash
python src/script1_metrics.py
```

**Script 2** — visualisation on a single sample:
```bash
python src/script2_visualise.py
```

## Configuration

| Parameter | Description |
|---|---|
| `model` | Model to use (currently: `fragment-adjacency-predictor`) |
| `data_path` | Path to the dataset folder |
| `checkpoint_path` | Path to save/load model weights |
| `n_images` | Images per training sample (default: 10) |
| `max_iterations` | Maximum training iterations |
| `eval_every` | Evaluate on validation set every N iterations |
| `patience` | Stop after N evaluations with no improvement |
