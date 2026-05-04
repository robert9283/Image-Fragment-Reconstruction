# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtualenv
source venv/bin/activate

# Run training (reads config.yaml)
python main.py

# Compare all logged runs
python scripts/compare_runs.py
python scripts/compare_runs.py --plot   # also saves runs_comparison.png

# Submission scripts (edit DATA_PATH / CHECKPOINT_PATH inside first)
python src/script1_metrics.py           # collect metrics over 1000 samples
python src/script2_visualise.py         # visualise clustering on one batch

# Analysis scripts (run from project root)
python scripts/anova_analysis.py
python scripts/roc_curves.py
python scripts/failure_modes.py
python scripts/weight_distribution.py
```

There are no automated tests. Evaluation is integrated into the training loop and into the submission scripts above.

## Architecture

The task is **self-supervised image fragment clustering**: 64×64 ImageNet images are sliced into a 4×4 grid of 16×16 patches ("fragments"), and the model must learn embeddings that allow clustering those fragments back to their source images, without any image-level labels during training.

### Data flow

`to_share/src/data.py` (`Imagenet64`) — provided by the assignment, not modified. Yields batches of images via `datagen_cls(batch_size, ds, augmentation)`.

`src/fragments.py` — `extract_fragments(images)` turns `(N, 64, 64, 3)` into `(N*16, 16, 16, 3)` plus integer source-image labels. `build_adjacency(n_images)` returns a binary `(N*16, N*16)` matrix marking spatially adjacent fragment pairs within each image.

### Model

`src/model.py` — abstract `BaseModel` interface (`train_step`, `get_output`, `save`, `load`).

`src/fragment_adjacency_predictor.py` — the only concrete model. A Siamese CNN (`CNNEncoder`: 3-conv + 2 pool → 256-d embedding) paired with one or two `ComparisonHead` MLPs:

- **Adjacency head** (always active): predicts whether two fragments are spatially adjacent. Trained with weighted BCE; `beta` in config scales the positive-class weight by the natural negative/positive ratio.
- **Same-image head** (active when `lambda_same > 0`): predicts whether two fragments come from the same source image.

Total loss: `lambda_adj * WBCE(adj) + lambda_same * WBCE(same)`.

`get_output` returns an `(N, N)` pairwise similarity matrix (using the same-image head when present, otherwise the adjacency head), which is consumed by the clustering step.

### Clustering

`src/clustering.py` — `cluster(model_output)` routes on output shape:
- `(N, N)` → spectral clustering on the similarity matrix
- `(N, D)` → k-means on embeddings

When `balanced_clustering: true` in config, a Hungarian-assignment balanced k-means enforces exactly `GRID*GRID = 16` fragments per cluster. Metrics: ARI, NMI, purity via `compute_metrics`.

### Training loop (`main.py`)

1. Reads `config.yaml`, sets random seeds.
2. Creates `runs/{run_name}/` with a config snapshot; updates `runs/latest` symlink.
3. Each iteration: sample a batch → extract fragments + adjacency → `model.train_step`.
4. Every `eval_every` iterations: evaluate on `n_eval_batches` validation batches; save checkpoint if ARI improves; early-stop after `patience` non-improving evaluations.
5. Appends a one-line JSON summary to `results.jsonl`.

Run logs are stored as `runs/{run_name}/training_log.jsonl` (one JSON entry per eval).

### Key config parameters

| Key | Effect |
|---|---|
| `model` | Model name (`fragment-adjacency-predictor`) |
| `n_images` | Batch size in images (fragments = n_images × 16) |
| `beta` | Scales `pos_weight_adj`; set to `1/ratio` to approximate uniform weighting |
| `lambda_adj` / `lambda_same` | Loss weights for each head (set to 0 to disable) |
| `balanced_clustering` | Enforce equal cluster sizes during eval |
| `n_eval_batches` | Number of validation batches averaged per eval point |
| `seed` | Controls torch and numpy RNG |

### Adding a new model

Subclass `BaseModel` in a new file under `src/`, then add a branch in `load_model()` in `main.py`.
