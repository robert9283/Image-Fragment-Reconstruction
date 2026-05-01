import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))

import numpy as np
from data import Imagenet64
from fragments import extract_fragments, build_adjacency
from evaluate import cluster, compute_metrics

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model')

# ── hyperparameters ───────────────────────────────────────────────────────────
N_IMAGES     = 10
N_ITERATIONS = 1000
EVAL_EVERY   = 100

# ── data ──────────────────────────────────────────────────────────────────────
dataset   = Imagenet64(DATA_PATH)
train_gen = dataset.datagen_cls(batch_size=N_IMAGES, ds='train', augmentation=True)

# ── model ─────────────────────────────────────────────────────────────────────
# TODO: import and instantiate your model, e.g.:
# from my_model import MyModel
# model = MyModel()

# ── training loop ─────────────────────────────────────────────────────────────
for iteration in range(N_ITERATIONS):
    images, _ = next(train_gen)
    fragments, labels = extract_fragments(np.array(images))
    adjacency = build_adjacency(n_images=N_IMAGES)

    # TODO: loss = model.train_step(fragments, labels, adjacency)

    if (iteration + 1) % EVAL_EVERY == 0:
        # TODO: uncomment once model is implemented
        # model_output = model.get_output(fragments)
        # pred_labels  = cluster(model_output)
        # metrics      = compute_metrics(pred_labels, labels)
        # print(f"iter {iteration+1:4d}  ARI={metrics['ari']:.3f}  NMI={metrics['nmi']:.3f}  purity={metrics['purity']:.3f}")
        pass

# TODO: model.save(CHECKPOINT_PATH)
