"""
Submission script 1: evaluate the model on N_SAMPLES test batches and report metrics.

This script measures two things on the held-out test split:

  1. Adjacency prediction  — how well the model scores spatially adjacent
     fragment pairs higher than non-adjacent ones (AUROC, AUPRC, F1 at
     threshold 0.5).

  2. Fragment clustering   — how accurately balanced spectral clustering
     reconstructs the original source-image groupings (ARI, NMI, purity,
     each reported as mean ± std over N_SAMPLES batches).

Configuration is read from the config.yaml snapshot stored alongside the
checkpoint. Results are printed to stdout and saved as test_metrics.json in
the run directory.

Usage:
    python src/script1_metrics.py

Override paths via environment variables if needed:
    DATA_PATH=path/to/data CHECKPOINT_PATH=runs/my_run/model python src/script1_metrics.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'to_share', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import yaml
import numpy as np
from sklearn.metrics import f1_score
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID
from src.clustering import cluster, compute_metrics
from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor

# ── edit these two paths ──────────────────────────────────────────────────────
DATA_PATH       = os.environ.get('DATA_PATH',
                    os.path.join(os.path.dirname(__file__), '..', 'to_share', 'data'))
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH',
                    os.path.join(os.path.dirname(__file__), '..', 'runs', 'latest', 'model'))
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 1000  # number of independent test batches to evaluate


def load_model_from_checkpoint(checkpoint_path):
    """
    Read config.yaml from the run directory and instantiate a loaded model.

    The config snapshot written by main.py guarantees that hyperparameters
    (beta, lambda_adj, lambda_same, n_images) exactly match the saved weights.

    Args:
        checkpoint_path (str): Path to the checkpoint without extension.

    Returns:
        tuple[FragmentAdjacencyPredictor, dict]: Loaded model and its config.
    """
    run_dir = os.path.dirname(os.path.realpath(checkpoint_path))
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)

    # recompute the negative-to-positive pair ratio used to scale the WBCE
    # positive weight; for n_images=10 and a 4×4 grid this is always 52
    n_images = cfg.get('n_images', 10)
    n_pos    = n_images * 2 * GRID * (GRID - 1)
    n_pairs  = (n_images * GRID**2) * (n_images * GRID**2 - 1) // 2
    ratio    = (n_pairs - n_pos) / n_pos

    model = FragmentAdjacencyPredictor(
        pos_weight_adj = cfg.get('beta', 0.01923) * ratio,
        lambda_adj     = cfg.get('lambda_adj',  1.0),
        lambda_same    = cfg.get('lambda_same', 0.0),
    )
    model.load(checkpoint_path)
    return model, cfg


def evaluate_batch(model, gen, n_images):
    """
    Draw one batch from the generator and compute adjacency and clustering metrics.

    Adjacency metrics (AUROC, AUPRC) are threshold-free; F1 uses threshold 0.5.
    Clustering uses balanced spectral clustering, enforcing exactly GRID*GRID=16
    fragments per cluster to prevent the over-fragmentation bias.

    Args:
        model (FragmentAdjacencyPredictor): Loaded model in eval mode.
        gen: Test data generator yielding (images, _) tuples.
        n_images (int): Number of images per batch (= number of clusters).

    Returns:
        tuple[dict, dict]:
            adj  -- {'auroc': float, 'auprc': float, 'f1': float}
            cl   -- {'ari': float, 'nmi': float, 'purity': float}
    """
    images, _ = next(gen)
    fragments, true_labels = extract_fragments(np.array(images))
    adjacency = build_adjacency(n_images=n_images)

    # adjacency prediction
    adj_metrics = model.evaluate_adjacency(fragments, adjacency)
    probs, idx_i, idx_j = model._pair_scores(fragments)
    targets = adjacency[idx_i.cpu().numpy(), idx_j.cpu().numpy()]
    preds   = (np.array(probs) >= 0.5).astype(int)
    adj = {
        'auroc': adj_metrics['auroc'],
        'auprc': adj_metrics['auprc'],
        'f1':    float(f1_score(targets, preds, zero_division=0)),
    }

    # fragment clustering
    similarity  = model.get_output(fragments)
    pred_labels = cluster(similarity, n_per_cluster=GRID * GRID)
    cl = compute_metrics(pred_labels, true_labels)

    return adj, cl


def main():
    """
    Evaluate the model over N_SAMPLES batches and save aggregated metrics.

    Loads the model from CHECKPOINT_PATH, runs evaluate_batch() N_SAMPLES
    times, aggregates results, and writes test_metrics.json to the run directory.
    """
    model, cfg = load_model_from_checkpoint(CHECKPOINT_PATH)
    n_images   = cfg.get('n_images', 10)
    run_dir    = os.path.dirname(os.path.realpath(CHECKPOINT_PATH))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    dataset = Imagenet64(DATA_PATH)
    gen     = dataset.datagen_cls(batch_size=n_images, ds='test', augmentation=False)

    # accumulators — one entry per batch
    adjacency_results  = {'auroc': [], 'auprc': [], 'f1': []}
    clustering_results = {'ari': [], 'nmi': [], 'purity': []}

    for i in range(N_SAMPLES):
        adj, cl = evaluate_batch(model, gen, n_images)
        for k, v in adj.items(): adjacency_results[k].append(v)
        for k, v in cl.items():  clustering_results[k].append(v)

        # print running averages every 100 batches
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{N_SAMPLES}]  ARI={np.mean(clustering_results['ari']):.3f}  F1={np.mean(adjacency_results['f1']):.3f}")

    # aggregate across all batches; ARI/NMI/purity reported as mean ± std
    results = {
        'adjacency_prediction': {
            'auroc': round(float(np.mean(adjacency_results['auroc'])), 4),
            'auprc': round(float(np.mean(adjacency_results['auprc'])), 4),
            'f1':    round(float(np.mean(adjacency_results['f1'])),    4),
        },
        'fragment_clustering': {
            'ari_mean':    round(float(np.mean(clustering_results['ari'])),    4),
            'ari_std':     round(float(np.std(clustering_results['ari'])),     4),
            'nmi_mean':    round(float(np.mean(clustering_results['nmi'])),    4),
            'nmi_std':     round(float(np.std(clustering_results['nmi'])),     4),
            'purity_mean': round(float(np.mean(clustering_results['purity'])), 4),
            'purity_std':  round(float(np.std(clustering_results['purity'])),  4),
        }
    }

    out_path = os.path.join(run_dir, 'test_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n── Results ──────────────────────────────────────────")
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
