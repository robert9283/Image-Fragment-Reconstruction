import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'to_share', 'src'))

import json
import time
import datetime
import shutil
import yaml
import numpy as np
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency, GRID
from src.clustering import cluster, compute_metrics


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def n_neg_over_n_pos(n_images, grid):
    """Constant ratio of non-adjacent to adjacent pairs for the given grid."""
    n_pos = n_images * (2 * grid * (grid - 1))   # horizontal + vertical adjacencies
    n_pairs = (n_images * grid * grid) * (n_images * grid * grid - 1) // 2
    n_neg = n_pairs - n_pos
    return n_neg / n_pos


def load_model(name, cfg):
    if name == 'fragment-adjacency-predictor':
        from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor
        from src.fragments import GRID
        beta = float(cfg.get('beta', 1.0))
        ratio = n_neg_over_n_pos(cfg['n_images'], GRID)
        return FragmentAdjacencyPredictor(
            pos_weight_adj  = beta * ratio,
            lambda_adj      = float(cfg.get('lambda_adj',  1.0)),
            pos_weight_same = float(cfg.get('pos_weight_same', 1.0)),
            lambda_same     = float(cfg.get('lambda_same', 0.0)),
        )
    # TODO: add further models here
    else:
        raise ValueError(f"Unknown model: '{name}'")


def setup_run_dir(cfg):
    """
    Create runs/{run_name}/, snapshot the config, refresh the runs/latest symlink.
    Returns the absolute path of the run directory.
    """
    run_name = cfg.get('run_name') or datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_dir  = os.path.join(PROJECT_ROOT, 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    latest = os.path.join(PROJECT_ROOT, 'runs', 'latest')
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(run_name, latest)

    return run_dir, run_name


def append_results_summary(run_name, cfg, best_ari, best_iter, total_minutes, log_path):
    """Append a one-line summary of this run to the top-level results.jsonl."""
    with open(log_path) as f:
        entries = [json.loads(line) for line in f]
    last = entries[-1] if entries else {}

    summary = {
        'run':           run_name,
        'model':         cfg.get('model'),
        'beta':          cfg.get('beta', 1.0),
        'lambda_adj':    cfg.get('lambda_adj',  1.0),
        'lambda_same':   cfg.get('lambda_same', 0.0),
        'best_ari':      round(best_ari, 4),
        'iter_at_best':  best_iter,
        'final_auroc':   last.get('auroc'),
        'final_auprc':   last.get('auprc'),
        'final_ari':     last.get('ari'),
        'final_nmi':     last.get('nmi'),
        'duration_min':  round(total_minutes, 1),
        'n_eval_batches':    cfg.get('n_eval_batches', 1),
        'balanced_clustering': cfg.get('balanced_clustering', False),
        'notes':         cfg.get('notes', ''),
    }
    results_path = os.path.join(PROJECT_ROOT, 'results.jsonl')
    with open(results_path, 'a') as f:
        f.write(json.dumps(summary) + '\n')


def main():
    cfg = load_config()
    run_dir, run_name = setup_run_dir(cfg)
    print(f"Run: {run_name}")
    print(f"Dir: {run_dir}")

    dataset   = Imagenet64(cfg['data_path'])
    train_gen = dataset.datagen_cls(batch_size=cfg['n_images'], ds='train', augmentation=True)
    val_gen   = dataset.datagen_cls(batch_size=cfg['n_images'], ds='test',  augmentation=False)

    model = load_model(cfg['model'], cfg)

    log_path        = os.path.join(run_dir, 'training_log.jsonl')
    checkpoint_path = os.path.join(run_dir, 'model')
    log_file        = open(log_path, 'w')
    train_start     = time.time()

    best_ari        = -1.0
    best_iter       = 0
    patience_count  = 0
    train_time_acc  = 0.0

    for iteration in range(cfg['max_iterations']):
        images, _ = next(train_gen)
        fragments, labels = extract_fragments(np.array(images))
        adjacency = build_adjacency(n_images=cfg['n_images'])

        t0   = time.time()
        loss = model.train_step(fragments, labels, adjacency)
        train_time_acc += time.time() - t0

        if (iteration + 1) % cfg['eval_every'] == 0:
            n_eval = cfg.get('n_eval_batches', 1)
            n_per_cluster = GRID * GRID if cfg.get('balanced_clustering', False) else None

            t0 = time.time()
            adj_acc = {'auroc': [], 'auprc': []}
            cl_acc  = {'ari': [], 'nmi': [], 'purity': []}
            for _ in range(n_eval):
                val_images, _ = next(val_gen)
                val_fragments, val_labels = extract_fragments(np.array(val_images))
                val_adjacency = build_adjacency(n_images=cfg['n_images'])
                a = model.evaluate_adjacency(val_fragments, val_adjacency)
                c = compute_metrics(cluster(model.get_output(val_fragments),
                                            n_per_cluster=n_per_cluster), val_labels)
                for k in adj_acc: adj_acc[k].append(a[k])
                for k in cl_acc:  cl_acc[k].append(c[k])
            adj_metrics = {k: float(np.mean(v)) for k, v in adj_acc.items()}
            cl_metrics  = {k: float(np.mean(v)) for k, v in cl_acc.items()}
            eval_time   = time.time() - t0

            improved = cl_metrics['ari'] > best_ari
            if improved:
                best_ari       = cl_metrics['ari']
                best_iter      = iteration + 1
                patience_count = 0
                model.save(checkpoint_path)

            log_entry = {
                'iteration':       iteration + 1,
                'timestamp':       round(time.time() - train_start, 1),
                'loss':            round(loss, 6),
                'auroc':           round(adj_metrics['auroc'],     4),
                'auprc':           round(adj_metrics['auprc'],     4),
                'ari':             round(cl_metrics['ari'],        4),
                'nmi':             round(cl_metrics['nmi'],        4),
                'purity':          round(cl_metrics['purity'],     4),
                'train_time_s':    round(train_time_acc,           2),
                'eval_time_s':     round(eval_time,                3),
            }
            train_time_acc = 0.0
            log_file.write(json.dumps(log_entry) + '\n')
            log_file.flush()

            print(
                f"iter {iteration+1:5d}  loss={loss:.4f}"
                f"  AUROC={adj_metrics['auroc']:.3f}  AUPRC={adj_metrics['auprc']:.3f}"
                f"  ARI={cl_metrics['ari']:.3f}  NMI={cl_metrics['nmi']:.3f}"
                f"  {'*' if improved else ''}"
            )

            patience_count += 1
            if patience_count >= cfg['patience']:
                print(f"Early stopping: no improvement for {cfg['patience']} evaluations.")
                break

    log_file.close()
    total_minutes = (time.time() - train_start) / 60
    append_results_summary(run_name, cfg, best_ari, best_iter, total_minutes, log_path)
    print(f"Best ARI={best_ari:.3f} at iteration {best_iter}.")
    print(f"Run directory: {run_dir}")
    print(f"Summary appended to {os.path.join(PROJECT_ROOT, 'results.jsonl')}")


if __name__ == '__main__':
    main()
