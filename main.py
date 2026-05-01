import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'to_share', 'src'))

import json
import yaml
import numpy as np
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency
from src.clustering import cluster, compute_metrics


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(name):
    if name == 'fragment-adjacency-predictor':
        from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor
        return FragmentAdjacencyPredictor()
    # TODO: add further models here
    else:
        raise ValueError(f"Unknown model: '{name}'")


def main():
    cfg = load_config()

    dataset   = Imagenet64(cfg['data_path'])
    train_gen = dataset.datagen_cls(batch_size=cfg['n_images'], ds='train', augmentation=True)
    val_gen   = dataset.datagen_cls(batch_size=cfg['n_images'], ds='test',  augmentation=False)

    model = load_model(cfg['model'])

    log_path = os.path.join(os.path.dirname(__file__), 'training_log.jsonl')
    log_file = open(log_path, 'w')

    best_ari        = -1.0
    patience_count  = 0

    for iteration in range(cfg['max_iterations']):
        images, _ = next(train_gen)
        fragments, labels = extract_fragments(np.array(images))
        adjacency = build_adjacency(n_images=cfg['n_images'])

        loss = model.train_step(fragments, labels, adjacency)

        if (iteration + 1) % cfg['eval_every'] == 0:
            val_images, _ = next(val_gen)
            val_fragments, val_labels = extract_fragments(np.array(val_images))
            val_adjacency = build_adjacency(n_images=cfg['n_images'])

            adj_metrics = model.evaluate_adjacency(val_fragments, val_adjacency)
            cl_metrics  = compute_metrics(cluster(model.get_output(val_fragments)), val_labels)

            improved = cl_metrics['ari'] > best_ari
            if improved:
                best_ari       = cl_metrics['ari']
                patience_count = 0
                model.save(cfg['checkpoint_path'])

            log_entry = {
                'iteration': iteration + 1,
                'loss':      round(loss, 6),
                'precision': round(adj_metrics['precision'], 4),
                'recall':    round(adj_metrics['recall'],    4),
                'f1':        round(adj_metrics['f1'],        4),
                'ari':       round(cl_metrics['ari'],        4),
                'nmi':       round(cl_metrics['nmi'],        4),
                'purity':    round(cl_metrics['purity'],     4),
            }
            log_file.write(json.dumps(log_entry) + '\n')
            log_file.flush()

            print(
                f"iter {iteration+1:5d}  loss={loss:.4f}"
                f"  F1={adj_metrics['f1']:.3f}  prec={adj_metrics['precision']:.3f}  rec={adj_metrics['recall']:.3f}"
                f"  ARI={cl_metrics['ari']:.3f}  NMI={cl_metrics['nmi']:.3f}"
                f"  {'*' if improved else ''}"
            )

            patience_count += 1
            if patience_count >= cfg['patience']:
                print(f"Early stopping: no improvement for {cfg['patience']} evaluations.")
                break

    log_file.close()
    print(f"Best ARI={best_ari:.3f}. Checkpoint saved to {cfg['checkpoint_path']}")
    print(f"Training log saved to {log_path}")


if __name__ == '__main__':
    main()
