import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'to_share', 'src'))

import yaml
import numpy as np
from data import Imagenet64
from src.fragments import extract_fragments, build_adjacency
from src.evaluate import cluster, compute_metrics


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(name):
    if name == 'fragment-adjacency-predictor':
        # TODO: from src.fragment_adjacency_predictor import FragmentAdjacencyPredictor
        # TODO: return FragmentAdjacencyPredictor()
        raise NotImplementedError(f"Model '{name}' not yet implemented")
    # TODO: add further models here
    else:
        raise ValueError(f"Unknown model: '{name}'")


def main():
    cfg = load_config()

    dataset   = Imagenet64(cfg['data_path'])
    train_gen = dataset.datagen_cls(batch_size=cfg['n_images'], ds='train', augmentation=True)
    val_gen   = dataset.datagen_cls(batch_size=cfg['n_images'], ds='test',  augmentation=False)

    model = load_model(cfg['model'])

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

            model_output = model.get_output(val_fragments)
            pred_labels  = cluster(model_output)
            metrics      = compute_metrics(pred_labels, val_labels)

            improved = metrics['ari'] > best_ari
            if improved:
                best_ari       = metrics['ari']
                patience_count = 0
                model.save(cfg['checkpoint_path'])

            print(f"iter {iteration+1:5d}  loss={loss:.4f}  ARI={metrics['ari']:.3f}  NMI={metrics['nmi']:.3f}  purity={metrics['purity']:.3f}  {'*' if improved else ''}")

            patience_count += 1
            if patience_count >= cfg['patience']:
                print(f"Early stopping: no improvement for {cfg['patience']} evaluations.")
                break

    print(f"Best ARI={best_ari:.3f}. Checkpoint saved to {cfg['checkpoint_path']}")


if __name__ == '__main__':
    main()
