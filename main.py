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

    model = load_model(cfg['model'])

    for iteration in range(cfg['n_iterations']):
        images, _ = next(train_gen)
        fragments, labels = extract_fragments(np.array(images))
        adjacency = build_adjacency(n_images=cfg['n_images'])

        loss = model.train_step(fragments, labels, adjacency)

        if (iteration + 1) % cfg['eval_every'] == 0:
            model_output = model.get_output(fragments)
            pred_labels  = cluster(model_output)
            metrics      = compute_metrics(pred_labels, labels)
            print(f"iter {iteration+1:4d}  loss={loss:.4f}  ARI={metrics['ari']:.3f}  NMI={metrics['nmi']:.3f}  purity={metrics['purity']:.3f}")

    model.save(cfg['checkpoint_path'])
    print(f"Checkpoint saved to {cfg['checkpoint_path']}")


if __name__ == '__main__':
    main()
