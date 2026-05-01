import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

N_CLUSTERS = 10


def cluster(model_output, k=N_CLUSTERS):
    """
    Routes to the right clustering algorithm based on model output shape.
      (N, D) where D != N  → k-means on embeddings
      (N, N)               → spectral clustering on similarity matrix
    Returns predicted labels (N,).
    """
    n = model_output.shape[0]
    if model_output.ndim == 2 and model_output.shape[1] != n:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        return km.fit_predict(model_output)
    else:
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        return sc.fit_predict(model_output)


def compute_metrics(pred_labels, true_labels):
    """Returns dict with ARI, NMI, and purity."""
    return {
        'ari':    adjusted_rand_score(true_labels, pred_labels),
        'nmi':    normalized_mutual_info_score(true_labels, pred_labels),
        'purity': _purity(true_labels, pred_labels),
    }


def _purity(true_labels, pred_labels):
    total = 0
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        total += np.bincount(true_labels[mask]).max()
    return total / len(true_labels)
