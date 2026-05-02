import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

N_CLUSTERS = 10


def cluster(model_output, k=N_CLUSTERS, n_per_cluster=None):
    """
    Routes to the right clustering algorithm based on model output shape.
      (N, D) where D != N  → (balanced) k-means on embeddings
      (N, N)               → spectral clustering on similarity matrix

    If `n_per_cluster` is given, enforces clusters of exactly that size via a
    Hungarian assignment of points to k×n_per_cluster slots after k-means
    initialisation.
    """
    n = model_output.shape[0]
    is_similarity = model_output.ndim == 2 and model_output.shape[1] == n

    # produce a feature matrix to cluster on
    if is_similarity:
        features = SpectralEmbedding(n_components=k, affinity='precomputed',
                                     random_state=42).fit_transform(model_output)
    else:
        features = model_output

    if n_per_cluster is None:
        if is_similarity:
            return SpectralClustering(n_clusters=k, affinity='precomputed',
                                      random_state=42).fit_predict(model_output)
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)

    return _balanced_kmeans(features, k=k, size=n_per_cluster)


def _balanced_kmeans(features, k, size, n_iter=20):
    """
    Balanced k-means: each cluster gets exactly `size` points.

    Iterates: (1) compute distances to centroids, (2) Hungarian-assign points
    to k×size slots with capacity `size` per cluster, (3) update centroids.
    """
    assert features.shape[0] == k * size, \
        f"need exactly {k * size} points, got {features.shape[0]}"

    # initialise centroids via regular k-means
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(features)
    centroids = km.cluster_centers_

    n = features.shape[0]
    for _ in range(n_iter):
        # cost[i, c*size + s] = dist(point i, centroid c)
        dists = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=-1)  # (n, k)
        cost = np.repeat(dists, size, axis=1)  # (n, k*size)

        row_ind, col_ind = linear_sum_assignment(cost)
        labels = (col_ind // size)[np.argsort(row_ind)]

        # update centroids
        new_centroids = np.array([features[labels == c].mean(axis=0) for c in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels


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
