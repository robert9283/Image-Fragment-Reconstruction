"""
Clustering algorithms and evaluation metrics for fragment assignment.

Given the pairwise similarity matrix produced by FragmentAdjacencyPredictor,
this module converts it into discrete cluster assignments (one cluster per
source image) and measures the quality of those assignments.

Functions:
    cluster          -- Route to spectral clustering or (balanced) k-means.
    _balanced_kmeans -- Capacity-constrained k-means via Hungarian assignment.
    compute_metrics  -- Compute ARI, NMI, and purity from predicted labels.
    _purity          -- Fraction of fragments assigned to their majority cluster.
"""
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

N_CLUSTERS = 10


def cluster(model_output, k=N_CLUSTERS, n_per_cluster=None):
    """
    Assign fragments to clusters based on model output.

    Routes to the appropriate algorithm depending on output shape:
      - (N, N) similarity matrix → spectral embedding + k-means (or balanced k-means)
      - (N, D) embedding matrix  → k-means directly on the embeddings

    When n_per_cluster is set, balanced k-means is used to enforce exactly
    n_per_cluster fragments per cluster (requires N == k * n_per_cluster).

    Args:
        model_output (np.ndarray): Either an (N, N) pairwise similarity matrix
            or an (N, D) embedding matrix.
        k (int): Number of clusters. Default 10 (one per source image).
        n_per_cluster (int or None): If set, each cluster is forced to contain
            exactly this many fragments. Pass GRID*GRID (=16) for balanced
            clustering. Default None (unconstrained).

    Returns:
        np.ndarray: Integer cluster label for each fragment, shape (N,).
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
    Capacity-constrained k-means where every cluster gets exactly `size` points.

    Initialises centroids with standard k-means, then iterates:
      1. Compute squared distances from every point to every centroid.
      2. Solve a linear assignment problem over k*size slots (size slots per cluster)
         to find the globally cheapest assignment that respects capacities.
      3. Recompute centroids as the mean of assigned points.
    Stops early if centroids do not change between iterations.

    Args:
        features (np.ndarray): Feature matrix of shape (N, D), where N == k * size.
        k (int): Number of clusters.
        size (int): Required number of points per cluster.
        n_iter (int): Maximum number of iterations. Default 20.

    Returns:
        np.ndarray: Integer cluster label for each point, shape (N,).

    Raises:
        AssertionError: If features.shape[0] != k * size.
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
    """
    Compute clustering quality metrics given predicted and true fragment labels.

    Args:
        pred_labels (np.ndarray): Predicted cluster assignments, shape (N,).
        true_labels (np.ndarray): Ground-truth source-image indices, shape (N,).

    Returns:
        dict: {
            'ari':    Adjusted Rand Index (float, range roughly -1 to 1),
            'nmi':    Normalised Mutual Information (float, range 0 to 1),
            'purity': Cluster purity (float, range 0 to 1),
        }
    """
    return {
        'ari':    adjusted_rand_score(true_labels, pred_labels),
        'nmi':    normalized_mutual_info_score(true_labels, pred_labels),
        'purity': _purity(true_labels, pred_labels),
    }


def _purity(true_labels, pred_labels):
    """
    Compute cluster purity: fraction of fragments in their cluster's majority class.

    For each predicted cluster, counts how many fragments belong to the most
    common true source image, then divides the total count by N.

    Args:
        true_labels (np.ndarray): Ground-truth source-image indices, shape (N,).
        pred_labels (np.ndarray): Predicted cluster assignments, shape (N,).

    Returns:
        float: Purity score in [0, 1]. Higher is better.
    """
    total = 0
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        total += np.bincount(true_labels[mask]).max()
    return total / len(true_labels)
