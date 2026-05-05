"""
Abstract base class for all models in this project.

Any new model must subclass BaseModel and implement the four abstract methods
below. The training loop in main.py and the submission scripts only interact
with models through this interface, so a new model can be plugged in by adding
a branch to load_model() in main.py without touching any other code.

Classes:
    BaseModel -- Abstract interface for fragment-based self-supervised models.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract interface that every model in this project must implement.

    The training loop calls train_step() at every iteration and get_output()
    at every evaluation step. save() and load() handle checkpointing.
    """

    @abstractmethod
    def train_step(self, fragments, labels, adjacency):
        """
        Perform one gradient update on a single training batch.

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).
            labels (np.ndarray): Integer source-image index per fragment,
                shape (N,). Used to construct same-image training targets.
            adjacency (np.ndarray): Binary adjacency matrix of shape (N, N).
                Entry (i, j) = 1 if fragments i and j are spatially adjacent
                within the same source image.

        Returns:
            float: Scalar training loss for this iteration.
        """

    @abstractmethod
    def get_output(self, fragments):
        """
        Compute the model output used for clustering.

        Returns either an embedding matrix or a similarity matrix; the
        clustering module inspects the shape to choose the right algorithm:
          - (N, D) embedding matrix  → k-means
          - (N, N) similarity matrix → spectral clustering

        Args:
            fragments (np.ndarray): Fragment array of shape (N, 16, 16, 3).

        Returns:
            np.ndarray: Either an (N, D) embedding matrix or an (N, N)
                pairwise similarity matrix.
        """

    @abstractmethod
    def save(self, path):
        """
        Save model weights to disk.

        Args:
            path (str): Destination path without file extension.
        """

    @abstractmethod
    def load(self, path):
        """
        Load model weights from disk.

        Args:
            path (str): Source path without file extension.
        """
