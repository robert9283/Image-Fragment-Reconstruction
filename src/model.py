from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def train_step(self, fragments, labels, adjacency):
        """
        fragments:  (N, 16, 16, 3) float32
        labels:     (N,) int — source image index, for constructing training signal
        adjacency:  (N, N) float32 — binary adjacency matrix, for graph-based approaches

        Returns scalar loss value.
        """

    @abstractmethod
    def get_output(self, fragments):
        """
        fragments: (N, 16, 16, 3) float32

        Returns either:
          (N, D) embedding matrix  — will be clustered with k-means
          (N, N) similarity matrix — will be clustered with spectral clustering
        The evaluate module inspects the shape to select the right algorithm.
        """

    @abstractmethod
    def save(self, path):
        """Save model weights to path."""

    @abstractmethod
    def load(self, path):
        """Load model weights from path."""
