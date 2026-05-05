"""
ImageNet-64 dataset loader.

Loads the downsampled 64×64 ImageNet dataset from pickle files and exposes
an infinite batch generator used by the training loop and evaluation scripts.

The dataset is stored in two subdirectories:
    data/train_data/  — one or more pickle batches (CIFAR-style format)
    data/dev_data/    — dev_data_batch_1 used as the test split

Constants:
    N_CLASSES  -- Number of ImageNet classes (1000).
    IMAGE_SIZE -- Pixel dimensions of each image (64, 64).

Classes:
    Imagenet64 -- Dataset loader with an infinite batch generator.
"""
import numpy as np
from pathlib import Path
import os
import pickle
import tensorflow as tf
from tqdm import tqdm

N_CLASSES = 1000
IMAGE_SIZE = (64, 64)


def normalize_img(img_batch):
    """
    Scale a batch of uint8 images to float32 in [0, 1].

    Conversion is performed on the CPU to avoid placing tensors on the GPU
    unnecessarily during data loading.

    Args:
        img_batch: Array-like of shape (N, 64, 64, 3) with values in [0, 255].

    Returns:
        tf.Tensor: Float32 tensor of shape (N, 64, 64, 3) with values in [0, 1].
    """
    with tf.device("cpu:0"):
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        normalized_tensor = img_tensor / 255.0
    return normalized_tensor


def init_augmentor():
    """
    Create a Keras ImageDataGenerator with the fixed augmentation policy.

    Applied augmentations: random rotation (±20°), small horizontal and
    vertical shifts (±10%), and random channel intensity shifts (±0.2).
    fill_mode='reflect' avoids black border artefacts at image edges.

    Returns:
        tf.keras.preprocessing.image.ImageDataGenerator: Configured augmentor.
    """
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=10,
        # zoom_range=0.1,
        channel_shift_range=0.2,
        fill_mode="reflect",
    )


class Imagenet64(object):
    """
    Loader for the downsampled 64×64 ImageNet dataset.

    On construction, all training pickle files are read from
    data_path/train_data/ and concatenated into a single in-memory array.
    The test split is read from data_path/dev_data/dev_data_batch_1.
    Labels are converted from 1-indexed to 0-indexed integers.

    Attributes:
        data_path (Path): Root directory of the dataset.
        data (dict): Dictionary with keys 'x_train', 'y_train',
            'x_test', 'y_test' holding the full in-memory arrays.
    """

    def __init__(self, data_path):
        """
        Load all training and test data into memory.

        Args:
            data_path (str | Path): Root directory containing train_data/
                and dev_data/ subdirectories.

        Raises:
            AssertionError: If the number of training images does not match
                the number of training labels, or if the training split does
                not contain all 1000 classes.
        """
        self.data_path = Path(str(data_path))

        # Load all training pickle batches and concatenate them.
        train_files = os.listdir(self.data_path / "train_data")
        x_train = []
        y_train = []
        for train_file in train_files:
            with open(self.data_path / "train_data" / train_file, "rb") as fo:
                data = pickle.load(fo)
                # Data is stored as (N, 3, 64, 64); transpose to (N, 64, 64, 3).
                x = (
                    data["data"]
                    .reshape((data["data"].shape[0], 3, 64, 64))
                    .transpose((0, 2, 3, 1))
                )
                # Labels are 1-indexed in the pickle files; shift to 0-indexed.
                y = np.array(data["labels"]) - 1

                x_train.append(x)
                y_train.append(y)
        del x, y
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        assert x_train.shape[0] == len(y_train)

        # Load the single dev batch as the test split.
        with open(self.data_path / "dev_data/dev_data_batch_1", "rb") as fo:
            data = pickle.load(fo)
            x_test = (
                data["data"]
                .reshape((data["data"].shape[0], 3, 64, 64))
                .transpose((0, 2, 3, 1))
            )
            y_test = np.array(data["labels"]) - 1

        self.data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

        # Sanity check: training split must cover all 1000 classes.
        n_classes = 1000
        assert (
            len(np.unique(self.data["y_train"])) == n_classes and
            len(np.unique(self.data["y_train"])) >= len(np.unique(self.data["y_test"]))
        )

    def datagen_cls(self, batch_size, ds="train", augmentation=False):
        """
        Infinite generator that yields (images, labels) batches.

        Shuffles the dataset at the start of each epoch using a deterministic
        seed (the epoch index) for reproducibility. Incomplete final batches
        (fewer than batch_size images) are silently skipped. When augmentation
        is enabled, the Keras augmentor is applied after normalisation.

        Args:
            batch_size (int): Number of images per batch.
            ds (str): Dataset split — 'train' or 'test'. Default 'train'.
            augmentation (bool): Whether to apply random augmentations.
                Default False.

        Yields:
            tuple[tf.Tensor, np.ndarray]:
                x -- Float32 image batch, shape (batch_size, 64, 64, 3),
                     values in [0, 1].
                y -- Integer class labels, shape (batch_size,). These are
                     ImageNet class indices and are not used by the
                     self-supervised training loop.
        """
        epoch_i = 0
        ds_size = len(self.data[f"y_{ds}"])

        augmentor = init_augmentor()

        while True:
            # New permutation each epoch; seed ensures reproducibility.
            np.random.seed(epoch_i)
            perm = np.random.permutation(ds_size)

            for i in range(0, ds_size, batch_size):
                selection = perm[i : i + batch_size]

                # Skip the last incomplete batch.
                if len(selection) < batch_size:
                    continue

                x, y = self.data[f"x_{ds}"][selection], self.data[f"y_{ds}"][selection]

                # Normalise to [0, 1] before optional augmentation.
                x = normalize_img(x)

                if augmentation:
                    x, y = next(augmentor.flow(x, y, batch_size=batch_size))

                # x: images
                # y: labels - you can ignore, not important here
                yield x, y

            epoch_i += 1


if __name__ == "__main__":
    ds = Imagenet64(
        "path_to_data_folder",
        n_decomposed_features=None,
    )
    dg = ds.datagen_cls(1024, augmentation=True)

    for i in tqdm(range(1000)):
        next(dg)
