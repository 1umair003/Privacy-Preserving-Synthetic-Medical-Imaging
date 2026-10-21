"""Data loading, label encoding and tf.data pipelines for the CNN classifiers.

Expected folder layout (any of real / synthetic / hybrid / test):

    data_dir/
        Bacterial_Pneumonia/  *.png
        COVID-19/             *.png
        Normal/               *.png
        Tuberculosis/         *.png
        Viral_Pneumonia/      *.png
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder

from common import config

PREPROCESS = {
    "ResNet50": keras.applications.resnet50.preprocess_input,        # BGR mean subtraction
    "InceptionV3": keras.applications.inception_v3.preprocess_input,  # x/127.5 - 1
    "Xception": keras.applications.xception.preprocess_input,          # x/127.5 - 1
}


# ---------------------------------------------------------------- file listing
def list_images(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (file paths, class-name labels) for every image under data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths, labels = [], []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in config.IMAGE_EXTENSIONS:
                paths.append(str(f))
                labels.append(class_dir.name)
    if not paths:
        raise ValueError(f"No images found under {data_dir}")
    return np.array(paths), np.array(labels)


# ---------------------------------------------------------------- label encoding (Sec. 3.1)
def fit_label_encoder(labels: np.ndarray, save_path: str | Path) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(sorted(set(labels)))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as fh:
        pickle.dump(encoder, fh)
    return encoder


def load_label_encoder(path: str | Path) -> LabelEncoder:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- class weights (Eq. 27)
def class_weights(y: np.ndarray, num_classes: int) -> dict:
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    return {c: float(total / (num_classes * max(n, 1))) for c, n in enumerate(counts)}


# ---------------------------------------------------------------- augmentation (Sec. 3.2)
class RandomBrightnessScale(keras.layers.Layer):
    """Multiply pixel values by a random factor in [low, high] (0-255 range)."""

    def __init__(self, low: float, high: float, **kwargs):
        super().__init__(**kwargs)
        self.low, self.high = low, high

    def call(self, x, training=None):
        if not training:
            return x
        factor = tf.random.uniform([], self.low, self.high)
        return tf.clip_by_value(x * factor, 0.0, 255.0)


def build_augmenter(seed: int = config.SEED) -> keras.Sequential:
    a = config.AUGMENTATION
    layers = [
        keras.layers.RandomRotation(a["rotation_degrees"] / 360.0, fill_mode="nearest", seed=seed),
        keras.layers.RandomTranslation(a["shift"], a["shift"], fill_mode="nearest", seed=seed),
        keras.layers.RandomShear(x_factor=a["shear"], y_factor=a["shear"], fill_mode="nearest", seed=seed),
        keras.layers.RandomZoom(a["zoom"], a["zoom"], fill_mode="nearest", seed=seed),
        RandomBrightnessScale(*a["brightness"]),
    ]
    if a["horizontal_flip"]:
        layers.append(keras.layers.RandomFlip("horizontal", seed=seed))
    return keras.Sequential(layers, name="augmentation")


# ---------------------------------------------------------------- tf.data pipeline
def make_dataset(
    paths: np.ndarray,
    y: np.ndarray,
    model_name: str,
    num_classes: int,
    batch_size: int = config.BATCH_SIZE,
    training: bool = False,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """Decode -> resize -> (augment) -> model-specific preprocessing -> one-hot."""
    size = config.INPUT_SIZE[model_name]
    preprocess = PREPROCESS[model_name]
    augmenter = build_augmenter(seed) if training else None

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size), method="bilinear")
        img = tf.cast(img, tf.float32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, y.astype(np.int32)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augmenter is not None:
        ds = ds.map(lambda x, l: (augmenter(x, training=True), l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, l: (preprocess(x), tf.one_hot(l, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
