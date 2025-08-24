from __future__ import annotations
import tensorflow as tf
import numpy as np

def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch: int = 64,
    cache: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Tiny tf.data wrapper for array inputs. CPU-friendly."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10_000), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=False)
    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)
