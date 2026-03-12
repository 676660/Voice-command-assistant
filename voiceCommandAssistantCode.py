import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

(train_ds, val_ds, test_ds), ds_info = tfds.load(
    "speech_commands",
    split=["train", "validation", "test"],
    as_supervised=True,
    with_info=True,
)

print(ds_info)