import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

def get_spectrogram(audio):
    #audio = tf.cast(audio, tf.float32)
    #audio = audio / 32768.0

    #audio = audio[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)

    stft = tf.signal.stft(
        audio,
        frame_length=255,
        frame_step=128
    )

    spectrogram = tf.abs(stft)
    return spectrogram

(train_ds, val_ds, test_ds), ds_info = tfds.load(
    "speech_commands",
    split=["train", "validation", "test"],
    as_supervised=True,
    with_info=True,
)

label_names = ds_info.features["label"].names

gyldige_labels = [
    label_names.index("up"),
    label_names.index("down"),
    label_names.index("left"),
    label_names.index("right"),
]

label_map = {
    label_names.index("up"): 0,
    label_names.index("down"): 1,
    label_names.index("left"): 2,
    label_names.index("right"): 3,
}

def behold_kun_retninger(audio, label):
    return tf.reduce_any(tf.equal(label, gyldige_labels))

def preprocess(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = spectrogram[..., tf.newaxis]

    ny_label = tf.case(
        [(tf.equal(label, k), lambda v=v: tf.constant(v, dtype=tf.int64)) for k, v in label_map.items()],
        exclusive=True
    )

    return spectrogram, ny_label

train_ds = train_ds.filter(behold_kun_retninger).map(preprocess)
val_ds = val_ds.filter(behold_kun_retninger).map(preprocess)
test_ds = test_ds.filter(behold_kun_retninger).map(preprocess)

batch_size = 32

train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

for spectrograms, labels in train_ds.take(1):
    print("Spectrogram batch shape:", spectrograms.shape)
    print("Labels:", labels[:10].numpy())