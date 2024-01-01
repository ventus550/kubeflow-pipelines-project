import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from itertools import groupby
from src.utils import SampleBatch
import numpy as np


def save(model, path: Path | str, metadata={}, frozen=False):
    path = Path(path)
    if frozen:
        metadata["frozen"] = True
        model = keras.models.Model(
            inputs=model.input, outputs=model.output, name=model.name
        )
        model.trainable = False
    weights = model.get_weights()
    config = model.get_config()
    config["name"] = {"name": config["name"], **metadata}
    model = tf.keras.models.Model.from_config(config)
    model.set_weights(weights)
    model.save(path)
    return model


def load(path: Path | str):
    path = Path(path)
    model = keras.models.load_model(path)
    model.meta = model.get_config()["name"]
    model._name = path.name
    return model


def format_data(X, Y, classes, maxlen=None):
    maxlen = maxlen or len(max(Y, key=len))
    tokenize = keras.layers.StringLookup(vocabulary=classes, num_oov_indices=0)
    Y = tokenize(tf.strings.unicode_split(Y, input_encoding="UTF-8"))
    Y = Y.to_tensor(default_value=-1, shape=(None, maxlen))
    X = np.expand_dims(X, -1)
    return X, Y


def to_sparse_filtered(tensor, filtered_value, dtype=tf.int32) -> tf.SparseTensor:
    sparse = tf.cast(tf.sparse.from_dense(tensor), dtype=dtype)
    return tf.sparse.retain(sparse, tf.not_equal(sparse.values, filtered_value))


def edit_distance(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.int32)
    input_size = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
    input_size = tf.fill([batch_size], input_size)

    decode, _ = keras.backend.ctc_decode(y_pred, input_length=input_size, greedy=True)
    sparse_truths = to_sparse_filtered(y_true, -1)
    sparse_decode = to_sparse_filtered(decode[0], -1)

    edit_distances = tf.edit_distance(sparse_decode, sparse_truths, normalize=False)
    return tf.reduce_mean(edit_distances)


def ctc_loss(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.int32)
    pred_size = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
    clss_size = tf.cast(tf.shape(y_pred)[2], dtype=tf.int32)
    true_size = tf.cast(tf.shape(y_true)[1], dtype=tf.int32)
    pred_size = tf.fill([batch_size, 1], pred_size)
    true_size = tf.fill([batch_size, 1], true_size)

    # Make padding non-negative since ctc_batch_cost doesn't like it
    # Non-class tokens will be discarded by the function anyways
    y_true = tf.where(y_true == -1, clss_size, tf.cast(y_true, dtype=tf.int32))
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, pred_size, true_size)


def best_path_search(timesteps):
    return np.argmax(timesteps, axis=1)


def ctc_decode(classes, timesteps, path_algorithm=best_path_search):
    path = path_algorithm(timesteps)
    characters = [classes[key] for key, _ in groupby(path) if key < len(classes)]
    return "".join(characters)


def batch_prediction(model, X, Y):
    P = model.predict(X)
    L = [ctc_decode(characters, p) for p in P]
    D = [edit_distance([y], [p]) for y, p in zip(Y, P)]
    return SampleBatch(X, L, D)

characters = [
	'!', '"', '#', '&', "'", '(', ')', '*', '+', ',',
	'-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
	'7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D',
	'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
	'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
	'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
	'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
	's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]