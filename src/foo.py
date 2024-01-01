import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Union
from PIL import Image, ImageDraw, ImageFont
from itertools import groupby
import numpy as np


def save(model, path: Union[Path, str], metadata={}, frozen=False):
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


def load(path: Union[Path, str]):
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


def text2img(text, font_path="./data/fonts/Quicksand.otf", width=128, height=32):
    # Create a new image
    font_path = str(font_path)
    image = Image.new("RGB", (width, height), color="white")

    # Get a drawing context for the image
    draw = ImageDraw.Draw(image)

    # Set up text
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)

    # Resize font to fit text within image
    while font.getsize(text)[0] < width and font.getsize(text)[1] < height:
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)

    # Decrease font size until text fits within image
    while (
        font.getsize(text)[0] > width or font.getsize(text)[1] > height
    ) and font_size > 0:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # Calculate text position for center alignment
    x = (width - font.getsize(text)[0]) // 2
    y = (height - font.getsize(text)[1]) // 2

    # Draw text onto image
    draw.text((x, y), text, fill="black", font=font)
    return 1 - np.mean(image, axis=2, dtype=float)