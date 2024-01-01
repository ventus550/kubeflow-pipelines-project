import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import io
import base64
from typing import Union
from pathlib import Path


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


def capture_image() -> str:
    """Retrieves matplotlib image in binary."""
    image = plt.gcf()
    buf = io.BytesIO()
    # save image to memory
    image.savefig(buf, format='png', bbox_inches="tight")
    binary_image = buf.getvalue()

    image_base64_utf8_str = base64.b64encode(binary_image).decode('utf-8')
    image_type = "png"
    dataurl = f'data:image/{image_type};base64,{image_base64_utf8_str}'
    return dataurl
