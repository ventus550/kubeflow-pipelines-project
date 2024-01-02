import io
import base64
import random
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple
from google.cloud import storage


def gsdownload(filename: str, gs_full_resource_uri: str = "gs://<bucket>/<resource>"):
    storage_client = storage.Client()
    separator = "/"
    urisplit = gs_full_resource_uri.split(separator)
    bucket = storage_client.bucket(urisplit[2])
    resource = separator.join(urisplit[3:])
    blob = bucket.blob(resource)
    blob.download_to_filename(filename)


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


class Sample(NamedTuple):
    image: np.ndarray
    label: str
    value: int
    
    def show(self, plot=plt):
        add_title = getattr(plot, "set_title", getattr(plot, "title", None))
        plot.imshow(self.image, cmap="gray")
        add_title(self.label)
        plot.axis("off")


class SampleBatch(list):
    def __init__(self, X: np.ndarray, Y: list[str], V: list[int]):
        assert len(X) == len(Y) == len(V) >= 12
        
        super().__init__(Sample(*tup) for tup in zip(X, Y, V))
    
    def show(self):
        _, subs = plt.subplots(3, 4, figsize=(15, 4))
        for sample, subplot in zip(random.choices(self, k=12), subs.ravel()):
            sample.show(subplot)