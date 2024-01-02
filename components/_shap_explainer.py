from kfp.dsl import component, Dataset, Input, Output, Model, Markdown
from src.secrets import configs

@component(base_image=configs.keras_image, packages_to_install=["shap==0.44"])
def shap_explainer(
    rows: int,
    cols: int,
    dataset: Input[Dataset],
    model: Input[Model],
    explanation: Output[Markdown]
):                                                         
    import cv2
    import shap
    import numpy
    import keras
    import tensorflow as tf
    
    from src import aitoolkit
    from src.utils import capture_image
    
    characters = aitoolkit.characters
    X, Y = numpy.load(dataset.path + ".npz").values()
    X, Y = aitoolkit.format_data(X, Y, characters)
    model = aitoolkit.load(model.path)
    
    # repurpose model for "classification"
    classifier = keras.models.Model(model.input, tf.reduce_max(model.output, axis=1)[:, :len(characters)])

    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    explainer = shap.Explainer(classifier, masker, output_names=list(characters))
    indices = numpy.random.randint(len(X), size=rows)
    shap_values = explainer(X[indices], max_evals=100, batch_size=1000, outputs=shap.Explanation.argsort.flip[:cols])

    labels = shap_values.output_names
    if len(numpy.shape(shap_values.output_names)) == 1:
        # shap is an awful library and thus requires awful fixes
        labels = numpy.reshape(labels * rows, (rows, cols))

    shap.image_plot(shap_values, labels=labels)
    open(explanation.path, 'w').write(f"![Image]({capture_image()})")

