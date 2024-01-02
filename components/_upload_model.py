from kfp.dsl import component, Dataset, Input, Output, Model, Markdown
from src.secrets import configs


@component(
    base_image=configs.keras_image,
    packages_to_install=["google-cloud-secret-manager>=2.17.0"],
)
def upload_model(
    model: Input[Model],
    container_image: str = "europe-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-15:latest",
):
    import keras
    from google.cloud import aiplatform
    from src.secrets import configs
    from src import aitoolkit

    path = model.path
    model = aitoolkit.load(path)
    model = keras.Model(model.input, model.output)
    model.save(path, save_format="tf")

    print("uploading to model registry")
    model = aiplatform.Model.upload(
        display_name=configs.model,
        artifact_uri=path,
        serving_container_image_uri=container_image,
        location=configs.location,
        project=configs.project,
        staging_bucket=configs.bucket,
    )
