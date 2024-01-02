from kfp.dsl import component, Dataset, Input, Output, Model, Markdown
from src.secrets import configs

@component(base_image=configs.keras_image)
def upload_model(
    model: Input[Model],
    container_image: str = "europe-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-15:latest"
):                                                         
    from google.cloud import aiplatform
    from src import aitoolkit
    from src.secrets import configs
    
    model = aitoolkit.load(model.path)
    model.save(model.path, save_format="tf")

    print("uploading to model registry")
    model = aiplatform.Model.upload(
        display_name=configs.model,
        artifact_uri=model.path,
        serving_container_image_uri=container_image
    )

