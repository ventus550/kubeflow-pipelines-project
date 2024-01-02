# +
from google.cloud import aiplatform
from kfp.registry import RegistryClient
from kfp import compiler, dsl
from kfp.dsl import (
    Artifact, Dataset, Input, InputPath,
    Model, Output, OutputPath, component,
    ParallelFor, If
)

import google_cloud_pipeline_components.v1.custom_job.utils as custom_job
import components
from src.secrets import configs
# -

# ## Pipeline definition

# +
TENSORBOARD_ID = "2373397003624251392"

tensorboard = aiplatform.Tensorboard(
    location=configs.location,
    project=configs.project,
    tensorboard_name = TENSORBOARD_ID
)

@dsl.pipeline(
    pipeline_root=configs.pipeline_directory,
    name=configs.model,
)
def pipeline(
    dataset: str = f"{configs.data_directory}/words.npz",
    epochs: int = 10,
    upload: bool = True,
    upload_threshold: float = 0.0,
    foo: Input[Dataset] = None
):
    
    importer = dsl.importer(
        artifact_uri=dataset,
        artifact_class=Dataset,
        reimport=False,
    )
    
    data = components.split_data(ratio=0.1, dataset=importer.output)
    train = data.outputs["train"]
    test = data.outputs["test"]
    
    train_model_op = custom_job.create_custom_training_job_op_from_component(
        component_spec = components.train_model,
        display_name = configs.model,
        tensorboard = tensorboard.resource_name,
        base_output_directory = configs.pipeline_directory,
        service_account = configs.service_account
    )
    
    train_model = train_model_op(epochs=epochs, dataset=train, location = configs.location)
    model = train_model.outputs["trained_model"]
    
    components.shap_explainer(rows=4, cols=5, dataset=test, model=model)
    
    metric = components.metrics(dataset=test, model=model).outputs["Output"]

    with If(upload and metric > upload_threshold, name="metric > threshold"):
        components.upload_model(model=model)


# -


# ## Compilation and upload

client = RegistryClient(host=configs.artifactory)
compiler.Compiler().compile(pipeline_func=pipeline, package_path=configs.pipeline_name)
client.upload_pipeline(file_name=configs.pipeline_name, tags=["latest"])


