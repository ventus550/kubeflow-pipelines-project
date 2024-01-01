# +
from google.cloud import aiplatform
from kfp.registry import RegistryClient
from kfp import compiler, dsl
from kfp.dsl import (
    Artifact, Dataset, Input, InputPath,
    Model, Output, OutputPath, component,
    ParallelFor
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
    dataset: str = f"{configs.data_directory}/classification.npz",
    epochs: int = 10,
    foo: Input[Dataset] = None
):
    
    importer = dsl.importer(
        artifact_uri=dataset,
        artifact_class=Dataset,
        reimport=False,
    )
    
    data = components.split_data(ratio=0.1, dataset=importer.output)
    
    train_classifier_op = custom_job.create_custom_training_job_op_from_component(
        component_spec = components.train_classifier,
        display_name = configs.model,
        tensorboard = tensorboard.resource_name,
        base_output_directory = configs.pipeline_directory,
        service_account = configs.service_account
    )
    
    train_classifier = train_classifier_op(epochs=epochs, dataset=data.outputs["train"], location = configs.location)
    
    components.visualize(model=train_classifier.outputs["classifier"])


# -


# ## Compilation and upload

client = RegistryClient(host=configs.artifactory)
compiler.Compiler().compile(pipeline_func=pipeline, package_path=configs.pipeline_name)
client.upload_pipeline(file_name=configs.pipeline_name, tags=["latest"])


