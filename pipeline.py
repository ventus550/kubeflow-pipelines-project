# +
from google.cloud import aiplatform
from kfp.registry import RegistryClient
from kfp import compiler, dsl
from kfp.dsl import (
    Dataset,
    Input,
    Model,
    If,
    Else,
)

import google_cloud_pipeline_components.v1.custom_job.utils as custom_job
import components
from src.secrets import configs

# -

# ## Static Configuration

TENSORBOARD_ID = configs.tensorboard
TRAIN_LOCATION = configs.location
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"

# ## Pipeline definition

# +
tensorboard = aiplatform.Tensorboard(
    location=TRAIN_LOCATION, project=configs.project, tensorboard_name=TENSORBOARD_ID
)

training_config = dict(
    component_spec=components.train_model,
    display_name=configs.model,
    tensorboard=tensorboard.resource_name,
    base_output_directory=configs.pipeline_directory,
    service_account=configs.service_account,
)

gpu = dict(accelerator_type=ACCELERATOR_TYPE)


@dsl.pipeline(
    pipeline_root=configs.pipeline_directory,
    name=configs.model,
)
def pipeline(
    dataset: str = f"{configs.data_directory}/words.npz",
    epochs: int = 10,
    upload: bool = False,
    upload_threshold: float = 0.0,
    accelerator: bool = False,
    pretrained_model: Input[Model] = None,
):
    importer = dsl.importer(
        artifact_uri=dataset,
        artifact_class=Dataset,
        reimport=False,
    )

    data = components.split_data(ratio=0.1, dataset=importer.output)
    train = data.outputs["train"]
    test = data.outputs["test"]

    train_model_gpu_op = custom_job.create_custom_training_job_op_from_component(
        **(training_config | gpu)
    )
    train_model_cpu_op = custom_job.create_custom_training_job_op_from_component(
        **training_config
    )

    def branch(train_model_op):
        # replace with OneOf once https://github.com/kubeflow/pipelines/issues/10271 is resolved
        model = train_model_op.outputs["trained_model"]
        components.shap_explainer(rows=4, cols=5, dataset=test, model=model)
        metric = components.metrics(dataset=test, model=model).outputs["Output"]
        with If(upload and metric > upload_threshold, name="metric > threshold"):
            components.upload_model(model=model)

    with If(accelerator == True, name="gpu"):
        train_model_gpu = train_model_gpu_op(
            epochs=epochs, dataset=train, location=TRAIN_LOCATION
        )
        branch(train_model_gpu)
    with Else(name="cpu"):
        train_model_cpu = train_model_cpu_op(
            epochs=epochs, dataset=train, location=TRAIN_LOCATION
        )
        branch(train_model_cpu)


# -


# ## Compilation and upload

client = RegistryClient(host=configs.artifactory)
compiler.Compiler().compile(pipeline_func=pipeline, package_path=configs.pipeline_name)
client.upload_pipeline(file_name=configs.pipeline_name, tags=["latest"])
