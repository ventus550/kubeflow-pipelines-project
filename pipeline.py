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
# -

# ## Configuration

BUCKET_URI = f"gs://protocell"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root"
LOCATION = "europe-central2"
PROJECT = "protocell-404013"
DATA = f"{BUCKET_URI}/data"
ARTIFACTORY = f"https://{LOCATION}-kfp.pkg.dev/{PROJECT}/kubeflows"
PIPELINE_TEMPLATE_NAME = "classifier.yaml"


# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Tensorboard
aiplatform.init(location=LOCATION, project=PROJECT, staging_bucket=BUCKET_URI)
tensorboard = aiplatform.Tensorboard(location=LOCATION, project=PROJECT, tensorboard_name = "2373397003624251392")


# ## Pipeline definition

@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name="classification",
)
def pipeline(dataset: str = f"{DATA}/classification.npz", epochs: int = 10, foo: Input[Dataset] = None):
    
    importer = dsl.importer(
        artifact_uri=dataset,
        artifact_class=Dataset,
        reimport=False,
    )
    
    data = components.split_data(ratio=0.1, dataset=importer.output)
    
    train_classifier_op = custom_job.create_custom_training_job_op_from_component(
        component_spec = components.train_classifier,
        display_name = "train_classfier_display_name",
        tensorboard = tensorboard.resource_name,
        base_output_directory = PIPELINE_ROOT,
        service_account = "429426973958-compute@developer.gserviceaccount.com"
    )
    
    train_classifier = train_classifier_op(epochs=epochs, dataset=data.outputs["train"], location = LOCATION)
    

    components.visualize(model=train_classifier.outputs["classifier"])


# ## Compilation and upload

# +
client = RegistryClient(host=ARTIFACTORY)

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=PIPELINE_TEMPLATE_NAME
)

templateName, versionName = client.upload_pipeline(
  file_name=PIPELINE_TEMPLATE_NAME,
  tags=["latest"],
  extra_headers={"description":"testing"}
)
