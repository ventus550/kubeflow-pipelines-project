

# Overview
This repository contains a machine learning project implemented using Kubeflow on Google Cloud Platform. Kubeflow is an open-source platform for deploying, monitoring, and managing machine learning models on Kubernetes. This project leverages various GCP services to streamline the machine learning workflow, from data preparation to model deployment.

![https://imgur.com/gallery/6nhjYcL](https://i.imgur.com/YTHnfLJ.png)

# Relevant features
- Data ingestion
- Accelerated training
- Model diagnostics
- Model retention

# Google cloud services
The following services and APIs need to be enabled and configured accordingly.

### Secret Manager
Secret Manager enables one to store otherwise sensitive data. We use it for our project configuration.
The file is expected to be in json format and resemble the following:
```
{
	"bucket": "gs://<bucket>",
	"location": "<region>",
	"project": "<project name>",
	"artifactory": "<kubeflow artifact registry name>",
	"dockfactory": "<docker artifact registry name>",
	"model": "<model name>",
  	"tensorboard": "<tensorboard-id>",
	"service_account": "<id>-compute@developer.gserviceaccount.com"
}
```
Also, all other services should adhere to the configs provided.


### Vertex AI
- Pipelines (Kubeflow)
- Tensorboard
- Model Registry


### Cloud Storage
One standard class storage bucket with training data and a separate one for the tensorboard instance.

### Artifact Registry
Docker and kubeflow registry preferably with a frequent clean up rule.

### Cloud Build
To fully automate the CI/CD process the project can be connected to the github repository through Cloud Build.
This way docker images and the pipeline template are always up to date and synched with the code changes in the github repository.

