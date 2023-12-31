#!/bin/bash     
PROJECT_ID="protocell-404013"
REGION="europe-central2"
REPOSITORY="kubeflow-images"
IMAGE='keras'
IMAGE_TAG='keras:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG

# Create repository in the artifact registry
gcloud beta artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION

# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev

# Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG