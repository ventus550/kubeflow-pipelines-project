FROM europe-west1-docker.pkg.dev/protocell-404013/kubeflow-images/keras:latest
# use "FROM mirror.gcr.io/library/python:3.10" for the first keras image

WORKDIR /
COPY docker/keras/requirements.txt /requirements.txt
COPY src /src
RUN pip install --upgrade pip && pip install -r requirements.txt
ENTRYPOINT [ "bash" ]