FROM mirror.gcr.io/library/python:3.10
WORKDIR /
COPY requirements.txt /requirements.txt
COPY src /src
RUN pip install --upgrade pip && pip install -r requirements.txt
ENTRYPOINT [ "bash" ]