FROM python:3.11
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt