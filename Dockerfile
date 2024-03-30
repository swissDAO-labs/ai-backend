# syntax = docker/dockerfile:experimental
FROM us-docker.pkg.dev/colab-images/public/runtime

RUN apt-get update -y
RUN apt-get install -y libpoppler-dev poppler-utils ffmpeg libsm6 libxext6

ENV HOME=/root
WORKDIR /root

COPY . /root/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# This downloads and runs the models
# TODO: Double check paths
RUN --mount=type=cache,target=/root/.cache/huggingface python -c "from model import StableDiffusionXlLight; StableDiffusionXlLight()"

