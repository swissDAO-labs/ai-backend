FROM pytorch/pytorch:latest
# nvidia/cuda:11.4.0-base-ubuntu20.04

RUN apt-get update -y
RUN apt-get install -y libpoppler-dev poppler-utils ffmpeg libsm6 libxext6

ENV HOME=/root
WORKDIR /root

COPY . /root/
# RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# --mount=type=cache,target=/root/.cache/pip 

# This downloads and runs the models
# TODO: Double check paths
# RUN --mount=type=cache,target=/root/.cache/huggingface python -c "from model import StableDiffusionXlLight; StableDiffusionXlLight()"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--backlog", "5", "--timeout-keep-alive", "500", "--workers", "1"]

