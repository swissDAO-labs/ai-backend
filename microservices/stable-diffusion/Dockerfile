FROM us-docker.pkg.dev/colab-images/public/runtime

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

