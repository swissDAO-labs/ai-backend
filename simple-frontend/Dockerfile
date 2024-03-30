FROM python:3.9

ARG GRADIO_SERVER_PORT=2503
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /app

ADD requirements.txt main.py /app/

RUN pip install -r /app/requirements.txt
