FROM python:3.9

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /app

ADD requirements.txt main.py /app/

RUN pip install -r /app/requirements.txt

CMD ["python", "/workspace/main.py"]