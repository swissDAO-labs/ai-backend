




## TODOs

- Fix requirements.txt versions
- Upload Dockerfile into docker registry
- vscode .devcontainer is not working
- `microservices/stable-diffusion/Dockerfile` -> line `RUN --mount=type=cache,target=/root/.cache/huggingface python -c "from model import StableDiffusionXlLight; StableDiffusionXlLight()"` runs, but it doesn't cache it properly to docker-disk.



## Quick Start:

```
uvicorn --port 2503 --host 0.0.0.0 --backlog 100 --timeout-keep-alive 500 --workers 1 main:app
```