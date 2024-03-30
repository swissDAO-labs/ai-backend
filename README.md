

## TODOs

- Fix requirements.txt versions
- Upload Dockerfile into docker registry


## Quick Start:

```
uvicorn --port 2503 --host 0.0.0.0 --backlog 100 --timeout-keep-alive 500 --workers 1 main:app
```