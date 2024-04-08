# AI Backend

The AI Backend is a microservice designed to facilitate the generation and minting of NFTs using generative art AI models. It enables artists to provide unique prompts for specific collections, which serve as templates for NFT creation. Users can personalize their NFTs by providing additional input, which is used alongside with their wallet address to customize the generated art.


## Requirements:
- [poetry](https://python-poetry.org/)
- [Docker](https://github.com/docker)
- [Docker-compose](https://github.com/docker/compose)


## Usage

To start the microservice, use the following command:

```shell
docker compose up --build
```

To run the simple-frontend, first install the required dependencies:

```shell
poetry install --no-root --only frontend
poetry shell
```

Then, execute the following command:

```shell
python simple-frontend/main.py
```

This deploys the server on the localhost, allowing one to interact with the model


## Development setup

To set up a clean development environment, run the following command:

```shell
make new_env
```

In order to generate the `requirements.txt` files required for the Dockerfiles from the pyproject.toml, run the following:

```shell
make requirements
```
