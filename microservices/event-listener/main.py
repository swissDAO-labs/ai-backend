from __future__ import annotations

import os
import time
from typing import Optional

import requests
from web3 import Web3
from utils import get_abi, setup_logger
from hexbytes import HexBytes
from pydantic import Field, BaseModel
from web3.datastructures import AttributeDict


logger = setup_logger("event-listener")

INFURA_URL = "https://mainnet.infura.io/v3/"
DEFAULT_TIMEOUT = 10


# TODO:
SEED = 42
PROMPT = "Peaky Blinders NFT. Faces are not directly visible. No text."
CONTRACT_ADDRESS = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"  # uniswap_v2
MODEL_URL = "http://localhost:2500/predict"


class ContractError(Exception):
    """Exception raised for errors during contract interaction."""


# TODO: match the contract event specification in ABI
# example PairCreated:
event = AttributeDict(
    {
        "args": AttributeDict(
            {
                "token0": "0x1eb26dCa9a89d93A6117DeE5d2F9b96374Dafb22",
                "token1": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "pair": "0x983c59FAE90EF4E8a8fFF25A3128dA301BEE1eAd",
                "": 315227,
            }
        ),
        "event": "PairCreated",
        "logIndex": 515,
        "transactionIndex": 176,
        "transactionHash": HexBytes(
            "0xa9b8e57e4b9c407e5f1d4910aaaca171c914ad7ce05e5ad542bcd1148984f2c2"
        ),
        "address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        "blockHash": HexBytes(
            "0xefbda768f06e45f0386627daeaf6faa586f63805ec755b5f2f5d6f644718e9f2"
        ),
        "blockNumber": 19582979,
    }
)

# NOTE: assumed actual args
# class Args(BaseModel):
#     prompt: str


class Args(BaseModel):
    token0: str
    token1: str
    pair: str
    extra_info: Optional[int] = Field(alias="")  # unnamed field


class PairCreated(BaseModel, arbitrary_types_allowed=True):
    args: Args
    event: str
    logIndex: int
    transactionIndex: int
    transactionHash: HexBytes
    address: str
    blockHash: HexBytes
    blockNumber: int


def main(INFURA_PROJECT_ID: str) -> None:
    web3 = Web3(Web3.HTTPProvider(f"{INFURA_URL}{INFURA_PROJECT_ID}"))

    contract_address = web3.to_checksum_address(CONTRACT_ADDRESS)
    contract_abi = get_abi(contract_address)

    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
    event = next(filter(lambda x: x.event_name == PairCreated.__name__, contract.events), None)
    if event is None:  # application design error
        raise ContractError(f"No '{PairCreated.__name__}' event found in contract's events.")

    event_filter = event.create_filter(fromBlock="latest", address=contract_address)

    # Infura has 100k calls / day for free
    # 60 * 60 * 24 = 86400 seconds / day, with 10 sec delay only 8640 calls daily
    while True:
        for event in event_filter.get_new_entries():
            logger.info(f"Event: {event}")
            generate_image(PairCreated(**event))
        time.sleep(DEFAULT_TIMEOUT)


def generate_image(event: PairCreated) -> None:
    # TODO: Retrieve user prompt from event
    # tx_hash = event.transactionHash

    payload = {
        # "seed": event.args.seed,
        # "prompt": event.args.prompt
        "seed": SEED,
        "prompt": PROMPT,
    }

    # Use stable diffusion microservice to generate image
    headers = {"Content-Type": "application/json"}
    response = requests.post(MODEL_URL, json=payload, headers=headers)
    if not response.ok:
        message = f"API call failed with status code: {response.status_code}"
        logger.error(message)
        return message
    img_str = response.json().get("response")
    if not img_str:
        message = f"No image returned from event: {event}"
        logger.error(message)
        return message

    # TODO: Store image on Arweave
    # arweave_reference = store_image_on_arweave(img_str)


if __name__ == "__main__":
    if not (INFURA_PROJECT_ID := os.environ.get("INFURA_PROJECT_ID")):
        raise EnvironmentError("INFURA_PROJECT_ID not found")
    main(INFURA_PROJECT_ID)
