from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Union
from pathlib import Path

import requests


ABIValue = Union[str, bool, Dict[str, Union["ABIValue", List["ABIValue"]]]]
ABIData = List[Dict[str, ABIValue]]

DEFAULT_TIMEOUT = 10


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{name}.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_abi(contract_address: str) -> ABIData:
    """Get contract ABI."""

    abi_cache_dir = Path("abi_cache")
    abi_cache_dir.mkdir(parents=True, exist_ok=True)
    abi_cache_file = abi_cache_dir / f"{contract_address}.json"
    if abi_cache_file.exists():
        return json.loads(abi_cache_file.read_text())

    if not (ETHERSCAN_API_KEY := os.environ.get("ETHERSCAN_API_KEY")):
        raise EnvironmentError("ETHERSCAN_API_KEY not found")

    url = "https://api.etherscan.io/"
    params = {"apiKey": ETHERSCAN_API_KEY}
    api_url = f"{url}api?module=contract&action=getabi&address={contract_address}"

    response = requests.get(api_url, params=params, timeout=DEFAULT_TIMEOUT)
    if not response.ok:
        raise ValueError("Failed to retrieve ABI: Etherscan API request failed.")
    if (result := response.json().get("result")) is None:
        raise ValueError("Failed to retrieve ABI: ABI not found.")

    abi_cache_file.write_text(json.dumps(result))
    return json.loads(result)
