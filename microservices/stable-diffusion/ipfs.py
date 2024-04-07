from __future__ import annotations

import os
import hashlib
from typing import Union
from pathlib import Path

from PIL import Image
from pinatapy import PinataPy
from pydantic import Field, BaseModel


os.environ["PINATA_API_KEY"] = "e16f4a16b347a3898300"
os.environ["PINATA_API_SECRET"] = (
    "2c004947cf8a3fbfb9013843dc40eee7dd22220aef4127dabbabad0e370c3735"
)
GATEWAY = "https://ivory-fancy-hamster-735.mypinata.cloud"
LOCAL_STORAGE = "scarif"


class Result(BaseModel):
    ipfs_hash: str = Field(alias="IpfsHash")
    pin_size: int = Field(alias="PinSize")
    timestamp: str = Field(alias="Timestamp")

    @property
    def url(self) -> str:
        return f"{GATEWAY}/ipfs/{self.ipfs_hash}"

    def dump(self) -> dict[str, Union[int, str]]:
        return {**self.dict(), "url": self.url}


def post_on_ipfs(image: Image) -> Result:
    if not (PINATA_API_KEY := os.environ.get("PINATA_API_KEY")):
        raise ValueError("PINATA_API_KEY not found")
    if not (PINATA_API_SECRET := os.environ.get("PINATA_API_SECRET")):
        raise ValueError("PINATA_API_SECRET not found")

    pinata = PinataPy(PINATA_API_KEY, PINATA_API_SECRET)

    # store locally for upload
    path = Path(LOCAL_STORAGE)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / hashlib.sha256(image.tobytes()).hexdigest()
    image.save(file_path, format="png")

    result = pinata.pin_file_to_ipfs(str(file_path), save_absolute_paths=False)
    return Result(**result)
