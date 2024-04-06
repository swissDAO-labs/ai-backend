"""
Majority of this file was imported from
    - https://colab.research.google.com/drive/1SILA5bbgTKzr335K8RgqApqRJM7MchXV#scrollTo=ufD_d64nr08H
"""

from __future__ import annotations

from functools import lru_cache

import torch
from ipfs import Result, post_on_ipfs
from utils import setup_logger
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


logger = setup_logger("stable-diffusion-model")


class StableDiffusionXlLight:
    """
    Stable Diffusion XL Light connector
    """

    def __init__(self):
        logger.info(f"Initializing {self.__class__.__name__}")
        # Pick:
        # -    2, 4 or 8 steps for lora,
        # - 1, 2, 4 or 8 steps for unet.
        self.num_inference_steps = 4

        # Prefer "unet" over "lora" for quality.
        self.use_lora = False
        self.model_type = "lora" if self.use_lora else "unet"

        self.base = "stabilityai/stable-diffusion-xl-base-1.0"
        self.repo = "ByteDance/SDXL-Lightning"
        self.ckpt = f"sdxl_lightning_{self.num_inference_steps}step_{self.model_type}.safetensors"
        logger.info(f"Is CUDA is available: {torch.cuda.is_available()}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fetch the model from disk
        # At build, this should've been already downloaded
        self.unet = UNet2DConditionModel.from_config(
            self.base,
            subfolder="unet",
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

        self.unet.load_state_dict(
            load_file(
                hf_hub_download(
                    self.repo,
                    self.ckpt,
                ),
                device=self.device,
            ),
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base,
            unet=self.unet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None,
        ).to(self.device)

        if self.use_lora:
            self.pipe.load_lora_weights(hf_hub_download(self.repo, self.ckpt))
            self.pipe.fuse_lora()

            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )

    @lru_cache()
    def predict(self, seed: int, prompt: str) -> Result:
        """
        Generate a picture from a given random seed (must be an integer), and a prompt
        """
        images = self.pipe(
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).images
        logger.info("Image generated: time to pin it on IPFS!")
        result = post_on_ipfs(images[0])
        logger.info(f"Image pinned on IPFS: {result.ipfs_hash}")
        return result.dump()


if __name__ == "__main__":
    logger.info("Testing StableDiffusionXlLight")
    prompt = "Peaky Blinders NFT. Faces are not directly visible. No text."
    seed = 42
    model = StableDiffusionXlLight()
    result = model.predict(seed, prompt)
    logger.info(f"Result: {result}")
