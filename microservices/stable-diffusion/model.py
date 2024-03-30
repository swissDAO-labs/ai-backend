"""
    Majority of this file was imported from
        - https://colab.research.google.com/drive/1SILA5bbgTKzr335K8RgqApqRJM7MchXV#scrollTo=ufD_d64nr08H
"""

import torch

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import base64
from io import BytesIO

class StableDiffusionXlLight: 
    """
        Stable Diffusion XL Light connector
    """

    def __init__(self):
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
        print("Is CUDA is available: ", torch.cuda.is_available())
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

    def predict(self, seed, prompt) -> any:
        """
            Generate a picture from a given random seed (must be an integer), and a prompt
        """

        images = self.pipe(
            prompt = prompt,
            guidance_scale = 0.0,
            num_inference_steps = self.num_inference_steps,
            generator = torch.Generator(self.device).manual_seed(seed),
        ).images
        print("Output (1) is: ", images)
        buffered = BytesIO()
        images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print("Output (2) is: ", img_str)
        return img_str


if __name__ == "__main__":
    print("Testing StableDiffusionXlLight")
    prompt = "Peaky Blinders NFT. Faces are not directly visible. No text."
    # seed = random.randint(0, sys.maxsize)
    seed = 42
    model = StableDiffusionXlLight()
    output = model.predict(seed, prompt)
    print(output)










