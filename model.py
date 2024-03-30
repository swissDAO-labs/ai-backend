"""
    Majority of this file was imported from
        - https://colab.research.google.com/drive/1SILA5bbgTKzr335K8RgqApqRJM7MchXV#scrollTo=ufD_d64nr08H
"""

import torch

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class StableDiffusionXlLight: 
    """
        Stable Diffusion XL Light connector
    """

    def __init__(self):
        # Pick:
        # -    2, 4 or 8 steps for lora,
        # - 1, 2, 4 or 8 steps for unet.
        num_inference_steps = 4

        # Prefer "unet" over "lora" for quality.
        use_lora = False
        model_type = "lora" if use_lora else "unet"

        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = f"sdxl_lightning_{num_inference_steps}step_{model_type}.safetensors"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fetch the model from disk
        # At build, this should've been already downloaded
        self.unet = UNet2DConditionModel.from_config(
            base,
            subfolder="unet",
        ).to(self.device, torch.float16)

        self.unet.load_state_dict(
            load_file(
                hf_hub_download(
                    repo,
                    ckpt,
                ),
                device=self.device,
            ),
        )
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=self.unet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        if use_lora:
            self.pipe.load_lora_weights(hf_hub_download(repo, ckpt))
            self.pipe.fuse_lora()

            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )

        pass

    def predict(self, seed, prompt) -> any:
        """
            Generate a picture from a given random seed (must be an integer), and a prompt
        """

        images = self.pipe(
            prompt = prompt,
            guidance_scale = 0.0,
            num_inference_steps = self.num_inference_steps,
            generator = torch.Generator("cuda").manual_seed(seed),
        ).images
        # images[0].save("output.jpg")
        # TODO: Double check the format here
        print("Output is: ", images)
        print("Output shape is: ", images.shape)
        return images[0]


if __name__ == "__main__":
    print("Testing StableDiffusionXlLight")
    prompt = "Peaky Blinders NFT. Faces are not directly visible. No text."
    # seed = random.randint(0, sys.maxsize)
    seed = 42
    model = StableDiffusionXlLight()
    output = model.predict(seed, prompt)
    print(output)










