from __future__ import annotations

import os

import gradio as gr
import requests
from dotenv import load_dotenv


load_dotenv()


# Function to call the API and get the image
def get_image(seed: int, prompt: str) -> str:
    url = os.getenv("API_STABLEDIFFUSION")  # Adjust if necessary
    if not url:
        raise ValueError("API_STABLEDIFFUSION environment variable is not set.")

    headers = {"Content-Type": "application/json"}
    data = {"seed": seed, "prompt": prompt}
    response = requests.post(url, json=data, headers=headers)

    if response.ok:
        # return IPFS URL where image is pinned
        return response.json()["response"]["url"]
    else:
        # Error handling or fallback
        return f"API call to model failed with status code: {response.status_code}"

    # requests.get(url=url)


if __name__ == "__main__":
    # Gradio interface
    demo = gr.Interface(
        fn=get_image,
        inputs=[gr.Textbox(label="Seed"), gr.Textbox(label="Prompt")],
        outputs=gr.Image(type="pil"),
        title="Image Generation via FastAPI",
        description="Enter a seed and a prompt to generate an image.",
    )

    demo.launch()
