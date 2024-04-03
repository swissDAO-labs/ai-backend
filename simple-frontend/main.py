from __future__ import annotations

import io
import os
import base64

import gradio as gr
import requests
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


# Function to call the API and get the image
def get_image(seed, prompt):
    url = os.getenv("API_STABLEDIFFUSION")  # Adjust if necessary
    if not url:
        raise ValueError("API_STABLEDIFFUSION environment variable is not set.")

    headers = {"Content-Type": "application/json"}
    data = {"seed": seed, "prompt": prompt}
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        # Assuming the API returns an image as a byte array under "response" key
        base64_image = response.json()["response"]
        image_bytes = base64.b64decode(base64_image)
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
    else:
        # Error handling or fallback
        return "API call failed with status code: {}".format(response.status_code)


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
