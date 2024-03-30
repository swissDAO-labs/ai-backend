from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from model import StableDiffusionXlLight

# Instantiate the model
extractor = StableDiffusionXlLight()

app = FastAPI()

@app.get("/health")
async def health_check():
    """
        Example request: 
        ```
            curl http://localhost:2500/health
        ```
    """
    return {"status": "healthy"}

class PredictionParameters(BaseModel):
    seed: int
    prompt: str  # TODO: Could be kept constant across runs

@app.post("/predict")
async def predict(params: PredictionParameters) -> Union[dict]:
    """
        Example request:
        ```
            curl -X POST -H "Content-Type: application/json" -d '{"seed": 42, "prompt": "Peaky Blinders NFT. Faces are not directly visible. No text."}' http://127.0.0.1:2500/predict
        ```
    """
    # TODO: Replace with logging
    print("Parameters ", params)
    try:
        # Read PDF file
        out = extractor.predict(seed=params.seed, prompt=params.prompt)
        return {
            "response": out
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
