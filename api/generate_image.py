from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from gradio_client import Client
import base64

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    try:
        client = Client("Heartsync/NSFW-Uncensored")
        result = client.predict(
            prompt=request.prompt,
            negative_prompt="text, talk bubble, low quality, watermark, signature",
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=7,
            num_inference_steps=28,
            api_name="/infer"
        )

        with open(result, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return {"image": encoded}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 👇 Required for Vercel to recognize this as a handler
handler = Mangum(app)
