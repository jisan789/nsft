from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client
from pathlib import Path
import base64

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_image/")
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

        # If result is a local file path, convert to base64 string
        if isinstance(result, str) and Path(result).exists():
            with open(result, "rb") as image_file:
                image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
        else:
            # Assume result is already a base64 string
            base64_image = result

        return {
            "prompt": request.prompt,
            "image_base64": base64_image
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
