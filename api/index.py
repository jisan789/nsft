from fastapi import FastAPI, HTTPException
from gradio_client import Client
from pydantic import BaseModel
import base64
from io import BytesIO

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_image/")
async def generate_image(request: ImageRequest):
    try:
        client = Client("Heartsync/NSFW-Uncensored")
        # This returns the image bytes or URL, you need to verify what it returns
        # The old code assumed a filename; here we want raw bytes

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

        # If 'result' is image bytes or a PIL image, convert it to base64
        # If it's a filepath, you'll need to download or handle differently

        # Assuming result is a PIL.Image or bytes, let's handle bytes:
        if isinstance(result, bytes):
            image_bytes = result
        elif hasattr(result, "read"):  # file-like
            image_bytes = result.read()
        else:
            # If result is a URL (string), fetch it? Vercel does not allow downloads inside serverless function easily
            # So let's error for now
            raise HTTPException(status_code=500, detail="Unexpected result type from Gradio Client")

        base64_data = base64.b64encode(image_bytes).decode("utf-8")

        return {"image": base64_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
