from fastapi import FastAPI, HTTPException
from gradio_client import Client
from pydantic import BaseModel
import base64
import httpx

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

        # Log for debugging (in Vercel logs or locally)
        print(f"Result (type={type(result)}): {result}")

        if isinstance(result, str):
            if result.startswith("http"):
                # It's an image URL, fetch the image bytes
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.get(result)
                    response.raise_for_status()
                    image_bytes = response.content

                base64_data = base64.b64encode(image_bytes).decode("utf-8")
                return {"image": base64_data}
            else:
                # It's some other string — just return it in the response for now
                return {"result": result}

        elif isinstance(result, bytes):
            base64_data = base64.b64encode(result).decode("utf-8")
            return {"image": base64_data}

        else:
            raise HTTPException(status_code=500, detail=f"Unexpected result type: {type(result)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
