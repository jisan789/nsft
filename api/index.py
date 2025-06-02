from fastapi import FastAPI, HTTPException
from gradio_client import Client
from pydantic import BaseModel
import base64

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate_image/")
async def generate_image(request: ImageRequest):
    try:
        client = Client("Heartsync/NSFW-Uncensored")
        response = client.predict(
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

        # Extract the file path string from the response dictionary
        filepath = response["result"]

        # Open the image file and encode it to base64
        with open(filepath, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")

        return {"image": base64_data}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
