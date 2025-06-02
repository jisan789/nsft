from fastapi import FastAPI, HTTPException
from gradio_client import Client
from pydantic import BaseModel

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

        # Debug prints to check output type and content
        print("Result type:", type(result))
        # print("Result content:", result)  # Uncomment if you want to see full output

        # If the result is already a base64 encoded string or a dict with image data, just return it
        # Here we assume it's a base64 string, so return directly:
        return {"image": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
