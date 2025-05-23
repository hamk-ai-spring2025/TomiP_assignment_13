import os
import io
import warnings
import uuid # For generating unique filenames
from fastapi import FastAPI
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# --- FastAPI App Setup ---
app = FastAPI()

# --- Stability AI Configuration ---
# Ensure STABILITY_KEY is set in the environment where Uvicorn runs
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443' 
stability_api = None # We will initialize this when the app starts

# --- Helper: Initialize Stability API ---
# This function will run once when FastAPI starts up
@app.on_event("startup")
async def startup_event():
    global stability_api
    api_key = os.environ.get('STABILITY_KEY')
    if not api_key:
        print("FATAL ERROR: STABILITY_KEY environment variable not set.")
        print("Please set STABILITY_KEY before running the Uvicorn server.")
        # In a real app, you might want to prevent startup or handle this more gracefully
        # For now, we'll let it proceed, but image generation will fail.
        return 
        
    try:
        stability_api = client.StabilityInference(
            key=api_key,
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0",
        )
        print("Successfully connected to Stability AI API on startup.")
    except Exception as e:
        print(f"Error connecting to Stability AI on startup: {e}")
        stability_api = None # Ensure it's None if connection failed

# --- Aspect Ratio Definitions ---
ASPECT_RATIOS = {
    "1:1_square": (1024, 1024),
    "16:9_widescreen": (1344, 768),
    "9:16_tall": (768, 1344),
    "3:2_landscape": (1216, 832),
    "2:3_portrait": (832, 1216),
}

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Image Generator API!"}

@app.post("/generate-image/") # Changed to POST, more appropriate for actions
async def generate_image_endpoint(
    prompt: str, 
    negative_prompt: str = "",  # Optional, defaults to empty string
    aspect_ratio_key: str = "1:1_square" # Optional, defaults to square
):
    if stability_api is None:
        api_key_status = "NOT SET or connection FAILED" if not os.environ.get('STABILITY_KEY') else "Set, but connection FAILED on startup"
        return {
            "error": "Stability API not initialized. Check server logs.",
            "detail": f"STABILITY_KEY status: {api_key_status}. Please ensure it's set correctly when starting Uvicorn."
        }

    if aspect_ratio_key not in ASPECT_RATIOS:
        return {"error": "Invalid aspect_ratio_key", "available_keys": list(ASPECT_RATIOS.keys())}

    image_width, image_height = ASPECT_RATIOS[aspect_ratio_key]
    
    # Generate a unique filename for the image
    unique_id = uuid.uuid4() # Generates a random unique ID
    output_filename = f"generated_image_{unique_id}.png"

    print(f"\nReceived request to generate image:")
    print(f"  Prompt: '{prompt}'")
    if negative_prompt:
        print(f"  Negative Prompt: '{negative_prompt}'")
    print(f"  Aspect Ratio: {aspect_ratio_key} ({image_width}x{image_height})")
    print(f"  Output Filename: {output_filename}")

    try:
        answers = stability_api.generate(
            prompt=[
                generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=1.0)),
                generation.Prompt(text=negative_prompt, parameters=generation.PromptParameters(weight=-1.0))
            ] if negative_prompt else prompt,
            steps=50,
            cfg_scale=7.0,
            width=image_width,
            height=image_height,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        image_saved_successfully = False
        safety_filter_triggered = False
        error_message = "No image generated."

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn("Request activated the API's safety filter.")
                    error_message = "Safety filter activated. Please modify prompt."
                    safety_filter_triggered = True
                    break
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    img.save(output_filename)
                    print(f"Image successfully saved as {output_filename}")
                    image_saved_successfully = True
                    break
            if image_saved_successfully or safety_filter_triggered:
                break
        
        if image_saved_successfully:
            return {
                "message": "Image generated successfully!",
                "filename": output_filename,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio_key
            }
        else:
            return {"error": error_message, "details": "Could not generate or save image."}

    except Exception as e:
        print(f"Error during image generation: {e}")
        return {"error": "Failed to generate image.", "details": str(e)}