import os
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

try:
    api_key = os.environ.get('STABILITY_KEY')
    if not api_key:
        raise ValueError("STABILITY_KEY environment variable not set or empty.")
    
    stability_api = client.StabilityInference(
        key=api_key,
        verbose=True,
        engine="stable-diffusion-xl-1024-v1-0",
    )
    print("Successfully connected to Stability AI API.")
except Exception as e:
    print(f"Error connecting to Stability AI: {e}")
    exit()

# --- Image Generation Parameters ---
prompt_text = "A fantasy castle on a floating island, vibrant colors, digital art"
negative_prompt_text = "blurry, ugly, deformed, watermark, signature, text" # Our new negative prompt!

# Define some aspect ratio options (width, height)
aspect_ratios = {
    "1:1_square": (1024, 1024),
    "16:9_widescreen": (1344, 768), # Let's try this one
    "9:16_tall": (768, 1344)
}
selected_aspect_ratio = "16:9_widescreen" # Change this to test others
image_width, image_height = aspect_ratios[selected_aspect_ratio]

output_filename = f"castle_widescreen.png" # Change filename based on prompt/ratio

print(f"\nGenerating image for prompt: '{prompt_text}'")
if negative_prompt_text:
    print(f"Negative prompt: '{negative_prompt_text}'")
print(f"Aspect ratio: {selected_aspect_ratio} ({image_width}x{image_height})")

# The core API call to generate the image
answers = stability_api.generate(
    prompt=[
        generation.Prompt(text=prompt_text, parameters=generation.PromptParameters(weight=1.0)),
        generation.Prompt(text=negative_prompt_text, parameters=generation.PromptParameters(weight=-1.0)) # Negative weight
    ] if negative_prompt_text else prompt_text, # Only add negative prompt if it's not empty
    # seed=12345, # Optional: use a seed for reproducibility
    steps=50,
    cfg_scale=7.0,
    width=image_width,   # Use our selected width
    height=image_height, # Use our selected height
    samples=1,
    sampler=generation.SAMPLER_K_DPMPP_2M
)

# --- Process and Save the Image ---
# (The saving part of the code remains the same, just make sure to use the new output_filename)
image_saved = False
safety_filter_activated = False
last_artifact_type_info = "No specific artifact processed." 

for resp in answers:
    for artifact in resp.artifacts:
        last_artifact_type_info = f"Artifact type: {artifact.type}, Finish reason: {artifact.finish_reason}"
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filter and could not be processed. "
                "Please modify the prompt and try again.")
            print("Safety filter activated. No image generated.")
            safety_filter_activated = True
            break 
        elif artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(output_filename) # Use the new filename
            print(f"Image successfully saved as {output_filename}")
            image_saved = True
            break
    if image_saved or safety_filter_activated:
        break

if not image_saved and not safety_filter_activated:
    print(f"No image artifact found in the response. Last artifact info: {last_artifact_type_info}")

print("\nScript finished.")