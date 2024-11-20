import torch
from diffusers import StableDiffusionPipeline
import os
from PIL import Image

# Configuration
model_path = "./saved_models/epoch-10"  # Path to your trained model
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)
num_images = 5  # Number of images to generate
prompt = "A high-resolution photo of a male face"  # Prompt used during training

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

# Load the fine-tuned Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    safety_checker=None,  # Disable safety checker for custom models
    feature_extractor=None  # Disable feature extractor if not needed
).to(device)

# Generate images
for i in range(num_images):
    with torch.no_grad():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Save the image
    image_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
    image.save(image_path)
    print(f"Saved {image_path}")

print("Image generation complete.")
