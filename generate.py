# generate.py

import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image

def generate_faces(
    model_path: str,
    prompt: str,
    num_images: int,
    output_dir: str,
    device: torch.device,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    disable_safety_checker: bool = True
):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    print(f"Text Encoder Hidden Size: {text_encoder.config.hidden_size}")  # Should be 768

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)

    print("Loading noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    print(f"Loading fine-tuned UNet from {model_path}...")
    unet = UNet2DConditionModel.from_pretrained(model_path).to(device)
    print(f"UNet Loaded. UNet Cross Attention Dim: {unet.config.cross_attention_dim}")

    print("Initializing Stable Diffusion Pipeline...")
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None if disable_safety_checker else "CompVis/stable-diffusion-safety-checker",
        feature_extractor=None if disable_safety_checker else "CompVis/stable-diffusion-safety-checker"
    ).to(device)

    # Disable gradient calculations for inference
    torch.set_grad_enabled(False)

    # Optional: Enable memory-efficient attention if using GPU
    # pipe.enable_attention_slicing()

    print("Starting image generation...")
    for i in range(1, num_images + 1):
        print(f"Generating image {i}/{num_images}...")
        try:
            # Encode the prompt
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            print(f"Tokenized Prompt Shape: {text_inputs.input_ids.shape}")  # Should be [1, 77]

            # Generate image
            output = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            image = output.images[0]

            # Save the image
            image_path = os.path.join(output_dir, f"generated_image_{i}.png")
            image.save(image_path)
            print(f"Saved image to {image_path}")

        except Exception as e:
            print(f"Error generating image {i}: {e}")

    print("Image generation complete.")

def main():
    # Configuration
    model_path = "./fine_tuned_model_all_images/final_model"  # Path to Model
    output_dir = "./generated_images_6"
    os.makedirs(output_dir, exist_ok=True)
    num_images = 5000  # Number of images to generate
    prompt = "A high-resolution photo of a face"  # Prompt used for generation

    # Device configuration
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # Parameters for generation
    num_inference_steps = 50  # More steps = higher quality
    guidance_scale = 7.5      # Higher guidance scale for adherence to prompt

    # Generate images
    generate_faces(
        model_path=model_path,
        prompt=prompt,
        num_images=num_images,
        output_dir=output_dir,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        disable_safety_checker=True  # Set to False if you want to enable safety checks, had to disable due to model flagging generated images
    )

if __name__ == "__main__":
    main()
