# train.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm
from data_loader import load_data  # Ensure data_loader.py is in the same directory or in PYTHONPATH

# Configuration Class
class Config:
    # Paths
    data_dir = "./img_align_celeba"  # Directory where CelebA images are stored
    output_dir = "./fine_tuned_model"
    os.makedirs(output_dir, exist_ok=True)

    # Training parameters
    epochs = 5
    batch_size = 16  # Adjust based on GPU memory
    learning_rate = 5e-6
    num_workers = 4
    save_steps = 500  # Save model every 500 steps
    max_grad_norm = 1.0  # Gradient clipping

    # Model parameters
    model_name = "CompVis/stable-diffusion-v1-4"  # Pretrained model
    image_size = 512
    prompt = "A high-resolution photo of a face"  # Generic prompt

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use cuda:0 if available
    print("Using device:", device)


def main():
    config = Config()

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Initialize dataset and dataloader with max_images=10000
    print("Loading dataset...")
    dataloader = load_data(
        img_dir=config.data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        max_images=10000  # Limit to 10,000 images
    )

    # Load Stable Diffusion components
    print("Loading Stable Diffusion components...")
    vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae").to(config.device)
    
    # Correctly load the Text Encoder without subfolder
    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(config.device)
    print(f"Text Encoder Hidden Size: {text_encoder.config.hidden_size}")  # Should be 768
    
    # Load the original UNet
    unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet").to(config.device)
    print(f"UNet Cross Attention Dim: {unet.config.cross_attention_dim}")  # Should be 768

    # Freeze VAE and Text Encoder
    print("Freezing VAE and Text Encoder...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Initialize noise scheduler
    print("Initializing noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name, subfolder="scheduler")

    # Optimizer
    print("Setting up optimizer...")
    optimizer = AdamW(unet.parameters(), lr=config.learning_rate)

    # Training Loop
    step = 0
    total_steps = config.epochs * len(dataloader)
    print("Starting training...")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        epoch_loss = 0.0
        unet.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            # Move data to device
            pixel_values = batch["pixel_values"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)

            # Encode text
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Encode images
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor as per Stable Diffusion

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=config.device)
            timesteps = timesteps.long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)

            # Optimizer step
            optimizer.step()

            # Update loss
            epoch_loss += loss.item()
            step += 1

            # Save model
            if step % config.save_steps == 0:
                save_path = os.path.join(config.output_dir, f"step-{step}")
                unet.save_pretrained(save_path)
                print(f"\nSaved model checkpoint at step {step}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

    # Save the final model
    final_save_path = os.path.join(config.output_dir, "final_model")
    unet.save_pretrained(final_save_path)
    print(f"\nTraining complete. Final model saved at {final_save_path}.")


if __name__ == "__main__":
    main()
