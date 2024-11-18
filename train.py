# train.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_processing import load_data
from diffusers import StableDiffusionPipeline, DDPMScheduler
import os

# Configuration
img_dir = "./img_align_celeba"
output_dir = "./saved_models"
batch_size = 32  # Adjust based on your GPU memory
num_epochs = 10  # Start with 1 for testing
learning_rate = 1e-5
max_train_steps = None  # Set to None to go through all epochs
save_every = 1000  # Save checkpoint every N steps

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Selected device: ", device)
# Load data
dataloader = load_data(img_dir=img_dir, batch_size=batch_size)

# Load pre-trained Stable Diffusion model
access_token = "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" # for hugging face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    token=access_token
).to(device)

# Freeze VAE and text encoder to save memory
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# Use a DDPM scheduler for training
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# Optimizer
optimizer = AdamW(pipe.unet.parameters(), lr=learning_rate)

# Training loop
global_step = 0
for epoch in range(num_epochs):
    pipe.unet.train()
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        latents = pipe.vae.encode(batch.to(device, dtype=torch.float16)).latent_dist.sample()
        latents = latents * 0.18215  # Scaling as per SD implementation

        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()

        # Add noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings for conditioning (we can use a generic prompt)
        prompt = ["A high-resolution photo of a face"] * bsz
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

        # Predict the noise residual
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Compute the loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # Save checkpoint
        if global_step % save_every == 0:
            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
            pipe.save_pretrained(save_path)
            print(f"Saved checkpoint at step {global_step}")

        # Break if max steps reached
        if max_train_steps and global_step >= max_train_steps:
            break

    # Save model at the end of each epoch
    epoch_save_path = os.path.join(output_dir, f"epoch-{epoch+1}")
    pipe.save_pretrained(epoch_save_path)
    print(f"Saved model after epoch {epoch+1}")

print("Training complete.")