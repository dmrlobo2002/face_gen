import pytorch_fid.fid_score as fid_score
import os
import torch
from PIL import Image

# Image validation in case any images get corrupted during unzipping
def is_valid_image(image_path):

  try:
    Image.open(image_path).verify()  # Actual img verifciation step (ensure it is not corrupted, etc...)
    return True
  except Exception as e:
    print(f"Error loading image {image_path}: {e}")
    return False

def calculate_fid(real_dir, gen_dir, batch_size=100, device='cuda:2' if torch.cuda.is_available() else 'cpu', dims=2048):
#   Validate that any images are not corrupted
#   # Validate images in real_dir
#   real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#   valid_real_images = [img for img in real_images if is_valid_image(img)]
#   if len(valid_real_images) != 2900:
#     raise ValueError("No valid images found in real_dir.")

#   # Validate images in gen_dir
#   gen_images = [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#   valid_gen_images = [img for img in gen_images if is_valid_image(img)]
#   if len(valid_gen_images) != 2900:
#     raise ValueError("No valid images found in gen_dir.")

  fid_value = fid_score.calculate_fid_given_paths(
      [real_dir, gen_dir],  
      batch_size=batch_size, 
      device=device,
      dims=dims
  )
  return fid_value

# Specify dirs containing images you want to compute FID on
real_dir = 'real_subset/'
gen_dir = 'gen_subset/'

try:
  fid_score_value = calculate_fid(real_dir, gen_dir)
  print("FID Score:", fid_score_value)
except ValueError as e:
  print(f"Error: {e}")