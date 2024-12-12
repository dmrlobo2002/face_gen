import pytorch_fid.fid_score as fid_score
import os
import torch
from PIL import Image

def is_valid_image(image_path):
  """
  Checks if the given image file is valid.

  Args:
    image_path: Path to the image file.

  Returns:
    True if the image is valid, False otherwise.
  """
  try:
    Image.open(image_path).verify()  # Verify the image file without loading it completely
    return True
  except Exception as e:
    print(f"Error loading image {image_path}: {e}")
    return False

def calculate_fid(real_dir, gen_dir, batch_size=100, device='cuda:2' if torch.cuda.is_available() else 'cpu', dims=2048):
  """
  Calculates the FID score for a given pair of real and generated image directories.

  Args:
    real_dir: Path to the directory containing real images.
    gen_dir: Path to the directory containing generated images.
    batch_size: Batch size for image loading.
    device: Device to use for computation ('cuda' or 'cpu').
    dims: Dimensionality of the feature vectors.

  Returns:
    The FID score.
  """

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
      [real_dir, gen_dir],  # Pass lists of valid image paths
      batch_size=batch_size, 
      device=device,
      dims=dims
  )
  return fid_value

# Example usage:
real_dir = 'real_subset/'
gen_dir = 'gen_subset/'

try:
  fid_score_value = calculate_fid(real_dir, gen_dir)
  print("FID Score:", fid_score_value)
except ValueError as e:
  print(f"Error: {e}")