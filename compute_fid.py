import os
import random
import shutil
from cleanfid import fid

# Directories
real_dir = "./img_align_celeba"
gen_dir = "./generated_images"
real_subset_dir = "./real_subset"
gen_subset_dir = "./gen_subset"

# Number of images to sample
num_images = 2900

# Ensure subset directories exist
os.makedirs(real_subset_dir, exist_ok=True)
os.makedirs(gen_subset_dir, exist_ok=True)

# List image files
real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
gen_images = [f for f in os.listdir(gen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle and sample
random.shuffle(real_images)
random.shuffle(gen_images)

# Determine number of images
num_images = min(num_images, len(real_images), len(gen_images))

# Copy subset of images to subset directories
#for img in real_images[:num_images]:
#    shutil.copy(os.path.join(real_dir, img), os.path.join(real_subset_dir, img))

#for img in gen_images[:num_images]:
#    shutil.copy(os.path.join(gen_dir, img), os.path.join(gen_subset_dir, img))

# Compute FID using subset directories
score = fid.compute_fid(real_subset_dir, gen_subset_dir, num_workers=24, batch_size=10)
print("FID score:", score)

# Optional: Clean up subset directories after computation
# Uncomment these lines if you want to remove the subset directories after use
# shutil.rmtree(real_subset_dir)
# shutil.rmtree(gen_subset_dir)