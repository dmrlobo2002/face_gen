# Face Generation with Fine-Tuned Stable Diffusion

This repository demonstrates how to fine-tune a Stable Diffusion model on the CelebA dataset and then generate new face images from a textual prompt. You can find the training dataset in OneDrive.

## Overview

- **Training Script (`train.py`)**:

  - Fine-tunes the UNet component of a pretrained Stable Diffusion model on the CelebA dataset.
  - The VAE and text encoder remain frozen, while only the UNets parameters are updated.

- **Generation Script (`generate.py`)**:
  - Uses the fine-tuned UNet along with the original VAE and text encoder to generate new face images conditioned on a prompt.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch (compatible with your GPU/CUDA setup)
- Transformers
- Diffusers
- TQDM
- Pillow

### Installation Example:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu11x
pip install diffusers transformers datasets tqdm pillow
```

_(Adjust the PyTorch installation command as needed.)_

## Data Preparation

1. **Download the CelebA dataset** and place all images into `./img_align_celeba/`.

2. **Data Loader**:
   - Ensure that `data_loader.py` is in the same directory as `train.py`.
   - This script handles dataset loading and tokenization.
   - Modify any paths or parameters in `data_loader.py` as needed.

## Training

To fine-tune the model:

```bash
python train.py
```

### Key Configurations in `train.py`:

- `Config.data_dir`: Directory with CelebA images.
- `Config.output_dir`: Directory to save fine-tuned models.
- `Config.epochs`, `Config.batch_size`, `Config.learning_rate`: Training hyperparameters.
- `Config.save_steps`: Frequency of checkpoint saving.
- `Config.model_name`: Base Stable Diffusion model (e.g., `CompVis/stable-diffusion-v1-4`).
- `Config.prompt`: Text prompt used during training.
- `Config.device`: GPU device (e.g., `cuda:0`).

After training, the final model is saved in `./fine_tuned_model_all_images/final_model`.

## Generation

After fine-tuning, generate images with:

```bash
python generate.py
```

### Key Configurations in `generate.py`:

- `model_path`: Path to the fine-tuned model (e.g., `./fine_tuned_model_all_images/final_model`).
- `prompt`: Prompt to generate images from.
- `num_images`: Number of images to generate.
- `output_dir`: Directory for saving generated images.
- `device`: GPU device (e.g., `cuda:0`).
- `num_inference_steps`, `guidance_scale`: Parameters influencing output quality and adherence to the prompt.

Generated images will be saved in the specified `output_dir`.

## Future Improvements

- Experiment with different prompts and training hyperparameters.
- Use data augmentation or face alignment during training.
- Explore prompt engineering for better image framing and consistency.
