# data_loader.py

import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
import torch


class CelebADataset(Dataset):
    """
    Custom Dataset class for CelebA dataset.
    """
    def __init__(self, img_dir, tokenizer, transform=None, prompt="A high-resolution photo of a face"):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the text prompts.
            transform (callable, optional): Transformation to apply to the images.
            prompt (str): Text prompt associated with each image.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.img_paths = [
            os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not self.img_paths:
            raise ValueError(f"No images found in {self.img_dir}. Please check the directory path and image formats.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding tokenized prompt.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: A dictionary containing:
                - "pixel_values" (torch.Tensor): Transformed image tensor.
                - "input_ids" (torch.Tensor): Tokenized prompt tensor.
        """
        max_retries = 10
        original_idx = idx  # Keep track of the original index to prevent infinite loops

        for attempt in range(max_retries):
            img_path = self.img_paths[idx]
            try:
                # Load and convert the image to RGB
                image = Image.open(img_path).convert("RGB")

                # Apply transformations if any
                if self.transform:
                    image = self.transform(image)

                # Tokenize the prompt
                text_inputs = self.tokenizer(
                    self.prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_inputs.input_ids.squeeze()  # Shape: [tokenizer_max_length]

                return {
                    "pixel_values": image,   # Shape: [3, 512, 512]
                    "input_ids": input_ids   # Shape: [tokenizer_max_length]
                }

            except (UnidentifiedImageError, OSError) as e:
                print(f"Warning: Skipping corrupted or unreadable file: {img_path}. Error: {e}")
                idx = (idx + 1) % len(self.img_paths)
                if idx == original_idx:
                    # All attempts failed; raise an error
                    raise RuntimeError(f"All attempts to load images failed. Please check the dataset.")

        # If all retries fail, raise an error
        raise RuntimeError(f"Failed to load a valid image after {max_retries} retries.")


def load_data(img_dir, tokenizer, batch_size=16, shuffle=True, num_workers=4, max_images=None):
    """
    Load the CelebA dataset and create a DataLoader.

    Args:
        img_dir (str): Path to the image directory.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the text prompts.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        max_images (int, optional): Maximum number of images to load. If None, load all available images.

    Returns:
        DataLoader: DataLoader for the CelebA dataset.
    """
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Increased resolution to match Stable Diffusion requirements
        transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with 50% probability
        transforms.RandomRotation(5),    # Rotate images by up to 5 degrees
        transforms.ColorJitter(
            brightness=0.1, 
            contrast=0.1, 
            saturation=0.1
        ),  # Slightly adjust brightness, contrast, and saturation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],  # Normalize each channel to [-1, 1]
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Create the full dataset
    full_dataset = CelebADataset(
        img_dir=img_dir,
        tokenizer=tokenizer,
        transform=transform,
        prompt="A high-resolution photo of a face"  # Generic prompt for all images
    )

    # If max_images is set, create a subset of the dataset
    if max_images is not None:
        if max_images > len(full_dataset):
            print(f"Requested max_images={max_images} exceeds dataset size={len(full_dataset)}. Using the full dataset.")
            max_images = len(full_dataset)
        indices = list(range(max_images))
        subset_dataset = Subset(full_dataset, indices)
        print(f"Loaded a subset of {max_images} images.")
    else:
        subset_dataset = full_dataset
        print(f"Loaded the full dataset with {len(subset_dataset)} images.")

    # Create the DataLoader
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True  # Ensures all batches have the same size
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Define the path to the CelebA dataset
    celeba_image_dir = "./img_align_celeba"  # Update this path as needed

    # Load the DataLoader with a maximum of 10,000 images
    dataloader = load_data(
        img_dir=celeba_image_dir,
        tokenizer=tokenizer,
        batch_size=16,    # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,    # Adjust based on your CPU cores
        max_images=None  # Limit to 10,000 images
    )

    # Iterate through one batch to verify
    for batch in dataloader:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]

        print(f"Pixel Values Shape: {pixel_values.shape}")  # Expected: [batch_size, 3, 512, 512]
        print(f"Input IDs Shape: {input_ids.shape}")        # Expected: [batch_size, tokenizer_max_length]
        break  # Remove this break to iterate through the entire dataset
