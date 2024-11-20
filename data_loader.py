import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CelebADataset(Dataset):
    """
    Custom Dataset class for CelebA dataset.
    """
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            transform (callable, optional): Transformation to apply to the images.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        """
        Returns the total number of images.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Transformed image.
        """
        img_path = self.img_paths[idx]
        try:
            # Load and convert the image
            image = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
            if self.transform:
                image = self.transform(image)
            return image
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping corrupted or unreadable file: {img_path}. Error: {e}")
            # Return the next valid image in the dataset
            return self.__getitem__((idx + 1) % len(self))


def load_data(img_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Load the CelebA dataset and create a DataLoader.

    Args:
        img_dir (str): Path to the image directory.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: DataLoader for the CelebA dataset.
    """
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 64)),   # Resize images to 64x64 pixels
        transforms.ToTensor(),        # Convert images to PyTorch tensors
        transforms.Normalize([0.5], [0.5])  # Normalize pixel values to [-1, 1]
    ])

    # Create the dataset
    dataset = CelebADataset(img_dir=img_dir, transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
