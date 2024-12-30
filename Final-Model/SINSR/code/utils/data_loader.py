import os
import json
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import cv2

# Load configuration
config_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json'  # Update this path accordingly
with open(config_path, 'r') as f:
    config = json.load(f)

# Define the ImageDataset class for loading data directly from disk
class ImageDataset(Dataset):
    def __init__(self, image_dir, target_size):
        """
        Initializes the ImageDataset instance.

        Args:
        - image_dir (str): Directory containing the images.
        - target_size (tuple): Target size for the images.
        """
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.target_size = (target_size[1], target_size[0]) # Ensure the size is (width, height)

    def __len__(self):
        """
        Returns the number of images in the directory.

        Returns:
        - int: Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses an image from the directory.

        Args:
        - idx (int): Index of the image to retrieve.

        Returns:
        - np.ndarray: Retrieved and preprocessed image.
        """
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, self.target_size)  # Resize to target size
        image = image / 255.0  # Normalize to [0, 1]
        return image

# Update the paths and target shapes
train_hr_path = config['train_hr_path']
train_lr_path = config['train_lr_path']
valid_hr_path = config['valid_hr_path']
valid_lr_path = config['valid_lr_path']
target_size_hr = (config['target_size_hr'][1], config['target_size_hr'][0])                   # Ensure correct order (width, height)
target_size_lr = (config['target_size_lr'][1], config['target_size_lr'][0])                   # Ensure correct order (width, height)
batch_size = config['batch_size']

# Initialize the datasets
train_hr_dataset = ImageDataset(train_hr_path, target_size_hr)  # Ensure target_size matches preprocessing
train_lr_dataset = ImageDataset(train_lr_path, target_size_lr)  # Ensure target_size matches preprocessing
valid_hr_dataset = ImageDataset(valid_hr_path, target_size_hr)  # Ensure target_size matches preprocessing
valid_lr_dataset = ImageDataset(valid_lr_path, target_size_lr)  # Ensure target_size matches preprocessing

# Create DataLoaders
train_hr_dataloader = DataLoader(train_hr_dataset, batch_size=batch_size, shuffle=True)
train_lr_dataloader = DataLoader(train_lr_dataset, batch_size=batch_size, shuffle=True)
valid_hr_dataloader = DataLoader(valid_hr_dataset, batch_size=batch_size, shuffle=False)
valid_lr_dataloader = DataLoader(valid_lr_dataset, batch_size=batch_size, shuffle=False)

