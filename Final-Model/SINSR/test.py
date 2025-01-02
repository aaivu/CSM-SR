import os
import numpy as np
import cv2

# Paths to the data directories
train_hr_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_HR'
train_lr_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4'
valid_hr_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_valid_HR'
valid_lr_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_valid_LR_bicubic/X4'

# Function to load images and check shapes
def check_image_shapes(image_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image {image_path}")
        print(f"{os.path.basename(image_path)}: {image.shape}")

# Check shapes of high-resolution training images
print("High-resolution training images:")
check_image_shapes(train_hr_path)

# Check shapes of low-resolution training images
print("Low-resolution training images:")
check_image_shapes(train_lr_path)

# Check shapes of high-resolution validation images
print("High-resolution validation images:")
check_image_shapes(valid_hr_path)

# Check shapes of low-resolution validation images
print("Low-resolution validation images:")
check_image_shapes(valid_lr_path)

# These are the actual HR image shapes :: (768, 1024, 3) 
# These are the actual LR image shapes :: (192, 256, 3)