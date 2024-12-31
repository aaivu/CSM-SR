import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import cv2

# PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(hr_image, sr_image):
    """
    Calculate PSNR between two images.
    
    Args:
        hr_image (numpy.ndarray): High-resolution image.
        sr_image (numpy.ndarray): Super-resolution image.
    
    Returns:
        float: PSNR value.
    """
    mse = np.mean((hr_image - sr_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# SSIM (Structural Similarity Index)
def calculate_ssim(hr_image, sr_image):
    """
    Calculate SSIM between two images.
    
    Args:
        hr_image (numpy.ndarray): High-resolution image.
        sr_image (numpy.ndarray): Super-resolution image.
    
    Returns:
        float: SSIM value.
    """
    hr_image_gray = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
    sr_image_gray = cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(hr_image_gray, sr_image_gray, full=True)
    return ssim_value


# LPIPS (Learned Perceptual Image Patch Similarity)
def calculate_lpips(hr_image, sr_image):
    """
    Calculate LPIPS between two images.
    
    Args:
        hr_image (numpy.ndarray): High-resolution image.
        sr_image (numpy.ndarray): Super-resolution image.
    
    Returns:
        float: LPIPS value.
    """
    loss_fn = lpips.LPIPS(net='alex')
    
    # Ensure images are in RGB format
    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    
    # Convert images to tensors
    hr_image = torch.tensor(hr_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    sr_image = torch.tensor(sr_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Calculate LPIPS
    lpips_value = loss_fn(hr_image, sr_image)
    return lpips_value.item()


