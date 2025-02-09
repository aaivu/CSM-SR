import tensorflow as tf


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 in [0,1]
    return image

def compute_psnr(y_true, y_pred):
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr_value

def compute_ssim(y_true, y_pred):
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return ssim_value

import torch
import lpips

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='vgg')

import torch
import lpips
import numpy as np

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='vgg')

def compute_lpips(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    # Ensure input shapes are [batch_size, height, width, channels]
    if len(y_true.shape) == 3:
        y_true = np.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 3:
        y_pred = np.expand_dims(y_pred, axis=0)
    
    # Convert images to tensors and normalize to [-1, 1]
    y_true_torch = torch.tensor(y_true * 2 - 1).permute(0, 3, 1, 2).float()
    y_pred_torch = torch.tensor(y_pred * 2 - 1).permute(0, 3, 1, 2).float()
    
    # Compute LPIPS score
    lpips_value = lpips_model(y_true_torch, y_pred_torch)
    return lpips_value.mean().item()


# Example usage
y_true = load_image("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/S1/train_HR/Urban100_img_079_SRF_4_HR.png")
# #y_pred = load_image("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experimental_results/train_gen/generated_images6_t3/epoch_69_step_91_replica_0.jpg")
# y_pred = load_image("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experimental_results/train_gen/generated_images1_t2/epoch_0_step_91_replica_0.jpg")
y_pred = load_image("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experimental_results/train_gen/generated_images7_t1_45d/epoch_10_step_139_replica_0.jpg")


# Example usage
lpips_value = compute_lpips(y_true, y_pred)
print("LPIPS: ", lpips_value)

# Example usage3
ssim_value = compute_ssim(y_true, y_pred)
print("SSIM: ", ssim_value.numpy())

# Example usage
psnr_value = compute_psnr(y_true, y_pred)
print("PSNR: ", psnr_value.numpy())
