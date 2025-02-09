import cv2
import numpy as np

def custom_downsample(image, scale_factor, blur_kernel_size=(5, 5), sigma=1.0):
    """
    Custom downsampling algorithm with Gaussian blur.

    Parameters:
    - image: Input image (numpy array).
    - scale_factor: Factor by which to downsample the image.
    - blur_kernel_size: Size of the Gaussian blur kernel.
    - sigma: Standard deviation for Gaussian blur.

    Returns:
    - downsampled_image: Downsampled image (numpy array).
    """
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, blur_kernel_size, sigma)

    # Calculate the new dimensions
    new_width = int(image.shape[1] / scale_factor)
    new_height = int(image.shape[0] / scale_factor)

    # Resize the image using INTER_AREA interpolation
    downsampled_image = cv2.resize(blurred_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return downsampled_image

# Example usage
if __name__ == "__main__":
    # Load an example image
    image = cv2.imread('example_image.jpg')

    # Downsample the image by a factor of 2
    downsampled_image = custom_downsample(image, scale_factor=2)

    # Save the downsampled image
    cv2.imwrite('downsampled_image.jpg', downsampled_image)

    # Display the original and downsampled images
    cv2.imshow('Original Image', image)
    cv2.imshow('Downsampled Image', downsampled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
