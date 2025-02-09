import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG19 # type: ignore


# Define loss functions
def adversarial_loss(real_output, fake_output):
    """
    Computes the adversarial loss using binary cross-entropy.

    Args:
    - real_output (Tensor): Ground truth labels.
    - fake_output (Tensor): Predicted labels.

    Returns:
    - Tensor: Adversarial loss.
    """
    adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))) + \
               tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output)))
    return adv_loss

def perceptual_loss(vgg, y_true, y_pred):
    """
    Computes the perceptual loss using features extracted from a pre-trained VGG19 network.

    Args:
    - vgg (Model): Pre-trained VGG19 model for feature extraction.
    - y_true (Tensor): Ground truth images.
    - y_pred (Tensor): Predicted images.

    Returns:
    - Tensor: Perceptual loss.
    """
    y_true = tf.image.resize(y_true, (1024, 768))  # Ensure y_true is resized
    y_pred = tf.image.resize(y_pred, (1024, 768))  # Ensure y_pred is resized
    vgg_features_true = vgg(y_true)
    vgg_features_pred = vgg(y_pred)
    return tf.reduce_mean(tf.square(vgg_features_true - vgg_features_pred))

def gradient_loss(y_true, y_pred):
    """
    Computes the gradient loss using Sobel edges.

    Args:
    - y_true (Tensor): Ground truth images.
    - y_pred (Tensor): Predicted images.

    Returns:
    - Tensor: Gradient loss.
    """
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(grad_true - grad_pred))

def second_order_gradient_loss(y_true, y_pred):
    """
    Computes the second-order gradient loss using Sobel edges.

    Args:
    - y_true (Tensor): Ground truth images.
    - y_pred (Tensor): Predicted images.

    Returns:
    - Tensor: Second-order gradient loss.
    """
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    grad2_true_x = tf.image.sobel_edges(grad_true[:, :, :, 0, :])
    grad2_true_y = tf.image.sobel_edges(grad_true[:, :, :, 1, :])
    grad2_pred_x = tf.image.sobel_edges(grad_pred[:, :, :, 0, :])
    grad2_pred_y = tf.image.sobel_edges(grad_pred[:, :, :, 1, :])
    grad2_true = tf.concat([grad2_true_x, grad2_true_y], axis=-1)
    grad2_pred = tf.concat([grad2_pred_x, grad2_pred_y], axis=-1)
    return tf.reduce_mean(tf.square(grad2_true - grad2_pred))

def total_variation_loss(y_pred):
    """
    Computes the total variation loss to encourage spatial smoothness.

    Args:
    - y_pred (Tensor): Predicted images.

    Returns:
    - Tensor: Total variation loss.
    """
    return tf.reduce_mean(tf.image.total_variation(y_pred))

def structure_similarity_loss(y_true, y_pred):
    """
    Computes the structural similarity loss between ground truth and predicted images.

    Args:
    - y_true (Tensor): Ground truth images.
    - y_pred (Tensor): Predicted images.

    Returns:
    - Tensor: Structural similarity loss.
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def structure_aware_loss(y_true, y_pred, lambda_tv=0.5, lambda_sm=0.25):
    """
    Combines total variation and structural similarity losses.

    Args:
    - y_true (Tensor): Ground truth images.
    - y_pred (Tensor): Predicted images.
    - lambda_tv (float): Weight for total variation loss.
    - lambda_sm (float): Weight for structural similarity loss.

    Returns:
    - Tensor: Combined structure-aware loss.
    """
    return lambda_tv * total_variation_loss(y_pred) + lambda_sm * structure_similarity_loss(y_true, y_pred)

def total_loss(vgg, y_true, y_pred, discriminator_output_real, discriminator_output_fake, 
               lambda_adv=1.0, lambda_perceptual=1.0, lambda_grad=1.0, 
               lambda_second=1.0, lambda_struct=1.0):
    """
    Computes the total loss as a weighted sum of different loss components.

    Args:
    - vgg (Model): Pre-trained VGG19 model for feature extraction.
    - y_true (Tensor): Ground truth images.
    - y_pred (Tensor): Predicted images.
    - discriminator_output_real (Tensor): Discriminator output for real images.
    - discriminator_output_fake (Tensor): Discriminator output for fake images.
    - lambda_adv (float): Weight for adversarial loss.
    - lambda_perceptual (float): Weight for perceptual loss.
    - lambda_grad (float): Weight for gradient loss.
    - lambda_second (float): Weight for second-order gradient loss.
    - lambda_struct (float): Weight for structure-aware loss.

    Returns:
    - Tensor: Total loss value.
    """
    adv_loss = adversarial_loss(discriminator_output_real, tf.ones_like(discriminator_output_real, dtype=tf.float32)) + \
               adversarial_loss(discriminator_output_fake, tf.zeros_like(discriminator_output_fake, dtype=tf.float32))
    perc_loss = perceptual_loss(vgg, y_true, y_pred)
    grad_loss = gradient_loss(y_true, y_pred)
    second_grad_loss = second_order_gradient_loss(y_true, y_pred)
    struct_loss = structure_aware_loss(y_true, y_pred)

    total_loss_value = (lambda_adv * adv_loss) + (lambda_perceptual * perc_loss) + (lambda_grad * grad_loss) + \
                       (lambda_second * second_grad_loss) + (lambda_struct * struct_loss)
    
    return total_loss_value
