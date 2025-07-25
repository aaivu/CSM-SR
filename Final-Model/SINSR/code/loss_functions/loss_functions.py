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
    y_true = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
    y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

    layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

    feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in layers])
    
    # weights = [2.0, 2.0, 1.0, 1.0, 1.5]  # Emphasize higher layers

    vgg_features_true = feature_extractor(y_true)
    vgg_features_pred = feature_extractor(y_pred)
    
    perc_loss = tf.reduce_mean([tf.reduce_mean(tf.square(f_true - f_pred)) for f_true, f_pred in zip(vgg_features_true, vgg_features_pred)]) ## added weight for this..
    # perc_loss = tf.add_n([w * tf.reduce_mean(tf.square(f_true - f_pred))
    #                       for w, f_true, f_pred in zip(weights, vgg_features_true, vgg_features_pred)])
    return perc_loss

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

def structure_aware_loss(y_true, y_pred, lambda_tv=0.25, lambda_sm=1.0):
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
    # return lambda_sm * structure_similarity_loss(y_true, y_pred) #+  lambda_tv * total_variation_loss(y_pred)
    return lambda_sm * structure_similarity_loss(y_true, y_pred)
    
def color_consistency_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def texture_matching_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))

def contextual_loss(y_true, y_pred, h=0.5):
    def cosine_similarity(x1, x2):
        x1 = tf.nn.l2_normalize(x1, axis=-1)
        x2 = tf.nn.l2_normalize(x2, axis=-1)
        return tf.reduce_sum(x1 * x2, axis=-1)
    
    def contextual_similarity(y_true_patches, y_pred_patches):
        cs = cosine_similarity(y_true_patches, y_pred_patches)
        return cs / (tf.reduce_sum(cs, axis=-1, keepdims=True) + 1e-5)
    
    y_true_patches = tf.image.extract_patches(y_true, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    y_pred_patches = tf.image.extract_patches(y_pred, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    
    y_true_patches = tf.reshape(y_true_patches, (tf.shape(y_true)[0], -1, 3, 3, tf.shape(y_true)[-1]))
    y_pred_patches = tf.reshape(y_pred_patches, (tf.shape(y_pred)[0], -1, 3, 3, tf.shape(y_pred)[-1]))
    
    cs = contextual_similarity(y_true_patches, y_pred_patches)
    cs = tf.clip_by_value(cs, 1e-10, 1.0)  # Clip values to avoid log(0)
    # cont_loss = -tf.reduce_sum(tf.math.log(cs))
    cont_loss = -tf.reduce_sum(tf.math.log(cs)) / tf.cast(tf.size(cs), tf.float32) # Normalize loss return cont_loss
    return cont_loss

def gram_matrix(x):
    channels = int(x.shape[-1])
    a = tf.reshape(x, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def style_checking_loss(y_true, y_pred):
    gram_true = gram_matrix(y_true)
    gram_pred = gram_matrix(y_pred)
    return tf.reduce_mean(tf.square(gram_true - gram_pred))

def auxiliary_loss(y_true, y_pred, lambda_color=0.5, lambda_texture=0.5, lambda_cont=0.25):
    color_loss = color_consistency_loss(y_true, y_pred)
    texture_loss = texture_matching_loss(y_true, y_pred)
    # style_loss = style_checking_loss(y_true, y_pred)
    cont_loss = contextual_loss(y_true, y_pred)
    # cont_loss = quantum_loss(y_true, y_pred)
    return lambda_color * color_loss + lambda_texture * texture_loss + lambda_cont * cont_loss #+ lambda_cont * style_loss

# Define the global variable for the feature extractor
feature_extractor = None

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        vgg = VGG19(include_top=False, weights='imagenet')
        layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in layers])
    return feature_extractor

def feature_matching_loss(vgg, y_true, y_pred):

    # Preprocess the real and fake outputs
    y_true = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
    y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

    # Get the feature extractor
    feature_extractor = get_feature_extractor()

    real_features = feature_extractor(y_true)
    fake_features = feature_extractor(y_pred)
    loss = 0
    for real, fake in zip(real_features, fake_features):
        loss += tf.reduce_mean(tf.square(real - fake))
    return loss



def total_loss(vgg, y_true, y_pred, discriminator_output_real, discriminator_output_fake, 
               lambda_adv=1.0, 
               lambda_perceptual=1.0, 
               lambda_gradient=1.0, 
               lambda_second=1.0, 
               lambda_struct=1.0, 
               lambda_aux=1.0
               ):
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
    aux_loss = auxiliary_loss(y_true, y_pred)
    
    # fm_loss = feature_matching_loss(vgg, y_true, y_pred)

    total_loss_value = (lambda_adv * adv_loss) + (lambda_perceptual * perc_loss)  + (lambda_gradient * grad_loss)+ \
                       (lambda_second * second_grad_loss) + (lambda_struct * struct_loss) + (lambda_aux * aux_loss) #+ (fm_loss)
    
    individual_losses = { 
        'adv': adv_loss, 
        'perc': perc_loss, 
        'grad': grad_loss, 
        'second_grad': second_grad_loss, 
        'struct': struct_loss, 
        'aux': aux_loss } 
    
    return total_loss_value, individual_losses
