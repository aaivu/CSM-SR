import os
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from architectures.layer import ResizeLayer

# Residual Block: A block for feature extraction with residual connections
def res_block(x, filters):
    """
    Constructs a residual block with convolutional layers and activation.

    Args:
    - x (Tensor): Input tensor.
    - filters (int): Number of filters for the convolutional layers.

    Returns:
    - Tensor: Output tensor after applying the residual block.
    """
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

# Attention Block: A block for focusing on relevant features
def attention_block(x, filters):
    """
    Constructs an attention block to focus on relevant features.

    Args:
    - x (Tensor): Input tensor.
    - filters (int): Number of filters for the convolutional layers.

    Returns:
    - Tensor: Output tensor after applying the attention mechanism.
    """
    f = layers.Conv2D(filters, (1, 1), padding='same')(x)
    g = layers.Conv2D(filters, (1, 1), padding='same')(x)
    h = layers.Conv2D(filters, (1, 1), padding='same')(x)
    attn = layers.Add()([f, g])
    attn = layers.Activation('relu')(attn)
    attn = layers.Conv2D(1, (1, 1), padding='same')(attn)
    attn = layers.Activation('sigmoid')(attn)
    out = layers.Multiply()([h, attn])
    out = layers.Add()([x, out])
    return out

# Generator Model
def generator(input_shape=(192, 256, 3)):
    """
    Builds the super-resolution generator model with multi-scale processing and residual blocks.

    Args:
    - input_shape (tuple): Shape of the input image.

    Returns:
    - model (Model): Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Multi-scale processing with residual blocks
    for _ in range(5):
        x = res_block(x, 64)
    
    # Attention mechanism
    x = attention_block(x, 64)
    
    # Upsampling layers to produce higher resolution
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)

    # Final output layer
    outputs = layers.Conv2D(3, (3, 3), padding='same')(x)

    # Create and return the model
    return Model(inputs, outputs)

# Discriminator Model
def discriminator(input_shape=(768, 1024, 3)):
    """
    Builds the hybrid discriminator model using a combination of feature pyramid network (FPN) and PatchGAN.

    Args:
    - input_shape (tuple): Shape of the input image.

    Returns:
    - model (Model): Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Feature pyramid levels
    pyramid_levels = []
    for scale in range(3):
        if scale > 0:
            x = layers.MaxPooling2D()(x)
        level_output = layers.Conv2D(64 * (2 ** scale), (3, 3), padding='same')(x)
        level_output = layers.Activation('relu')(level_output)
        # Resize to the original input shape
        level_output = ResizeLayer(input_shape[:2])(level_output)
        pyramid_levels.append(level_output)
    
    # Concatenate pyramid features
    x = layers.Concatenate()(pyramid_levels)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)

    # PatchGAN-style convolutions
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    # Final convolution layer
    x = layers.Conv2D(1, (4, 4), padding='same')(x)

    # Create and return the model
    return Model(inputs, x)
