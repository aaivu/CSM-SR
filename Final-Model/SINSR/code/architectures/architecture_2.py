import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from layer import ResizeLayer


# Residual Block: A block for feature extraction with residual connections
def res_block(x, filters):
    """
    Constructs a residual block with convolutional layers, batch normalization, and activation.

    Args:
    - x (Tensor): Input tensor.
    - filters (int): Number of filters for the convolutional layers.

    Returns:
    - Tensor: Output tensor after applying residual block.
    """
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
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
    - Tensor: Output tensor after applying attention mechanism.
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
    Builds the super-resolution generator model with residual and attention blocks.

    Args:
    - input_shape (tuple): Shape of the input image.

    Returns:
    - model (Model): Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Multi-scale processing with residual blocks and attention
    for _ in range(10):  # Increased number of residual blocks for added complexity
        x = res_block(x, 64)
    
    x = attention_block(x, 64)
    
    # Upsampling layers to produce higher resolution
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)  # Increased number of filters
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Final output layer
    outputs = layers.Conv2D(3, (3, 3), padding='same')(x)
    
    # Create and return the model
    return Model(inputs, outputs)

# Discriminator Model
def discriminator(input_shape=(768, 1024, 3)):
    """
    Builds the discriminator model using a feature pyramid network (FPN).

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

    # Final output layer
    outputs = layers.Conv2D(1, (3, 3), padding='same')(x)

    # Create and return the model
    return Model(inputs, outputs)
