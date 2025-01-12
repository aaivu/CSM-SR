import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from layer import ResizeLayer

# Generator Model 1
def generator(input_shape=(192, 256, 3)):
    """
    Builds the super-resolution generator model with multi-scale processing.

    Args:
    - input_shape (tuple): Shape of the input image.

    Returns:
    - model (Model): Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Multi-scale processing without downscaling
    scales = []
    for scale in range(3):
        scale_output = layers.Conv2D(64 * (2 ** scale), (3, 3), padding='same')(x)
        scale_output = layers.Activation('relu')(scale_output)
        scales.append(scale_output)
    
    # Concatenate multi-scale features
    x = layers.Concatenate()(scales)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)

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
