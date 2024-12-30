import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# Residual Block
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    return x

# Generator Model
def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (9, 9), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # B residual blocks
    for _ in range(16):
        x = res_block(x, 64)

    # Post-residual block
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, inputs])

    # Upsampling blocks
    for _ in range(2):
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        x = tf.nn.depth_to_space(x, 2)

    # Output layer
    outputs = layers.Conv2D(3, (9, 9), padding='same', activation='tanh')(x)
    return Model(inputs, outputs)

def discriminator(input_shape=(768, 1024, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Convolutional blocks
    filters = [64, 128, 128, 256, 256, 512, 512]
    strides = [2, 1, 2, 1, 2, 1, 2]
    for f, s in zip(filters, strides):
        x = layers.Conv2D(f, (3, 3), strides=(s, s), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)

