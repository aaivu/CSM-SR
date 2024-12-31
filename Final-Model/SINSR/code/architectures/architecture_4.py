import tensorflow as tf
from tensorflow.keras import layers, Model  # type: ignore

# Residual Block
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

# Attention Block
def attention_block(x, filters):
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

# Multi-Scale Generator Model
def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 64)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 64)
    scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 64)
    scale3 = layers.UpSampling2D(size=(4, 4))(scale3)

    # Combine multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Reduce the number of channels after concatenation
    multi_scale = layers.Conv2D(64, (1, 1), padding='same')(multi_scale)
    
    # Additional residual blocks
    for _ in range(5):
        multi_scale = res_block(multi_scale, 64)
    
    # Attention mechanism
    x = attention_block(multi_scale, 64)
    
    # Upsampling layers to produce higher resolution
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)

    # Final output layer
    outputs = layers.Conv2D(3, (3, 3), padding='same')(x)
    return Model(inputs, outputs)


# Custom Layer for Resizing
class ResizeLayer(layers.Layer):
    def __init__(self, size):
        super(ResizeLayer, self).__init__()
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, self.size)

# Residual Block
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

# Enhanced Discriminator Model
def discriminator(input_shape=(768, 1024, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 64)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 64)
    scale2 = ResizeLayer(input_shape[:2])(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 64)
    scale3 = ResizeLayer(input_shape[:2])(scale3)

    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
    x = layers.Activation('relu')(x)

    # PatchGAN-style convolutions
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [128, 256, 512, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    # Final convolution layer
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)


