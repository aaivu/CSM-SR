import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# Define Residual Dense Block (RDB)
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
    #     x = layers.Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(x)
    #     concat_features.append(x)
    #     x = layers.Concatenate()(concat_features)
    # return layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=4, res_block = 4):
    res = x
    for _ in range(res_block):  # 3 RDBs within the RRDB
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    # x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Multi-Scale Processing Branches
    # Scale 1: Original scale
    scale1 = x
    for _ in range(2):  # Add 23 RRDB blocks
        scale1 = rrdb(scale1, 64)

    # Scale 2: Half scale
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(2):  # Add 23 RRDB blocks
        scale2 = rrdb(scale2, 64)
    scale2 = layers.UpSampling2D(size=(2, 2))(scale2)  # Upsample back to original scale
    
    # Scale 3: Double scale
    scale3 = layers.UpSampling2D(size=(2, 2))(x)
    for _ in range(2):  # Add 23 RRDB blocks
        scale3 = rrdb(scale3, 64)
    scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)  # Downsample to original scale
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upsampling to the final resolution
    multi_scale = layers.UpSampling2D(size=(2, 2))(multi_scale)  # Upsample to 384 x 512
    multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    multi_scale = layers.UpSampling2D(size=(2, 2))(multi_scale)  # Upsample to 768 x 1024
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output, ensure shape (768, 1024, 3)
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
    return Model(inputs, outputs)


# Custom Layer for Resizing
class ResizeLayer(layers.Layer):
    def __init__(self, size):
        super(ResizeLayer, self).__init__()
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, self.size)

# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    return x

def discriminator(input_shape=(768, 1024, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    # x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 64)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 64)
    scale2 = ResizeLayer(input_shape[:2])(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 64)
    scale3 = ResizeLayer(input_shape[:2])(scale3)

    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # PatchGAN-style convolutions
    # for filters in [64, 128, 256, 512, 1024]:
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # x = SpectralNormalization(layers.Conv2D(1, (4, 4), padding='same'))(x)
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)
