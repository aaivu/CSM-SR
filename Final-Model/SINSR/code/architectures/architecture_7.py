import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# class AdaptiveAveragePooling2D(layers.Layer):
#     def __init__(self, pool_size=(2, 2), **kwargs):
#         super(AdaptiveAveragePooling2D, self).__init__(**kwargs)
#         self.pool_size = pool_size

#     def call(self, inputs):
#         input_shape = tf.shape(inputs)
#         target_height = input_shape[1] // self.pool_size[0]
#         target_width = input_shape[2] // self.pool_size[1]
#         return tf.image.resize(inputs, [target_height, target_width])

#     def get_config(self):
#         config = super().get_config()
#         config.update({"pool_size": self.pool_size})
#         return config


class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

# Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
        x = layers.Activation('relu')(x)
        
        # Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)

        # Depthwise separable convolution
        x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=4, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def generator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Original scale
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Original scale processing (scale1)
#     scale1 = x
#     for _ in range(3):
#         scale1 = rrdb(scale1, 128)

#     # Downscale by 2 (scale2) dynamically
#     scale2 = tf.image.resize(scale1, [tf.shape(scale1)[1] // 2, tf.shape(scale1)[2] // 2])
#     for _ in range(3):
#         scale2 = rrdb(scale2, 128)
#     # Upscale by 2 dynamically
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     # Upscale by 2 (scale4) dynamically
#     scale4 = tf.image.resize(scale1, [tf.shape(scale1)[1] * 2, tf.shape(scale1)[2] * 2])
#     for _ in range(3):
#         scale4 = rrdb(scale4, 128)
#     # Downscale by 2 dynamically
#     scale4 = tf.image.resize(scale4, [tf.shape(scale4)[1] // 2, tf.shape(scale4)[2] // 2])
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2 dynamically
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2 dynamically
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

# def discriminator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)

#     # Initial convolution block
#     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 32)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 32)
#     scale2 = PixelShuffle(scale=2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 32)
#     scale3 = PixelShuffle(scale=4)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)

# 1st one
# def generator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Original scale
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Original scale processing (scale1)
#     scale1 = x
#     for _ in range(3):
#         scale1 = rrdb(scale1, 128)

#     # Downscale by 2 (scale2)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(scale1)
#     for _ in range(3):
#         scale2 = rrdb(scale2, 128)
#     # Upscale by 2
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     # Upscale by 2 (scale4)
#     scale4 = PixelShuffle(scale=2)(scale1)
#     for _ in range(3):
#         scale4 = rrdb(scale4, 128)
#     # Downscale by 2
#     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)
# def discriminator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)

#     # Initial convolution block
#     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 32)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 32)
#     scale2 = PixelShuffle(scale=2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 32)
#     scale3 = PixelShuffle(scale=4)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)

# 2nd one..
def generator(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Original scale
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    # Original scale processing (scale1)
    scale1 = x
    for _ in range(2):
        scale1 = rrdb(scale1, 64)

    # Downscale by 2 (scale2)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(scale1)
    for _ in range(2):
        scale2 = rrdb(scale2, 64)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)
    # scale2 = layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(scale2)  # Ensure matching height
    
    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(scale1)
    for _ in range(2):
        scale4 = rrdb(scale4, 64)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)

    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
    return Model(inputs, outputs)


def discriminator(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 32)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 32)
    scale2 = PixelShuffle(scale=2)(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 32)
    scale3 = PixelShuffle(scale=4)(scale3)

    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)
