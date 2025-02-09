
################## -- Test -- ############################
## VERSION 1.0 ##
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore



# Custom DynamicUpsampling Layer
class DynamicUpsampling(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(DynamicUpsampling, self).__init__(**kwargs)
        self.scale = scale
        self.filters = tf.Variable(initial_value=tf.random.normal([3, 3, self.scale, self.scale]), trainable=True)

    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)
        upscaled = tf.nn.conv2d_transpose(inputs, self.filters, output_shape=[batch_size, height * self.scale, width * self.scale, channels], strides=[1, self.scale, self.scale, 1], padding='SAME')
        return upscaled

class PixelAdaptiveConvolution(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(PixelAdaptiveConvolution, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dynamic_filter_gen = layers.Conv2D(filters * kernel_size[0] * kernel_size[1] * filters, (1, 1))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        
        dynamic_filters = self.dynamic_filter_gen(inputs)
        dynamic_filters = tf.reshape(dynamic_filters, [batch_size, height, width, self.kernel_size[0], self.kernel_size[1], channels, self.filters])
        
        # Create patches of the input with the same size as the dynamic filters
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        patches = tf.reshape(patches, [batch_size, height, width, self.kernel_size[0], self.kernel_size[1], channels])
        
        # Perform element-wise multiplication between patches and dynamic filters
        outputs = tf.einsum('bhwklc,bhwklcf->bhwf', patches, dynamic_filters)
        
        return outputs

# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
        x = layers.Activation('relu')(x)
        
        # Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)
        
        # # PixelAdaptiveConvolution
        # x = PixelAdaptiveConvolution(growth_rate, (3, 3))(x)
        # x = layers.Activation('relu')(x)

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

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Original scale
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    # Original scale processing (scale1)
    scale1 = x
    for _ in range(3):
        scale1 = rrdb(scale1, 128)

    # Downscale by 2 (scale2)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(3):
        scale2 = rrdb(scale2, 128)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)
    # scale2 = DynamicUpsampling(scale=2)(scale2)

    # # Downscale by 4 (scale3)
    # scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    # for _ in range(2):
    #     scale3 = rrdb(scale3, 128)
    # # Upscale by 4
    # scale3 = PixelShuffle(scale=4)(scale3)
    
    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(x)
    for _ in range(3):
        scale4 = rrdb(scale4, 128)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
    # # Upscale by 4 and Downscale by 4 (scale5)
    # scale5 = PixelShuffle(scale=4)(x)
    # for _ in range(2):
    #     scale5 = rrdb(scale5, 128)
    # scale5 = layers.AveragePooling2D(pool_size=(4, 4))(scale5)
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
    return Model(inputs, outputs)


# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

def discriminator(input_shape=(768, 1024, 3)):
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
