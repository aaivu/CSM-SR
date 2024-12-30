# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# # Define Advanced Blocks
# def sr_res_block(x, filters):
#     """
#     Super-Resolution Residual Block
#     :param x: Input tensor
#     :param filters: Number of filters for the convolutional layers
#     :return: Output tensor after applying residual block
#     """
#     res = x
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Add()([x, res])
#     return x

# def attention_block(x, filters):
#     """
#     Attention Block
#     :param x: Input tensor
#     :param filters: Number of filters for the convolutional layers
#     :return: Output tensor after applying attention mechanism
#     """
#     f = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     g = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     h = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     attn = layers.Add()([f, g])
#     attn = layers.Activation('relu')(attn)
#     attn = layers.Conv2D(1, (1, 1), padding='same')(attn)
#     attn = layers.Activation('sigmoid')(attn)
#     out = layers.Multiply()([h, attn])
#     out = layers.Add()([x, out])
#     return out

# # Define the Refined Generator Model
# def generator(input_shape=(192, 256, 3)):
#     """
#     Builds a refined generator model with both SR and Gradient branches
#     :param input_shape: Shape of the input tensor
#     :return: Generator model
#     """
#     inputs = layers.Input(shape=input_shape)
    
#     # SR Branch
#     sr_x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
#     sr_x = layers.Activation('relu')(sr_x)
#     for _ in range(8):
#         sr_x = sr_res_block(sr_x, 64)
#     sr_x = layers.Conv2D(64, (3, 3), padding='same')(sr_x)
#     sr_x = layers.UpSampling2D(size=(2, 2))(sr_x)
#     sr_x = attention_block(sr_x, 64)
#     sr_x = layers.UpSampling2D(size=(2, 2))(sr_x)
#     sr_output = layers.Conv2D(3, (3, 3), padding='same')(sr_x)

#     # Gradient Branch
#     grad_x = tf.image.sobel_edges(inputs)
#     grad_x = layers.Conv2D(64, (3, 3), padding='same')(grad_x)
#     grad_x = layers.Activation('relu')(grad_x)
#     for _ in range(8):
#         grad_x = sr_res_block(grad_x, 64)
#     grad_x = layers.Conv2D(64, (3, 3), padding='same')(grad_x)
#     grad_x = layers.UpSampling2D(size=(2, 2))(grad_x)
#     grad_x = layers.Conv2D(1, (1, 1), padding='same')(grad_x)
    
#     # Fusion Block
#     combined_output = layers.Add()([sr_output, grad_x])

#     return Model(inputs, combined_output)

# # Custom Layer for Resizing
# class ResizeLayer(layers.Layer):
#     """
#     Custom Layer for resizing tensors to a given size
#     """
#     def __init__(self, size):
#         super(ResizeLayer, self).__init__()
#         self.size = size

#     def call(self, inputs):
#         return tf.image.resize(inputs, self.size)

# # Residual Block for Discriminator
# def res_block(x, filters):
#     """
#     Residual Block for the Discriminator
#     :param x: Input tensor
#     :param filters: Number of filters for the convolutional layers
#     :return: Output tensor after applying residual block
#     """
#     res = x
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Add()([x, res])
#     return x

# def discriminator(input_shape=(768, 1024, 3)):
#     """
#     Builds a refined discriminator model with multi-scale processing and PatchGAN-style convolutions
#     :param input_shape: Shape of the input tensor
#     :return: Discriminator model
#     """
#     inputs = layers.Input(shape=input_shape)

#     # Initial convolution block
#     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 64)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 64)
#     scale2 = ResizeLayer(input_shape[:2])(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     scale3 = ResizeLayer(input_shape[:2])(scale3)

#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)

#     # PatchGAN-style convolutions
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(alpha=0.2)(x)

#     # Final convolution layer
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)


# # Build the refined generator model
# refined_sr_and_grad_generator = generator()
# refined_sr_and_grad_generator.summary()

# # Build the refined discriminator model
# refined_discriminator = discriminator(input_shape=(768, 1024, 3))
# refined_discriminator.summary()

# import tensorflow as tf
# from tensorflow.keras import layers, Model

# # Define Advanced Blocks
# def sr_res_block(x, filters):
#     res = x
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Add()([x, res])
#     return x

# def attention_block(x, filters):
#     f = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     g = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     h = layers.Conv2D(filters, (1, 1), padding='same')(x)
#     attn = layers.Add()([f, g])
#     attn = layers.Activation('relu')(attn)
#     attn = layers.Conv2D(1, (1, 1), padding='same')(attn)
#     attn = layers.Activation('sigmoid')(attn)
#     out = layers.Multiply()([h, attn])
#     out = layers.Add()([x, out])
#     return out

# def generator(input_shape=(192, 256, 3)):
#     """
#     Builds a generator model with both SR and Gradient branches and multi-scale processing
#     :param input_shape: Shape of the input tensor
#     :return: Generator model
#     """
#     inputs = layers.Input(shape=input_shape)
    
#     # SR Branch
#     sr_x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
#     sr_x = layers.Activation('relu')(sr_x)
#     for _ in range(8):
#         sr_x = sr_res_block(sr_x, 64)
#     sr_x = layers.Conv2D(64, (3, 3), padding='same')(sr_x)
#     sr_x = layers.UpSampling2D(size=(2, 2))(sr_x)
#     sr_x = attention_block(sr_x, 64)
#     sr_x = layers.UpSampling2D(size=(2, 2))(sr_x)
#     sr_output = layers.Conv2D(3, (3, 3), padding='same')(sr_x)

#     # Gradient Branch
#     grad_x = tf.image.sobel_edges(inputs)
#     grad_x = layers.Conv2D(64, (3, 3), padding='same')(grad_x)
#     grad_x = layers.Activation('relu')(grad_x)
#     for _ in range(8):
#         grad_x = sr_res_block(grad_x, 64)
#     grad_x = layers.Conv2D(64, (3, 3), padding='same')(grad_x)
#     grad_x = layers.UpSampling2D(size=(2, 2))(grad_x)
#     grad_x = layers.Conv2D(1, (1, 1), padding='same')(grad_x)
    
#     # Multi-Scale Processing
#     scale1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
#     scale1 = sr_res_block(scale1, 64)
    
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
#     scale2 = sr_res_block(scale2, 64)
#     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
    
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(inputs)
#     scale3 = sr_res_block(scale3, 64)
#     scale3 = layers.UpSampling2D(size=(4, 4))(scale3)
    
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

#     # Final Fusion Block
#     combined_features = layers.Concatenate()([sr_output, grad_x, multi_scale])
#     combined_output = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(combined_features)

#     return Model(inputs, combined_output)

import tensorflow as tf
from tensorflow.keras import layers, Model

# Define Advanced Blocks
def sr_res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    res = layers.Conv2D(filters, (1, 1), padding='same')(res)  # Ensure shapes match
    x = layers.Add()([x, res])
    return x

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

def sobel_edges_layer(x):
    edges = tf.image.sobel_edges(x)
    edges = tf.reshape(edges, [-1, edges.shape[1], edges.shape[2], edges.shape[3] * edges.shape[4]])
    return edges

def resize_to_shape(target_shape):
    def layer(x):
        return tf.image.resize(x, target_shape)
    return layers.Lambda(layer)

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # SR Branch
    sr_x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    sr_x = layers.Activation('relu')(sr_x)
    for _ in range(8):
        sr_x = sr_res_block(sr_x, 64)
    sr_x = layers.Conv2D(64, (3, 3), padding='same')(sr_x)
    sr_x = layers.UpSampling2D(size=(2, 2))(sr_x)
    sr_x = attention_block(sr_x, 64)
    sr_x = layers.UpSampling2D(size=(2, 2))(sr_x)
    sr_output = layers.Conv2D(3, (3, 3), padding='same')(sr_x)
    sr_output = resize_to_shape((768, 1024))(sr_output)  # Ensure matching shapes

    # Gradient Branch
    grad_x = layers.Lambda(sobel_edges_layer)(inputs)
    grad_x = layers.Conv2D(64, (3, 3), padding='same')(grad_x)
    grad_x = layers.Activation('relu')(grad_x)
    for _ in range(8):
        grad_x = sr_res_block(grad_x, 64)
    grad_x = layers.Conv2D(64, (3, 3), padding='same')(grad_x)
    grad_x = layers.UpSampling2D(size=(2, 2))(grad_x)
    grad_x = layers.Conv2D(3, (1, 1), padding='same')(grad_x)  # Ensure matching shapes
    grad_x = resize_to_shape((768, 1024))(grad_x)  # Ensure matching shapes
    
    # Multi-Scale Processing
    scale1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    scale1 = sr_res_block(scale1, 64)
    scale1 = resize_to_shape((768, 1024))(scale1)  # Ensure matching shapes
    
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    scale2 = layers.Conv2D(64, (3, 3), padding='same')(scale2)  # Ensure matching shapes
    scale2 = sr_res_block(scale2, 64)
    scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
    scale2 = resize_to_shape((768, 1024))(scale2)  # Ensure matching shapes
    
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(inputs)
    scale3 = layers.Conv2D(64, (3, 3), padding='same')(scale3)  # Ensure matching shapes
    scale3 = sr_res_block(scale3, 64)
    scale3 = layers.UpSampling2D(size=(4, 4))(scale3)
    scale3 = resize_to_shape((768, 1024))(scale3)  # Ensure matching shapes
    
    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Final Fusion Block
    combined_features = layers.Concatenate()([sr_output, grad_x, multi_scale])
    combined_output = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(combined_features)

    return Model(inputs, combined_output)



# Custom Layer for Resizing
class ResizeLayer(layers.Layer):
    """
    Custom Layer for resizing tensors to a given size
    """
    def __init__(self, size):
        super(ResizeLayer, self).__init__()
        self.size = size

    def call(self, inputs):
        return tf.image.resize(inputs, self.size)

# Residual Block for Discriminator
def res_block(x, filters):
    """
    Residual Block for the Discriminator
    :param x: Input tensor
    :param filters: Number of filters for the convolutional layers
    :return: Output tensor after applying residual block
    """
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    return x

def discriminator(input_shape=(768, 1024, 3)):
    """
    Builds a refined discriminator model with multi-scale processing and PatchGAN-style convolutions
    :param input_shape: Shape of the input tensor
    :return: Discriminator model
    """
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
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # PatchGAN-style convolutions
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    # Final convolution layer
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)

# Build the refined generator model with multi-scale processing
refined_sr_and_grad_generator_with_multiscale = generator()
refined_sr_and_grad_generator_with_multiscale.summary()

# Build the refined discriminator model
refined_discriminator = discriminator(input_shape=(768, 1024, 3))
refined_discriminator.summary()
