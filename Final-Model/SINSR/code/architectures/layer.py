import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# ResizeLayer: Custom layer for resizing inputs to a target size
class ResizeLayer(layers.Layer):
    def __init__(self, size):
        """
        Initializes the ResizeLayer with the target size.
        
        Args:
        - size (tuple): Target size to which inputs will be resized.
        """
        super(ResizeLayer, self).__init__()
        self.size = size

    def call(self, inputs):
        """
        Resizes the input tensor to the target size using bilinear interpolation.
        
        Args:
        - inputs (Tensor): Input tensor to be resized.
        
        Returns:
        - Tensor: Resized tensor.
        """
        return tf.image.resize(inputs, self.size)
