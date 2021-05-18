import tensorflow as tf 

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes

from typing import Tuple

@tf.keras.utils.register_keras_serializable(package='Custom', name='laplacian2D')
class Laplacian2DRegulizer(tf.keras.regularizers.Regularizer):
    
    def __init__(
        self, 
        alpha: float = 0.01,
        dtype: AcceptableDTypes = None,
        shape: Tuple[int, ...] = (-1,)):
        """Create new Laplacian2DRegulizer using Discrete Laplace operator and conv2d operation. 

        Currently only supports being applied to Dense layer i.e. [input, output]-shaped weights.

        Args:
            alpha (float, optional): Regularization strength. Defaults to 0.01.
            shape (tuple, optional): Real shape (to properly reshape flat weights of Dense layer) [H, W] for 2D. Defaults to (-1,).
        """        
        self.alpha = alpha
        self.shape = shape

        laplacian_kernel = tf.constant([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], dtype=dtype)
        laplacian_kernel = laplacian_kernel[:, :, tf.newaxis, tf.newaxis]
        self.__laplacian_kernel = laplacian_kernel 


    def __call__(self, x):
        if len(x.shape) == 2:
            _, dim_out = x.shape        # x: dims [input, output]
        else:
            raise NotImplementedError("Currently supports only regularizing layers with [input, output] shape.")

        x = tf.expand_dims(x, 0)    # add batch dim
        x = tf.expand_dims(x, -1)   # add input channels dim
        x = tf.reshape(x, (-1,) + self.shape + (dim_out,))  # reshape 1D dense weights to 2D

        # Broadcast Laplac operator kernel for each layer's output channel
        laplacian_kernel = tf.broadcast_to(self.__laplacian_kernel, (3, 3, dim_out, 1))

        # Needs to be depthwise_conv2d otherwise we square aggregate over channels -> not true Laplace L2
        conv_result = tf.nn.depthwise_conv2d(x, laplacian_kernel, [1, 1, 1, 1], "SAME")
        conv_result = tf.math.square(conv_result)

        return self.alpha * tf.math.reduce_sum(conv_result)

    def get_config(self):
        return {'alpha': float(self.alpha)}

