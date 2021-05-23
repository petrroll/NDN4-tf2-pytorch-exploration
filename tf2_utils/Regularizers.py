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
        shape: Tuple[int, ...] = (-1,)
    ):
        """Create new Laplacian2DRegulizer using Discrete Laplace operator and conv2d operation. 

        Currently only supports being applied to Dense layer i.e. [input, output]-shaped weights.

        Args:
            alpha (float, optional): Regularization strength. Defaults to 0.01.
            shape (tuple, optional): Real shape (to properly reshape flat weights of Dense layer) [H, W] for 2D. Defaults to (-1,).
        """        
        self.alpha = alpha
        self.shape = shape
        self.dtype = dtype

        laplacian_kernel = tf.constant([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], dtype=dtype)
        laplacian_kernel = laplacian_kernel[:, :, tf.newaxis, tf.newaxis]
        self.__laplacian_kernel = laplacian_kernel 


    def __call__(self, x):
        if len(x.shape) == 2:   # Flattened 2D input
            # [input, output] -input==H*W> [-1, H, W, output]
            _, dim_out = x.shape        # x: dims [input, output]

            x = tf.expand_dims(x, 0)    # add batch dim
            x = tf.expand_dims(x, -1)   # add input channels dim
            x = tf.reshape(x, (-1,) + self.shape + (dim_out,))  # reshape 1D dense weights to 2D

        elif len(x.shape) == 4: # Conv2D filter
            # [filter_height, filter_width, in_channels, out_channels] -> [I, H, W, output]
            _, _, _, dim_out = x.shape
            x = tf.transpose(x, [2, 0, 1, 3])
        else:
            raise NotImplementedError("Currently supports only regularizing layers with [input, output] shape.")


        # Broadcast Laplac operator kernel for each layer's output channel
        laplacian_kernel = tf.broadcast_to(self.__laplacian_kernel, (3, 3, dim_out, 1))

        # Needs to be depthwise_conv2d otherwise we square aggregate over channels -> not true Laplace L2
        conv_result = tf.nn.depthwise_conv2d(x, laplacian_kernel, [1, 1, 1, 1], "SAME")
        conv_result = tf.math.square(conv_result)

        return self.alpha * tf.math.reduce_sum(conv_result)

    def get_config(self):
        return {
            'alpha': self.alpha,
            'dtype': self.dtype,
            'shape': self.shape,
            }

