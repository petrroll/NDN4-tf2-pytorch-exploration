
import tensorflow as tf
import numpy as np

import tf2_utils.Constrains as tfuC

@tf.keras.utils.register_keras_serializable(package='Custom', name='diff_of_gaussians')
class DiffOfGaussians(tf.keras.layers.Layer):
    '''
    Implements Difference of gaussians layer from https://github.com/petrroll/msc-thesis
    '''
    def __init__(self, units: int=32, activation=None, use_bias:bool=True):
        super(DiffOfGaussians, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def _get_gaussian(self, X, Y, alpha, sigma, ux, uy):
        return (alpha) * (tf.exp(-((X - ux) ** 2 + (Y - uy) ** 2) / 2 / sigma) / (2*sigma*np.pi))   # implementation is slightly different than paper (not sigma^2)

    def _get_DoG(self):
        W = np.array(range(self.width))
        H = np.array(range(self.height))

        X, Y = np.meshgrid(W, H)
        X = tf.constant(np.expand_dims(X, 2).astype(np.float32))
        Y = tf.constant(np.expand_dims(Y, 2).astype(np.float32))

        gauss_1 = self._get_gaussian(
            X, Y, 
            self.a1, self.s1,
            self.x, self.y,
            )
        gauss_2 = self._get_gaussian(
            X, Y, 
            self.a2, self.s1 + self.s2, 
            self.x, self.y,
            )

        return gauss_1 - gauss_2

    def _add_DoG_param(self, name: str, min: float, max: float, max_constrain: bool = True):
        return self.add_weight(
            name=name,
            shape=(1, 1, self.units, ),
            initializer=tf.keras.initializers.RandomUniform(min, max),
            constraint=tfuC.MinMax(min, max if max_constrain else None),
            trainable=True
        )

    def build(self, input_shape):
        _, height, width, _ = input_shape 
        self.width = width
        self.height = height

        self.a1 = self._add_DoG_param("a1", tf.keras.backend.epsilon(), 10.0, False)
        self.a2 = self._add_DoG_param("a2", tf.keras.backend.epsilon(), 10.0, False)

        self.s1 = self._add_DoG_param("s1", 1, 0.82*max(width, height), True)
        self.s2 = self._add_DoG_param("s2", 1, 0.82*max(width, height), True)

        self.x = self._add_DoG_param("x", int(0.08*width), width-int(0.08*width), True)
        self.y = self._add_DoG_param("y", int(0.08*height), height-int(0.08*height), True)

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, 4) # [N, H, W, C] -> [N, H, W, C, 1]

        DoG_filter = self._get_DoG()
        DoG_filter = tf.expand_dims(DoG_filter, 0) # [H, W, num_filters] -> [0, H, W, num_filters]
        DoG_filter = tf.expand_dims(DoG_filter, 3) # [0, H, W, num_filters] -> [0, H, W, 1, num_filters]
        gaussed = tf.multiply(inputs, DoG_filter)

        out_pre = tf.reduce_sum(gaussed, axis=[1, 2, 3])
        out_pos = self.activation(out_pre)

        return out_pos + self.bias if self.use_bias else out_pos

    def get_config(self):
        return {
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
        }


