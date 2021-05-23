"""Implements Pearson's correlation."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import weights_broadcast_ops

from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes

VALID_MULTIOUTPUT = {"raw_values", "uniform_average"}

@tf.keras.utils.register_keras_serializable(package='Custom', name='pearson_r')
class PearsonR(Metric):
    """Computes Pearson's R.
    Probably quite numerically unstable implmementation.

    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    
    @typechecked
    def __init__(
        self,
        name: str = "pearson_r",
        dtype: AcceptableDTypes = None,
        y_shape: Tuple[int, ...] = (),
        multioutput: str = "uniform_average",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.y_shape = y_shape

        if multioutput not in VALID_MULTIOUTPUT:
            raise ValueError(
                "The multioutput argument must be one of {}, but was: {}".format(
                    VALID_MULTIOUTPUT, multioutput
                )
            )

        self.multioutput = multioutput
        self.sum_xy = self.add_weight(
            name="sum_xy", shape=y_shape, initializer="zeros", dtype=dtype
        )

        self.sum_squared_x = self.add_weight(
            name="sum_squared_x", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.sum_squared_y = self.add_weight(
            name="sum_squared_y", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.sum_x = self.add_weight(
            name="sum_x", shape=y_shape, initializer="zeros", dtype=dtype
        )
        self.sum_y = self.add_weight(
            name="sum_y", shape=y_shape, initializer="zeros", dtype=dtype
        )


        self.count = self.add_weight(
            name="count", shape=y_shape, initializer="zeros", dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = tf.cast(y_true, dtype=self._dtype)
        y_pred = tf.cast(y_pred, dtype=self._dtype)

        if sample_weight is None:
            sample_weight = 1
        sample_weight = tf.cast(sample_weight, dtype=self._dtype)
        sample_weight = weights_broadcast_ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        weighted_y_true = y_true * sample_weight

        xy = tf.math.multiply(weighted_y_true, y_pred)
        self.sum_xy.assign_add(tf.reduce_sum(xy, axis=0))

        self.sum_squared_x.assign_add(tf.reduce_sum(weighted_y_true ** 2, axis=0))
        self.sum_squared_y.assign_add(tf.reduce_sum(y_pred ** 2, axis=0))
        self.sum_x.assign_add(tf.reduce_sum(weighted_y_true, axis=0))
        self.sum_y.assign_add(tf.reduce_sum(y_pred, axis=0))

        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))

    def result(self) -> tf.Tensor:
        raw_scores = tf.math.divide(
            self.sum_xy - (self.sum_x * self.sum_y / self.count)
            ,
            tf.keras.backend.epsilon() + tf.sqrt(self.sum_squared_x - ((self.sum_x**2) / self.count)) * tf.sqrt(self.sum_squared_y - ((self.sum_y**2)/self.count))
        )
        

        if self.multioutput == "raw_values":
            return raw_scores
        if self.multioutput == "uniform_average":
            return tf.reduce_mean(raw_scores)
        raise RuntimeError(
            "The multioutput attribute must be one of {}, but was: {}".format(
                VALID_MULTIOUTPUT, self.multioutput
            )
        )

    def reset_states(self) -> None:
        # The state of the metric will be reset at the start of each epoch.
        K.batch_set_value([(v, tf.zeros_like(v)) for v in self.variables])
