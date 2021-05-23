import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='scaled_poisson')
class ScaledPoisson(tf.keras.losses.Loss):
    '''Poisson loss scaled by (usually) avereage of each output unit.

    Based on NDN3's Poisson loss implementation.
    '''
    def __init__(self, output_dim_weights):
        self.output_dim_weights = output_dim_weights
        self.reduction = tf.keras.losses.Reduction.AUTO

    def __call__(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is None
        return -tf.reduce_sum(tf.divide(
                    tf.multiply(y_true, tf.math.log(tf.keras.backend.epsilon() + y_pred)) - y_pred,
                    self.output_dim_weights + tf.keras.backend.epsilon()))

    def get_config(self):
        return {'output_dim_weights': self.output_dim_weights}