from typing import Optional
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='min_max')
class MinMax(tf.keras.constraints.Constraint):
  """Constrains tensors to have individual-values between min and max."""

  def __init__(
      self, 
      min: Optional[float] = None, 
      max: Optional[float] = None
      ):
    self.min = min
    self.max = max

  def __call__(self, w):
    if self.min and self.max:
        return tf.clip_by_value(w, self.min, self.max)
    elif self.min:
        return tf.maximum(w, self.min)
    elif self.max:
        return tf.minimum(w, self.max)

  def get_config(self):
    return {'min': self.min, 'max': self.max}
