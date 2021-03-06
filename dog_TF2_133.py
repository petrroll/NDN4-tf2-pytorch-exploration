#%%
import math
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

#%%
tf.random.set_seed(42)

#%%
input_data = np.load("./Data/region1/training_inputs.npy").astype(np.float32)
output_data = np.load("./Data/region1/training_set.npy").astype(np.float32)

input_val_data = np.load("./Data/region1/validation_inputs.npy").astype(np.float32)
output_val_data = np.load("./Data/region1/validation_set.npy").astype(np.float32)

scale_mean = np.mean(input_data)
scale_std = np.std(input_data)

input_data = (input_data - scale_mean) / scale_std
input_val_data = (input_val_data - scale_mean) / scale_std

#%%
input_shape = (31, 31, 1, )
output_size = 103
hidden_ratio = 0.2

#%%
input_data = input_data.reshape(-1, *input_shape)
input_val_data = input_val_data.reshape(-1, *input_shape)

#%%
time_str = datetime.now().strftime("%d-%m_%H-%M-%S")
exp_str = "dog_TF2_133"

#%%
import tf2_utils.Regularizers as tfuR
import tf2_utils.Metrics as tfuM
import tf2_utils.Layers as tfuL
import tf2_utils.Losses as tfuLo

tf.config.set_visible_devices([], 'GPU')

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=input_shape))
model.add(tfuL.DiffOfGaussians(
    units=9,
    activation=None
    ))
model.add(tf.keras.layers.Dense(
    int(output_size*0.2), 
    "softplus", 
    ))
model.add(tf.keras.layers.Dense(
    int(output_size), 
    "softplus", 
    kernel_regularizer=tf.keras.regularizers.L2(0.1)
    ))


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfuLo.ScaledPoisson(np.mean(output_data, axis=0)),
    metrics=[
        tfuM.PearsonR(y_shape=(103, ))
        ]
)
#%%
model.fit(
    x=input_data, y=output_data,
    batch_size=16,
    epochs=5000,
    callbacks=[tf.keras.callbacks.TensorBoard(
        log_dir=f"./Logdir/{exp_str}/{time_str}/",
        profile_batch='500,550')
        ],
    validation_data=(input_val_data, output_val_data),
    validation_freq = 100,
    verbose=0)
