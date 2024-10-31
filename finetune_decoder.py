from keras_cv.models.stable_diffusion.decoder import Decoder
from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

img_height = img_width = 512

decoder = Decoder(512, 512)

path = '/content/drive/MyDrive/stable_diffusion_4x4/decoder_dataset_scaled_linear_7.5_guidance'

# # # Load the dataset
reloaded_dataset = tf.data.Dataset.load(path)

train_size = int(0.8 * len(reloaded_dataset))  # 80% for training
val_size = len(reloaded_dataset) - train_size

train_dataset = reloaded_dataset.take(train_size)
val_dataset = reloaded_dataset.skip(train_size)

# # Now `train_dataset` contains pairs of (latent, image) for training the decoder
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
loss_function = tf.keras.losses.MeanSquaredError()

decoder.compile(optimizer=optimizer, loss=loss_function)

# If shapes are correct, proceed to training
history = decoder.fit(train_dataset, validation_data=val_dataset, epochs=100)
# #decoder.save('/content/drive/MyDrive/models/decoder_4x4/decoder_4x4_new.h5')
# decoder.save('/content/drive/MyDrive/models/decoder_4x4/decoder_4x4_new.h5')
decoder.save('/content/drive/MyDrive/stable_diffusion_4x4/decoder_model_scaled_linear/decoder3.h5')