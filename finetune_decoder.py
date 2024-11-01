from keras_cv.models.stable_diffusion.decoder import Decoder
from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

img_height = img_width = 512

decoder = Decoder(512, 512)

path = '/content/drive/MyDrive/stable_diffusion_4x4/decoder_dataset_scaled_linear_7.5_guidance_simpsons'

# Load the dataset
reloaded_dataset = tf.data.Dataset.load(path)

train_size = int(0.8 * len(reloaded_dataset))  # 80% for training
val_size = len(reloaded_dataset) - train_size

train_dataset = reloaded_dataset.take(train_size)
val_dataset = reloaded_dataset.skip(train_size)

# Define the combined loss function
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255.0))
    return mse + ssim_loss

# Use the combined loss function when compiling the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
decoder.compile(optimizer=optimizer, loss=combined_loss, metrics=['accuracy'])

# If shapes are correct, proceed to training
history = decoder.fit(train_dataset, validation_data=val_dataset, epochs=50)
decoder.save('/content/drive/MyDrive/stable_diffusion_4x4/decoder_model_scaled_linear/decoder_simpsons2.h5')
