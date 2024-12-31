import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from sd_train_utils.data_loader import create_dataframe
from sd_train_utils.tokenize import process_text
from sd_train_utils.prepare_tf_dataset_highloss import prepare_dataset
from sd_train_utils.visualize_dataset import save_sample_batch_images
from sd_train_utils.trainer2 import Trainer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True

if USE_MP:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

mse_loss = tf.keras.losses.MeanSquaredError()

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    # Ensure inputs are float32 for consistent operations
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute individual losses
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)

    # Combine losses with weights
    total_loss = mse_loss + 0.4 * ssim
    return total_loss


# Paths
dataset_visualize_image_path = "sample_batch_images.png"
directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/mixed_dataset'
pretrained_weights_path = '/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_4/final.h5'

# Learning Parameters
lr = 1e-4
beta_1, beta_2 = 0.9, 0.999
weight_decay = (1e-2,)
epsilon = 1e-08

# Create the dataframe
data_frame = create_dataframe(directory)

# Tokenize captions
tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))
all_captions = list(data_frame["caption"].values)
for i, caption in enumerate(all_captions):
    tokenized_texts[i] = process_text(caption)

# Prepare dataset
training_dataset = prepare_dataset(np.array(data_frame["image_path"]), tokenized_texts, batch_size=4)

# Model and trainer setup
image_encoder_weights_fpath = '/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_encoder.h5'
image_encoder = ImageEncoder(download_weights=False)
image_encoder.load_weights(image_encoder_weights_fpath)
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH, download_weights=True)
vae = tf.keras.Model(image_encoder.input, image_encoder.layers[-2].output)
noise_scheduler = NoiseScheduler(beta_schedule="scaled_linear")

if os.path.exists(pretrained_weights_path):
    diffusion_model.load_weights(pretrained_weights_path)
    print(f"Pretrained diffusion model weights loaded from {pretrained_weights_path}")

diffusion_ft_trainer = Trainer(
    diffusion_model=diffusion_model,
    vae=vae,
    noise_scheduler=noise_scheduler,
    use_mixed_precision=USE_MP,
)

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    clipnorm=1.0
)

diffusion_ft_trainer.compile(optimizer=optimizer, loss=mse_loss)


best_weights_filepath = os.path.join('/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_7', 'best_model.h5')
model_checkpoint_callback = ModelCheckpoint(filepath=best_weights_filepath, save_weights_only=True, monitor='loss', mode='min', save_best_only=True, verbose=1)

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

# Train the model
diffusion_ft_trainer.fit(
    training_dataset,
    epochs=4,
    callbacks=[model_checkpoint_callback, reduce_lr_on_plateau, early_stopping]
)