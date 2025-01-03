import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
#from diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from sd_train_utils.data_loader import create_dataframe
from sd_train_utils.tokenize import process_text
from sd_train_utils.prepare_tf_dataset import prepare_dataset
from sd_train_utils.visualize_dataset import save_sample_batch_images
from sd_train_utils.trainer import Trainer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True

# cant use float32
# maxiumum batch size is 4
# tried adam and adamw no difference
# 

if USE_MP:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Define a perceptual loss using VGG16
vgg = VGG16(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
selected_layers = ["block3_conv3", "block4_conv3"]
outputs = [vgg.get_layer(name).output for name in selected_layers]
vgg_model = Model(inputs=vgg.input, outputs=outputs)
vgg_model.trainable = False

def vgg_perceptual_loss(y_true, y_pred):
    y_true_rgb = y_true[..., :3]  # Use RGB channels only
    y_pred_rgb = y_pred[..., :3]

    # Normalize to match VGG input requirements
    y_true_rgb = (y_true_rgb + 1.0) / 2.0  # Assuming inputs are [-1, 1]
    y_pred_rgb = (y_pred_rgb + 1.0) / 2.0

    # Extract VGG features
    true_features = vgg_model(y_true_rgb)
    pred_features = vgg_model(y_pred_rgb)

    # Compute perceptual loss
    loss = 0.0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.abs(true_feat - pred_feat))
    return loss

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

mse = tf.keras.losses.MeanSquaredError()


def combined_loss(y_true, y_pred):
    # Compute individual losses
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    #perceptual = vgg_perceptual_loss(y_true, y_pred)

    # Ensure all losses are float32 for consistency
    mse_loss = tf.cast(mse_loss, tf.float32)
    ssim = tf.cast(ssim, tf.float32)
    #perceptual = tf.cast(perceptual, tf.float32)
    tf.print("MSE Loss:", mse_loss, "SSIM Loss:", ssim)

    # Combine losses with weights
    total_loss = mse_loss + 0.5 * ssim
    return total_loss

# Paths
dataset_visualize_image_path = "sample_batch_images.png"
#directory = '/content/drive/MyDrive/webvid-10-dataset-2/4x4_grid_images'

#directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_single_images'
#directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_2x2_images'
#directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_4x4_images'
#directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/mixed_dataset'

directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_best_quality_mixed'


#pretrained_weights_path = '/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_1/ckpt_epoch_70.h5_2x2_diffusion_model.h5'
pretrained_weights_path = '/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_7/best_mixed_3.h5'
# pretrained_vae = '/content/drive/MyDrive/models/vae.h5'

# Learning Parameters
lr = 1e-7
beta_1, beta_2 = 0.9, 0.999
weight_decay = (1e-2,)
epsilon = 1e-08

# Create the dataframe
data_frame = create_dataframe(directory)

# Collate the tokenized captions into an array.
tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))
all_captions = list(data_frame["caption"].values)
for i, caption in enumerate(all_captions):
    tokenized_texts[i] = process_text(caption)

# Prepare the dataset
training_dataset = prepare_dataset(np.array(data_frame["image_path"]), tokenized_texts, batch_size=4)

# Take a sample batch and investigate
sample_batch = next(iter(training_dataset))
for k in sample_batch:
    print(k, sample_batch[k].shape)

#save_sample_batch_images(sample_batch, dataset_visualize_image_path)

# Initialize the trainer and compile it
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder_weights_fpath = '/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_encoder.h5'

image_encoder = ImageEncoder(download_weights=False)
image_encoder.load_weights(image_encoder_weights_fpath)
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH, download_weights=True)
vae = tf.keras.Model(
    image_encoder.input,
    image_encoder.layers[-2].output,
)
noise_scheduler = NoiseScheduler(beta_schedule="scaled_linear")


#Load the pretrained weights
if os.path.exists(pretrained_weights_path):
    diffusion_model.load_weights(pretrained_weights_path)
    print(f"Pretrained diffusion model weights loaded from {pretrained_weights_path}")


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_dir, save_freq=10):
        super(CustomModelCheckpoint, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:  # Save every `save_freq` epochs
            filepath = os.path.join(self.ckpt_dir, f'epoch_{epoch + 1}.h5')
            self.model.save_weights(filepath)
            print(f'Saving checkpoint at epoch {epoch + 1}: {filepath}')

# Define the checkpoint directory and frequency
#ckpt_dir = '/content/drive/MyDrive/models/vae_diffusion_model_2x2'
ckpt_dir = '/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_7'
save_frequency = 50  # Save every 10 epochs

# Fine-tuning
epochs = 150  # Adjust the number of epochs as needed
custom_ckpt_callback = CustomModelCheckpoint(ckpt_dir=ckpt_dir, save_freq=save_frequency)


diffusion_ft_trainer = Trainer(
    diffusion_model=diffusion_model,
    vae=vae,
    noise_scheduler=noise_scheduler,
    use_mixed_precision=USE_MP,
)


# Define the learning rate schedule
initial_learning_rate = lr
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

# Initialize the AdamW optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=weight_decay,  # Decoupled weight decay
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    clipnorm=1.0  # Gradient clipping for stability
)

#diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")
diffusion_ft_trainer.compile(optimizer=optimizer, loss=mse)

best_weights_filepath = os.path.join(ckpt_dir, 'best_model.h5')

model_checkpoint_callback = ModelCheckpoint(
    filepath=best_weights_filepath,
    save_weights_only=True,
    monitor='loss',  # Monitor validation loss
    mode='min',  # Use 'min' to save weights with the lowest validation loss
    save_best_only=True,  # Save only the best weights
    verbose=1  # Prints a message when saving the best weights
)

# Callback: Reduce learning rate on plateau
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='loss',          # Monitor training loss
    factor=0.5,              # Reduce learning rate by half
    patience=10,              # Number of epochs with no improvement
    min_lr=1e-9,             # Minimum learning rate
    verbose=1                # Prints a message when reducing learning rate
)

# Callback: Early stopping (optional)
early_stopping = EarlyStopping(
    monitor='loss',          # Monitor training loss
    patience=20,             # Stop training if no improvement for 10 epochs
    verbose=1
)

# Train the model with the callback
diffusion_ft_trainer.fit(
    training_dataset,
    epochs=epochs,
    callbacks=[custom_ckpt_callback, model_checkpoint_callback]
)
