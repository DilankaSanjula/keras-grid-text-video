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
from sd_train_utils.prepare_tf_dataset import prepare_dataset
from sd_train_utils.visualize_dataset import save_sample_batch_images
from sd_train_utils.trainer import Trainer

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True

# Paths
dataset_visualize_image_path = "sample_batch_images.png"
directory = '/content/drive/MyDrive/webvid-10m-dataset/grid_images_1'
pretrained_weights_path = '/content/drive/MyDrive/models/ckpt_epoch_4.h5'
pretrained_vae = '/content/drive/MyDrive/models/vae.h5'

# Learning Parameters
lr = 1e-5
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
training_dataset = prepare_dataset(np.array(data_frame["image_path"]), tokenized_texts, batch_size=6)

# Take a sample batch and investigate
sample_batch = next(iter(training_dataset))
for k in sample_batch:
    print(k, sample_batch[k].shape)

save_sample_batch_images(sample_batch, dataset_visualize_image_path)

# Initialize the trainer and compile it
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder = ImageEncoder()
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH)
vae = tf.keras.Model(
    image_encoder.input,
    image_encoder.layers[-2].output,
)
noise_scheduler = NoiseScheduler()



# Load the pretrained weights
if os.path.exists(pretrained_weights_path):
    diffusion_model.load_weights(pretrained_weights_path)
    print(f"Pretrained diffusion model weights loaded from {pretrained_weights_path}")

try:
    if os.path.exists(pretrained_vae):
        vae.load_weights(pretrained_vae)
        print(f"Pretrained vae weights loaded from {pretrained_vae}")
except Exception as exp:
    print(exp)


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_dir):
        super(CustomModelCheckpoint, self).__init__()
        self.ckpt_dir = ckpt_dir

    def on_epoch_end(self, epoch, logs=None):
        filepath = os.path.join(self.ckpt_dir, f'ckpt_epoch_{epoch + 1}')
        self.model.save_weights(filepath)
        print(f'Saving checkpoint at epoch {epoch + 1}: {filepath}')

# Fine-tuning
epochs = 8  # Adjust the number of epochs as needed
ckpt_dir = '/content/drive/MyDrive/models'
custom_ckpt_callback = CustomModelCheckpoint(ckpt_dir=ckpt_dir)


diffusion_ft_trainer = Trainer(
    diffusion_model=diffusion_model,
    vae=vae,
    noise_scheduler=noise_scheduler,
    use_mixed_precision=USE_MP,
)


# Compile the trainer
optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
)
diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")


diffusion_ft_trainer.fit(training_dataset, epochs=epochs, callbacks=[custom_ckpt_callback])
