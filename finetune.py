import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
import matplotlib.pyplot as plt
from sd_train_utils.data_loader import create_dataframe
from sd_train_utils.tokenize import process_text
from sd_train_utils.prepare_tf_dataset import prepare_dataset
from sd_train_utils.visualize_dataset import save_sample_batch_images
from sd_train_utils.trainer import Trainer


MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True

dataset_visualize_image_path = "sample_batch_images.png"
directory = '/content/drive/MyDrive/webvid-10m-dataset/grid_images_1'

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

# Prepare the dataset.
training_dataset = prepare_dataset(np.array(data_frame["image_path"]), tokenized_texts, batch_size=4)

# Take a sample batch and investigate.
sample_batch = next(iter(training_dataset))
for k in sample_batch:
    print(k, sample_batch[k].shape)

save_sample_batch_images(sample_batch, dataset_visualize_image_path)

# Utilize trainer class for finetuning
class DynamicCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_dir, save_freq):
        super(DynamicCheckpoint, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq

    def on_batch_end(self, batch, logs=None):
        total_steps = self.model.optimizer.iterations.numpy()
        if total_steps % self.save_freq == 0:
            epoch = self.params['epochs']
            step = total_steps
            filepath = os.path.join(self.ckpt_dir, f'ckpt_epoch{epoch}_step{step}.h5')
            self.model.save_weights(filepath)
            print(f'Saving checkpoint at epoch {epoch}, step {step}: {filepath}')

if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder = ImageEncoder()
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH)
vae = tf.keras.Model(
    image_encoder.input,
    image_encoder.layers[-2].output,
)
noise_scheduler = NoiseScheduler()

diffusion_ft_trainer = Trainer(
    diffusion_model=diffusion_model,
    vae=vae,
    noise_scheduler=noise_scheduler,
    use_mixed_precision=USE_MP,
)

optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
)
diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

epochs = 10

if os.path.exists('models'):
    ckpt_path = "models"
elif os.path.exists('/content/drive/MyDrive/models'):
    ckpt_path = '/content/drive/MyDrive/models'
else:
    ckpt_path = 'models'

model_path = os.path.join(ckpt_path, 'finetuned_stable_diffusion.h5')

# Build the model by running some data through it
dummy_data = {'images': sample_batch['images'], 'encoded_text': sample_batch['encoded_text']}
diffusion_ft_trainer(dummy_data)  # This ensures the model variables are created

if os.path.exists(model_path):
    # Load the model weights from the checkpoint
    diffusion_ft_trainer.load_weights(model_path)
    print(f"Checkpoint loaded from {model_path}")

dynamic_ckpt_callback = DynamicCheckpoint(ckpt_dir=ckpt_path, save_freq=500)

diffusion_ft_trainer.fit(training_dataset, epochs=epochs, callbacks=[dynamic_ckpt_callback])
