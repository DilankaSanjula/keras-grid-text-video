import numpy as np
import tensorflow as tf
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.decoder import Decoder
from sd_train_utils.data_loader import create_dataframe
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision training
set_global_policy("mixed_float16")

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 256  # Reduced resolution to save memory
USE_MP = True

# Paths
directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_4x4_2048_images'

# Create the dataframe from image directory
data_frame = create_dataframe(directory)
image_paths = np.array(data_frame["image_path"])  # Extract paths to images

# Load 4x4 grid images and prepare the dataset
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [RESOLUTION, RESOLUTION])
    image = (image / 127.5) - 1.0
    return image

def prepare_grid_dataset(image_paths, batch_size=4):  # Reduced batch size
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (x, x))  # Provide input as both x and y
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create the dataset for training the VAE
grid_dataset = prepare_grid_dataset(image_paths)

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Initialize the Encoder and Decoder
encoder = ImageEncoder(download_weights=False)
decoder = Decoder(img_height=RESOLUTION, img_width=RESOLUTION, download_weights=False)

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        latents = self.encoder(inputs)
        reconstructed = self.decoder(latents)
        return reconstructed

# Define Loss and Model
reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

def vae_loss(y_true, y_pred):
    return reconstruction_loss_fn(y_true, y_pred)

vae_model = VAE(encoder=encoder, decoder=decoder)
vae_model.compile(optimizer='adam', loss=vae_loss)

# Train the model
vae_model.fit(grid_dataset, epochs=50)

# Save weights
vae_model.encoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/trained_vae_encoder.h5")
vae_model.decoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/trained_vae_decoder.h5")
