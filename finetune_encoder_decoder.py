import numpy as np
import tensorflow as tf
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.decoder import Decoder
from sd_train_utils.data_loader import create_dataframe


# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True

# Paths
directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_4x4_2048_images'

# Create the dataframe from image directory
data_frame = create_dataframe(directory)
image_paths = np.array(data_frame["image_path"])  # Extract paths to images

# Load 4x4 grid images and prepare the dataset
def load_and_preprocess_image(file_path):
    # Read and decode the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize to desired resolution (512x512)
    image = tf.image.resize(image, [RESOLUTION, RESOLUTION])
    # Normalize to [-1, 1]
    image = (image / 127.5) - 1.0
    return image

def prepare_grid_dataset(image_paths, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Provide input as both x and y for reconstruction task
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create the dataset for training the VAE
grid_dataset = prepare_grid_dataset(image_paths)

# Initialize the Encoder and Decoder
img_height, img_width = 512, 512
encoder = ImageEncoder(download_weights=False)
decoder = Decoder(img_height=img_height, img_width=img_width, download_weights=False)

# Define the VAE model combining Encoder and Decoder
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        latents = self.encoder(inputs)
        reconstructed = self.decoder(latents)
        return reconstructed

# Custom Loss Function for VAE Training
reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

# Define the VAE loss function
def vae_loss(y_true, y_pred):
    return reconstruction_loss_fn(y_true, y_pred)

# Initialize the VAE
vae_model = VAE(encoder=encoder, decoder=decoder)
vae_model.compile(optimizer='adam', loss=vae_loss)

# Train the VAE
vae_model.fit(grid_dataset, epochs=50)  # Dataset already batches data

# Save Encoder and Decoder Weights Separately
encoder_weights_path = "/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/trained_vae_encoder.h5"
decoder_weights_path = "/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/trained_vae_decoder.h5"

vae_model.encoder.save_weights(encoder_weights_path)
vae_model.decoder.save_weights(decoder_weights_path)

print(f"Encoder weights saved to {encoder_weights_path}")
print(f"Decoder weights saved to {decoder_weights_path}")
