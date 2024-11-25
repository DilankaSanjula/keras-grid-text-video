import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
# from keras_cv.models.stable_diffusion.decoder import Decoder
from encoder import ImageEncoder
from decoder import Decoder
import os

MAX_PROMPT_LENGTH = 77
RESOLUTION = 512

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
    
vae_model = VAE(encoder=encoder, decoder=decoder)

vae_model.encoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_encoder.h5")
vae_model.decoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_decoder.h5")

# Function to display images
def display_images(original, reconstructed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow((original + 1.0) / 2.0)  # Undo normalization
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow((reconstructed + 1.0) / 2.0)  # Undo normalization
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")
    plt.show()


# Load and preprocess images
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [RESOLUTION, RESOLUTION])
    image = (image / 127.5) - 1.0
    return image

def prepare_grid_dataset(image_paths, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (x, x))  # Provide input as both x and y
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Preprocess a sample image
sample_image_path = "/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_4x4_images/4x4_grid_image_of_homer_simpson_cuddles_piglet.jpg"
original_image = load_and_preprocess_image(sample_image_path)
original_image = tf.expand_dims(original_image, axis=0)  # Add batch dimension

# Perform inference
latents = vae_model.encoder(original_image)
reconstructed_image = vae_model.decoder(latents)

# Remove batch dimension
original_image = tf.squeeze(original_image, axis=0)
reconstructed_image = tf.squeeze(reconstructed_image, axis=0)

# Save images to files
def save_images_to_file(original, reconstructed, output_dir="/content/drive/MyDrive/stable_diffusion_4x4/dataset/"):
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # Undo normalization for display
    original = (original + 1.0) / 2.0
    reconstructed = (reconstructed + 1.0) / 2.0

    # Save the original image
    plt.imshow(original)
    plt.axis("off")
    original_file = os.path.join(output_dir, "original_image.png")
    plt.savefig(original_file, bbox_inches='tight')
    plt.close()

    # Save the reconstructed image
    plt.imshow(reconstructed)
    plt.axis("off")
    reconstructed_file = os.path.join(output_dir, "reconstructed_image.png")
    plt.savefig(reconstructed_file, bbox_inches='tight')
    plt.close()

    print(f"Images saved to:\nOriginal: {original_file}\nReconstructed: {reconstructed_file}")

# Save the original and reconstructed images to files
save_images_to_file(original_image.numpy(), reconstructed_image.numpy())