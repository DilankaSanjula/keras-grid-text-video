from keras_cv.models.stable_diffusion.decoder import Decoder
import tensorflow as tf
import os
from sd_train_utils.generate_latents import generate_latent

# Load the decoder with pre-trained weights
decoder = Decoder(512, 512)

image_folder = '/content/drive/MyDrive/webvid-10-dataset-2/4x4_grid_images'

# Preprocessing function for the images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assumes the images are in JPEG format
    image = tf.image.resize(image, (512, 512))       # Resize to the shape expected by the Decoder
    image = image / 255.0  # Normalize the images to [0, 1] range
    return image

# Load image dataset
def load_image_paths(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]
    return image_paths

# Function to create latent vectors from image filenames (prompts)
def generate_latents_from_filenames(image_paths):
    latents = []
    for image_path in image_paths:
        prompt = os.path.splitext(os.path.basename(image_path))[0].replace('_', ' ')
        latent = generate_latent(prompt)  # Generate latent for the prompt
        latents.append(latent)
    return latents

# Create a tf.data.Dataset that contains both latents and images
def create_latent_image_dataset(image_folder, batch_size=4):
    image_paths = load_image_paths(image_folder)
    
    # Generate latents using image filenames as prompts
    latents = generate_latents_from_filenames(image_paths)
    
    # Create a dataset from the image paths and latents
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, latents))
    
    # Map the dataset to load images and pair them with latents
    def load_image_and_latent(image_path, latent):
        image = preprocess_image(image_path)
        return latent, image  # Return tuple (latent, image)
    
    dataset = dataset.map(load_image_and_latent, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Create the training dataset with latents and images
train_dataset = create_latent_image_dataset(image_folder)

# Now `train_dataset` contains pairs of (latent, image) for training the decoder
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

decoder.compile(optimizer=optimizer, loss=loss_function)

history = decoder.fit(train_dataset, epochs=100)

decoder.save('/content/drive/MyDrive/models/decoder_4x4/decoder_model.h5')