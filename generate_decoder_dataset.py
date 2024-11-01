from keras_cv.models.stable_diffusion.decoder import Decoder
from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

img_height = img_width = 512
stable_diffusion = StableDiffusion(
    img_width=img_width, img_height=img_height
)

stable_diffusion.diffusion_model.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_4x4_scaled_linear_simpsons/ckpt_epoch_100.h5_2x2_diffusion_model.h5")

decoder = Decoder(512, 512)

image_folder = '/content/drive/MyDrive/homer_simpson_resized_128_16F_Grid'
#image_folder = 'webvid10m_dataset_summed_approach/2x2_grid_images'

# Preprocessing function for the images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assumes the images are in JPEG format
    image = tf.image.resize(image, (512, 512))       # Resize to the shape expected by the Decoder
    image = (image / 127.5) - 1.0  # Rescaled the images to [-1, 1] range
    return image

# Load image dataset
def load_image_paths(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]
    return image_paths

# Function to create latent vectors from image filenames (prompts)
def generate_latents_from_filenames(image_paths):
    latents = []
    for image_path in image_paths:
        #print(image_path)
        prompt = os.path.splitext(os.path.basename(image_path))[0].replace('_', ' ')
        print(prompt)
        latent = stable_diffusion.text_to_latent(prompt,batch_size=1, unconditional_guidance_scale=7.5,num_steps=100)  # Generate latent for the prompt
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
    dataset = dataset.map(load_image_and_latent, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Adjusted load_image_and_latent function
def load_image_and_latent(image_path, latent):
    image = preprocess_image(image_path)
    latent = tf.squeeze(latent)  # Ensure latent tensor is in the correct shape

    return latent, image

# Ensure your decoder expects the correct input shape
def check_dataset_shapes(dataset):
    for latent, image in dataset.take(1):
        print("Latent shape:", latent.shape)
        print("Image shape:", image.shape)

        tf.print("Image min value:", tf.reduce_min(image))
        tf.print("Image max value:", tf.reduce_max(image))
        
        tf.print("Image shape:", tf.shape(image))
        tf.print("Latent shape:", tf.shape(latent))


train_dataset = create_latent_image_dataset(image_folder)
check_dataset_shapes(train_dataset)

path = '/content/drive/MyDrive/stable_diffusion_4x4/decoder_dataset_scaled_linear_7.5_guidance_simpsons'

def save_dataset(path):
    train_dataset.save(path)

save_dataset(path)

