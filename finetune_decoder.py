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

decoder = Decoder(512, 512)

image_folder = '/content/drive/MyDrive/webvid-10-dataset-2/debug_images'
#image_folder = '/content/drive/MyDrive/webvid-10-dataset-2/4x4_grid_images'
#image_folder = 'webvid10m_dataset_summed_approach/2x2_grid_images'

# Preprocessing function for the images
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assumes the images are in JPEG format
    image = tf.image.resize(image, (512, 512))       # Resize to the shape expected by the Decoder
    image = image / 255.0  # Normalize the images to [0, 1] range
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
        latent = stable_diffusion.text_to_latent(prompt,batch_size=1, unconditional_guidance_scale=40,num_steps=50)  # Generate latent for the prompt
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

    print("Image min value:", tf.reduce_min(image).numpy())
    print("Image max value:", tf.reduce_max(image).numpy())

    return latent, image

# Ensure your decoder expects the correct input shape
def check_dataset_shapes(dataset):
    for latent, image in dataset.take(1):
        print("Latent shape:", latent.shape)
        print("Image shape:", image.shape)

        # Check the range of image values to ensure they are in the correct range
        print("Image min value:", tf.reduce_min(image).numpy())
        print("Image max value:", tf.reduce_max(image).numpy())

        # Optionally, you can also check the latent value ranges
        print("Latent min value:", tf.reduce_min(latent).numpy())
        print("Latent max value:", tf.reduce_max(latent).numpy())


train_dataset = create_latent_image_dataset(image_folder)
check_dataset_shapes(train_dataset)

# #path = 'decoder_dataset/'
# path = '/content/drive/MyDrive/models/decoder_dataset'

# def save_dataset(path):
#     train_dataset.save(path)

# save_dataset(path)





# # # # Load the dataset
# reloaded_dataset = tf.data.Dataset.load(path)

# train_size = int(0.8 * len(reloaded_dataset))  # 80% for training
# val_size = len(reloaded_dataset) - train_size

# train_dataset = reloaded_dataset.take(train_size)
# val_dataset = reloaded_dataset.skip(train_size)

# # # Now `train_dataset` contains pairs of (latent, image) for training the decoder
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
# loss_function = tf.keras.losses.MeanSquaredError()

# decoder.compile(optimizer=optimizer, loss=loss_function)

# # If shapes are correct, proceed to training
# history = decoder.fit(train_dataset, validation_data=val_dataset, epochs=10)
# decoder.save('/content/drive/MyDrive/models/decoder_4x4/decoder_4x4.h5')