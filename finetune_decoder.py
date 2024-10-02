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

stable_diffusion.diffusion_model.load_weights("/content/drive/MyDrive/models/vae_diffusion_model/ckpt_epoch_100.h5_2x2_diffusion_model.h5")
# Load the decoder with pre-trained weights

prompts = ["Grid image of close up of handsome happy male professional typing on mobile phone in good mood"]
images_to_generate = 1
outputs = {}


for prompt in prompts:
    generated_latents = stable_diffusion.text_to_latent(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=40,num_steps=30
    )

generated_images = stable_diffusion.latent_to_image(generated_latents)

for i, image_array in enumerate(generated_images):
    img = Image.fromarray(image_array)
    file_path = "/content/drive/MyDrive/models/loaded_4x4_test.png"
    img.save(file_path)
    print(f"Saved: {file_path}")


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
        print(image_path)
        prompt = os.path.splitext(os.path.basename(image_path))[0].replace('_', ' ')
        latent = stable_diffusion.text_to_latent(prompt,batch_size=1, unconditional_guidance_scale=40,num_steps=50)  # Generate latent for the prompt
        latents.append(latent)
    return latents

# Create a tf.data.Dataset that contains both latents and images
def create_latent_image_dataset(image_folder, batch_size=8):
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

# Create and check dataset
train_dataset = create_latent_image_dataset(image_folder)
check_dataset_shapes(train_dataset)

# Now `train_dataset` contains pairs of (latent, image) for training the decoder
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

decoder.compile(optimizer=optimizer, loss=loss_function)

# If shapes are correct, proceed to training
history = decoder.fit(train_dataset, epochs=20)
decoder.save('/content/drive/MyDrive/models/decoder_4x4/decoder_4x4.h5')