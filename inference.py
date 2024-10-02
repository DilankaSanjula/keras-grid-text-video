import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.decoder import Decoder
from sd_train_utils.trainer import Trainer
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib.pyplot as plt
from PIL import Image

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True
NUM_INFERENCE_STEPS = 5
GUIDANCE_SCALE = 7.5

# Paths
pretrained_weights_path = '/content/drive/MyDrive/models/best_weights.h5'
pretrained_decoder_path = '/content/drive/MyDrive/models/decoder_4x4/decoder.h5'

# Initialize components
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

#set_global_policy('mixed_float16')

text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH)
vae = Decoder(512, 512)  # Decoder (VAE) to convert latent representation to image
noise_scheduler = NoiseScheduler()


diffusion_ft_trainer = Trainer(
    diffusion_model=diffusion_model,
    vae=vae,
    noise_scheduler=noise_scheduler,
    use_mixed_precision=USE_MP,
)

# Load weights
if os.path.exists(pretrained_weights_path):
    diffusion_model.load_weights(pretrained_weights_path)
    print(f"Pretrained diffusion model weights loaded from {pretrained_weights_path}")

if os.path.exists(pretrained_decoder_path):
    vae.load_weights(pretrained_decoder_path)
    print(f"Pretrained VAE weights loaded from {pretrained_decoder_path}")

# Prepare text prompt
def encode_prompt(prompt):
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.encode(prompt)
    tokens = tokens[:MAX_PROMPT_LENGTH]
    tokens += [49407] * (MAX_PROMPT_LENGTH - len(tokens))  # Padding
    tokens = np.array(tokens)[None, :]  # Batch dimension
    return tokens

def generate_image(prompt):
    tokens = encode_prompt(prompt)
    attention_mask = np.where(tokens != 49407, 1, 0)
    # # Pass both tokens and attention mask to the text encoder
    encoded_text = text_encoder([tokens, attention_mask])

    # Start from random noise
    batch_size = 1
    latent = tf.random.normal((batch_size, RESOLUTION // 8, RESOLUTION // 8, 4))

    print("Latent dtype:", latent.dtype)
    
    encoded_text = tf.cast(encoded_text, tf.float32)
    print("Encoded text dtype after casting:", encoded_text.dtype)

    latent = tf.cast(latent, tf.float32)

    # Inference loop
    for i in range(NUM_INFERENCE_STEPS):
        t = NUM_INFERENCE_STEPS - i - 1
        timestep = tf.convert_to_tensor([t], dtype=tf.float32)

        # Denoise the latent representation
        timestep_embedding =  diffusion_ft_trainer.get_timestep_embedding(timestep)
        print("Timestep embedding dtype:", timestep_embedding.dtype)
        model_output = diffusion_model([latent, timestep_embedding, encoded_text])

        # Apply guidance (if needed)
        if GUIDANCE_SCALE > 1:
            uncond_latent = diffusion_model([latent, timestep_embedding, encoded_text * 0])
            model_output = uncond_latent + GUIDANCE_SCALE * (model_output - uncond_latent)

        # Update latent representation
        model_output = tf.cast(model_output, tf.float32)
        latent = noise_scheduler.step(model_output, t, latent)

    # Decode the latent representation into an image
    image = vae(latent)
    image = (image + 1) / 2  # Rescale to [0, 1]

    return image

def save_image(image, path):
    # Convert the TensorFlow tensor to a PIL Image and ensure it's in the correct format
    image = Image.fromarray((image.numpy()[0] * 255).astype('uint8'))

    # Save the image
    image.save(path)

# Define the path where you want to save the image
save_path = "/content/drive/MyDrive/models/futuristic_cityscape.png"


# Example usage
prompt = "A futuristic cityscape at sunset"


# Example usage
prompt = "A futuristic cityscape at sunset"
image = generate_image(prompt) 
save_image(image, save_path)