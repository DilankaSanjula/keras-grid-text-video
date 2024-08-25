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

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# Paths
pretrained_weights_path = '/content/drive/MyDrive/models/best_weights.h5'
pretrained_vae_path = '/content/drive/MyDrive/models/vae.h5'

# Initialize components
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH)
vae = Decoder()  # Decoder (VAE) to convert latent representation to image
noise_scheduler = NoiseScheduler()

# Load weights
if os.path.exists(pretrained_weights_path):
    diffusion_model.load_weights(pretrained_weights_path)
    print(f"Pretrained diffusion model weights loaded from {pretrained_weights_path}")

if os.path.exists(pretrained_vae_path):
    vae.load_weights(pretrained_vae_path)
    print(f"Pretrained VAE weights loaded from {pretrained_vae_path}")

# Prepare text prompt
def encode_prompt(prompt):
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.encode(prompt)
    tokens = tokens[:MAX_PROMPT_LENGTH]
    tokens += [49407] * (MAX_PROMPT_LENGTH - len(tokens))  # Padding
    tokens = np.array(tokens)[None, :]  # Batch dimension
    return tokens

def generate_image(prompt):
    encoded_text = text_encoder(encode_prompt(prompt))
    
    # Start from random noise
    batch_size = 1
    latent = tf.random.normal((batch_size, RESOLUTION // 8, RESOLUTION // 8, 4))

    # Inference loop
    for i in range(NUM_INFERENCE_STEPS):
        t = NUM_INFERENCE_STEPS - i - 1
        timestep = tf.convert_to_tensor([t], dtype=tf.int32)

        # Denoise the latent representation
        timestep_embedding = diffusion_model.get_timestep_embedding(timestep)
        model_output = diffusion_model([latent, timestep_embedding, encoded_text])

        # Apply guidance (if needed)
        if GUIDANCE_SCALE > 1:
            uncond_latent = diffusion_model([latent, timestep_embedding, encoded_text * 0])
            model_output = uncond_latent + GUIDANCE_SCALE * (model_output - uncond_latent)

        # Update latent representation
        latent = noise_scheduler.step(model_output, t, latent)

    # Decode the latent representation into an image
    image = vae(latent)
    image = (image + 1) / 2  # Rescale to [0, 1]

    return image

# Example usage
prompt = "A futuristic cityscape at sunset"
generated_image = generate_image(prompt)

# Save the image
generated_image_path = "generated_image.png"
keras.preprocessing.image.save_img(generated_image_path, generated_image[0])
print(f"Generated image saved to {generated_image_path}")