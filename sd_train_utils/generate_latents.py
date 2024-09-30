import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from sd_train_utils.trainer import Trainer
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True
NUM_INFERENCE_STEPS = 2
GUIDANCE_SCALE = 7.5

# Paths
pretrained_weights_path = '/content/drive/MyDrive/models/vae_diffusion_model/ckpt_epoch_100.h5_2x2_diffusion_model.h5'
pretrained_vae_path = '/content/drive/MyDrive/models/decoder_4x4/decoder_model.h5'

# Initialize components
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder = ImageEncoder()
text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH)
noise_scheduler = NoiseScheduler()
vae = tf.keras.Model(
    image_encoder.input,
    image_encoder.layers[-2].output,
)
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


# Prepare text prompt
def encode_prompt(prompt):
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.encode(prompt)
    tokens = tokens[:MAX_PROMPT_LENGTH]
    tokens += [49407] * (MAX_PROMPT_LENGTH - len(tokens))  # Padding
    tokens = np.array(tokens)[None, :]  # Batch dimension
    return tokens

def generate_latent(prompt):
    tokens = encode_prompt(prompt)
    attention_mask = np.where(tokens != 49407, 1, 0)
    # # Pass both tokens and attention mask to the text encoder
    encoded_text = text_encoder([tokens, attention_mask])

    # Start from random noise
    batch_size = 1
    latent = tf.random.normal((batch_size, RESOLUTION // 8, RESOLUTION // 8, 4))

    # Inference loop
    for i in range(NUM_INFERENCE_STEPS):
        t = NUM_INFERENCE_STEPS - i - 1
        timestep = tf.convert_to_tensor([t], dtype=tf.int32)

        # Denoise the latent representation
        timestep_embedding = diffusion_ft_trainer.get_timestep_embedding(timestep)
        model_output = diffusion_model([latent, timestep_embedding, encoded_text])

        # Apply guidance (if needed)
        if GUIDANCE_SCALE > 1:
            uncond_latent = diffusion_model([latent, timestep_embedding, encoded_text * 0])
            model_output = uncond_latent + GUIDANCE_SCALE * (model_output - uncond_latent)

        # Update latent representation
        latent = noise_scheduler.step(model_output, t, latent)

    return latent

# # Example usage
# prompt = "A futuristic cityscape at sunset"
# generated_latent = generate_latent(prompt)