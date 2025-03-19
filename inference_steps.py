from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import time
import psutil
import os
import pynvml

try:
    pynvml.nvmlInit()
    gpu_available = True
except pynvml.NVMLError_LibraryNotFound:
    print("NVIDIA drivers not found. GPU usage will not be tracked.")
    gpu_available = False

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

img_height = img_width = 512
grid_model = StableDiffusion(
    img_width=img_width, img_height=img_height
)

# Load weights
grid_model.diffusion_model.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_7/best_mixed_3.h5")
grid_model.decoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_decoder.h5")

# Define prompts
prompts = [
    "grid image of homer cuddling a piglet",
    "grid_image_of_homer_blinking_slowly",
    "grid_image_of_homer_cutting_hair_with_large_scissors",
    "grid_image_of_homer_doing_ninja_moves_using_a_batton.jpg",
    "grid_image_of_homer_doing_weights_exercises_on_a_bench_press.jpg",
    "grid_image_of_homer_drooling.jpg",
    "grid_image_of_homer_eating_a_pink_doughnut.jpg",
    "grid_image_of_homer_escaping_fire",
    "grid_image_of_homer_frowning_and_runs_his_eyes_sideways.jpg",
    "grid_image_of_homer_hammering_a_nail_on_a_roof",
    "grid_image_of_homer_in_a_ballerina_dress_and_rotating",
]

images_to_generate = 1
total_inference_time = 0
total_memory_usage = 0
total_gpu_memory_usage = 0
num_prompts = len(prompts)
num_steps = 50

inference_times = []
memory_usages = []
gpu_memory_usages = []

prompt = "grid_image_of_homer_in_a_ballerina_dress_and_rotating",
for i in range(10):

    # Generate latents for the given prompt
    generated_latents = grid_model.text_to_latent(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=5, num_steps=i+1
    )

    # Decode the latents to images
    generated_images = grid_model.latent_to_image(generated_latents)


    for j, image_array in enumerate(generated_images):
        img = Image.fromarray(image_array)

        # Save the image with a filename reflecting the prompt
        sanitized_prompt = "".join([c if c.isalnum() or c in " _-" else "_" for c in prompt])  # Sanitize the prompt for file name
        file_path = f"/content/drive/MyDrive/stable_diffusion_4x4/dataset/inferenced_50/{i}.png"

        img.save(file_path)
        print(f"Saved: {file_path}")

