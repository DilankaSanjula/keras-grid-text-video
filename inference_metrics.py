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

for prompt in prompts:
    # Measure start time
    start_time = time.time()

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB

    # Get initial GPU memory usage
    if gpu_available and gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        initial_gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        initial_gpu_mem_used = initial_gpu_mem_info.used / 1024 / 1024  # in MB
    else:
        initial_gpu_mem_used = 0

    # Measure inference start time
    inference_start_time = time.time()

    # Generate latents for the given prompt
    generated_latents = grid_model.text_to_latent(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=5, num_steps=50
    )

    # Decode the latents to images
    generated_images = grid_model.latent_to_image(generated_latents)

    # Measure inference end time
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    total_inference_time += inference_time

    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # in MB
    memory_usage = final_memory - initial_memory  # Approximate memory used by the generation process
    total_memory_usage += memory_usage

    # Get final GPU memory usage
    if gpu_available and gpus:
        final_gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        final_gpu_mem_used = final_gpu_mem_info.used / 1024 / 1024  # in MB
        gpu_memory_usage = final_gpu_mem_used - initial_gpu_mem_used  # Approximate GPU memory used
    else:
        gpu_memory_usage = 0

    inference_times.append(inference_time)
    memory_usages.append(memory_usage)
    gpu_memory_usages.append(gpu_memory_usage)
    total_gpu_memory_usage += gpu_memory_usage

    for i, image_array in enumerate(generated_images):
        img = Image.fromarray(image_array)

        # Save the image with a filename reflecting the prompt
        sanitized_prompt = "".join([c if c.isalnum() or c in " _-" else "_" for c in prompt])  # Sanitize the prompt for file name
        file_path = f"/content/drive/MyDrive/stable_diffusion_4x4/dataset/inferenced_50/{sanitized_prompt}_{i}.png"

        img.save(file_path)
        print(f"Saved: {file_path}, Inference Time: {inference_time:.2f} seconds, Memory Usage: {memory_usage:.2f} MB, GPU Memory Usage: {gpu_memory_usage:.2f} MB")

print(f"\nTotal Inference Time for {num_prompts} prompts: {total_inference_time:.2f} seconds")
print(f"Average Inference Time per prompt: {total_inference_time / num_prompts:.2f} seconds")
print(f"Total Memory Usage for {num_prompts} prompts: {total_memory_usage:.2f} MB")
print(f"Average Memory Usage per prompt: {total_memory_usage / num_prompts:.2f} MB")
print(f"Total GPU Memory Usage for {num_prompts} prompts: {total_gpu_memory_usage:.2f} MB")
print(f"Average GPU Memory Usage per prompt: {total_gpu_memory_usage / num_prompts:.2f} MB")

# Plotting Inference Time vs. Memory Usage
plt.figure(figsize=(8, 6))
plt.scatter(memory_usages, inference_times, alpha=0.7, label='Memory vs. Inference Time')
plt.title("Inference Time vs. Memory Usage")
plt.xlabel("Memory Usage (MB)")
plt.ylabel("Inference Time (seconds)")
plt.grid(True)
plt.legend()
plt.savefig("/content/drive/MyDrive/stable_diffusion_4x4/dataset/inferenced/inference_time_vs_memory_usage.png")  # Save the plot
plt.show()