from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt
from PIL import Image


img_height = img_width = 512
grid_model = StableDiffusion(
    img_width=img_width, img_height=img_height
)

# We just reload the weights of the fine-tuned diffusion model.
grid_model.diffusion_model.load_weights("/content/drive/MyDrive/models/models/ckpt_epoch_100.h5_2x2_diffusion_model.h5")

grid_model.decoder.load_weight("/content/drive/MyDrive/models/decoder_4x4/decoder_4x4.h5")

prompts = ["a portrait of a king"]
images_to_generate = 1
outputs = {}


for prompt in prompts:
    generated_latents = grid_model.text_to_latent(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=40,num_steps=30
    )

generated_images = grid_model.latent_to_image(generated_latents)

for i, image_array in enumerate(generated_images):
    img = Image.fromarray(image_array)
    file_path = f"image2_{i}.png"
    img.save(file_path)
    print(f"Saved: {file_path}")
    
