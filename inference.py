from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

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

# We just reload the weights of the fine-tuned diffusion model.
#grid_model.diffusion_model.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_1/ckpt_epoch_70.h5_2x2_diffusion_model.h5")
grid_model.diffusion_model.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_7/best_mixed_3.h5")
#grid_model.diffusion_model.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/diffusion_model_stage_7/best.h5")

grid_model.decoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_decoder.h5")
#grid_model.decoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_model_scaled_linear/decoder_simpsons2.h5")

# Print the summary of the decoder model
print("\n=== Decoder Model Summary ===")
grid_model.decoder.summary()

#prompts = ["Grid image of close up of handsome happy male professional typing on mobile phone in good mood"]
# prompts = ["grid image of homer cuddling a piglet"]
# images_to_generate = 1
# outputs = {}

# for prompt in prompts:
#     generated_latents = grid_model.text_to_latent(
#         prompt, batch_size=images_to_generate, unconditional_guidance_scale=10,num_steps=200
#     )

# generated_images = grid_model.latent_to_image(generated_latents)

# print("Decoded image min value:", generated_images.min())
# print("Decoded image max value:", generated_images.max())

# for i, image_array in enumerate(generated_images):
#     img = Image.fromarray(image_array)
#     file_path = f"/content/drive/MyDrive/stable_diffusion_4x4/stage_7_best_{i}.png"
#     img.save(file_path)
#     print(f"Saved: {file_path}")
    


# Define multiple prompts
prompts = [
    "grid image of homer cuddling a piglet",
    "grid_image_of_homer_blinking_slowly",
    "grid_image_of_homer_cutting_hair_with_large_scissors",
    "grid_image_of_homer_doing_ninja_moves_using_a_batton.jpg",
    "grid_image_of_homer_doing_weights_exercises_on_a_bench_press.jpg",
    "grid_image_of_homer_drinking_a_barrel_of_beer_without_a_tshirt.jpg",
    "grid_image_of_homer_drooling.jpg",
    "grid_image_of_homer_eating_a_pink_doughnut.jpg",
    "grid_image_of_homer_escaping_fire",
    "grid_image_of_homer_frowning_and_runs_his_eyes_sideways.jpg",
    "grid_image_of_homer_hammering_a_nail_on_a_roof",
    "grid_image_of_homer_hiding_in_grass",
    "grid_image_of_homer_in_a_ballerina_dress_and_rotating",
    "grid_image_of_homer_in_a_blue_coat_thinking_about_something",
    "grid_image_of_homer_lighting_cigarette_using_a_money_note",
    "grid_image_of_homer_loosening_his_belt_before_eating",
    "grid_image_of_homer_making_corn_flakes_dressed_like_a_chef",
    "grid_image_of_homer_on_a_floating_bed_enjoying_a_drink",
    "grid_image_of_homer_pouring_bleach_on_his_eyes",
    "grid_image_of_homer_rotating_on_a_chair_in_a_control room",
    "grid_image_of_homer_screaming_with_his_hands_on_his_head_and_tongue_out",
    "grid_image_of_homer_screaming",
    "grid_image_of_homer_showing_middle_finger_while_drowning",
    "grid_image_of_homer_simpson_enjoys_driving_on_a_beautiful_day",
    "grid_image_of_homer_smoking",
    "grid_image_of_homer_staring_at_his_tummy_on_bed_wearing_a_red_pant",
    "grid_image_of_homer_vacation_floating_in_water",
    "grid_image_of_homer_simpson_rotates_on_a_chair_in_a_control_room",
    "grid_image_of_homer_running_on_treadmill",
]

images_to_generate = 1

for prompt in prompts:
    # Generate latents for the given prompt
    generated_latents = grid_model.text_to_latent(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=5, num_steps=100
    )
    
    # Decode the latents to images
    generated_images = grid_model.latent_to_image(generated_latents)
    
    for i, image_array in enumerate(generated_images):
        img = Image.fromarray(image_array)
        
        # Save the image with a filename reflecting the prompt
        sanitized_prompt = "".join([c if c.isalnum() or c in " _-" else "_" for c in prompt])  # Sanitize the prompt for file name
        file_path = f"/content/drive/MyDrive/stable_diffusion_4x4/dataset/inferenced/{sanitized_prompt}_{i}.png"
        
        img.save(file_path)
        print(f"Saved: {file_path}")