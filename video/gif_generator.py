from PIL import Image
import os

input_image_path = 'images/grid_image_of_homer_escaping_fire_0.png'

# Directory where sub-images are saved
output_dir = 'generated_gifs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sub_image_size = (128, 128)

input_image = Image.open(input_image_path)

sub_images = []
image_index = 1
for y in range(0, 512, 128):
    for x in range(0, 512, 128):
        box = (x, y, x + 128, y + 128)
        sub_image = input_image.crop(box)
        sub_images.append(sub_image)
        image_index += 1

# Save each sub-image as a separate file in the output directory
image_paths = []
for idx, sub_image in enumerate(sub_images, start=1):
    img_path = os.path.join(output_dir, f'sub_image_{idx}.png')
    sub_image.save(img_path)
    image_paths.append(img_path)

# Create the GIF
gif_filename = 'output_animation_8.gif'
frames = [Image.open(image) for image in image_paths]
frames[0].save(gif_filename, format='GIF', append_images=frames[1:], save_all=True, duration=250, loop=0)

# Delete the sub-image files after creating the GIF
for img_path in image_paths:
    os.remove(img_path)

print(f"GIF saved as {gif_filename}")
