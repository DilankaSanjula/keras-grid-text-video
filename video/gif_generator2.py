from PIL import Image
import os

# Define directories
input_dir = 'real_images'
output_dir = 'real_gifs'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sub_image_size = (128, 128)

# Process all .png files in the input directory
for filename in os.listdir(input_dir):
    print(filename)
    if filename.lower().endswith('.jpg'):
        input_image_path = os.path.join(input_dir, filename)
        input_image = Image.open(input_image_path)
        width, height = input_image.size

        # Crop the image into sub-images of the specified size
        sub_images = []
        for y in range(0, height, sub_image_size[1]):
            for x in range(0, width, sub_image_size[0]):
                box = (x, y, x + sub_image_size[0], y + sub_image_size[1])
                sub_images.append(input_image.crop(box))

        # Create and save the GIF using the cropped sub-images
        print(filename)
        filename = filename.replace("4x4_grid_image_of_", "")
        gif_filename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.gif')
        
        if sub_images:
            sub_images[0].save(gif_filename, format='GIF', append_images=sub_images[1:],
                                 save_all=True, duration=250, loop=0)
        print(f"GIF saved as {gif_filename}")
