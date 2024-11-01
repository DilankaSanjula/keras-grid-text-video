from PIL import Image
import os

def process_gifs(source_folder, dest_folder, grid_folder):
    # Ensure the destination folders exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    if not os.path.exists(grid_folder):
        os.makedirs(grid_folder)

    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".gif"):
            # Construct full file path
            file_path = os.path.join(source_folder, file_name)
            
            # Open the GIF
            with Image.open(file_path) as img:
                # Create a list to hold the frames
                frames = []
                
                # Try to extract the first 16 frames
                try:
                    for i in range(16):
                        img.seek(i)
                        frames.append(img.copy())
                except EOFError:
                    print(f"Warning: '{file_name}' has less than 16 frames.")

                # Process and save only if there are frames
                if frames:
                    # Save the frames as a new GIF if we have 16 frames
                    if len(frames) == 16:
                        output_path = os.path.join(dest_folder, file_name)
                        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)
                        print(f"Processed {file_name} saved to {output_path}")

                    # Create a single image grid from the frames
                    grid_size = 4  # Always use a 4x4 grid
                    frame_width, frame_height = frames[0].size
                    grid_image = Image.new('RGB', (frame_width * grid_size, frame_height * grid_size))

                    # Paste each frame into the grid, repeat frames if necessary
                    for i in range(grid_size * grid_size):
                        frame = frames[i % len(frames)]  # Cycle through frames if fewer than 16
                        x = i % grid_size * frame_width
                        y = i // grid_size * frame_height
                        grid_image.paste(frame, (x, y))

                    # Change output path extension to .jpg
                    grid_output_path = os.path.join(grid_folder, "grid_" + file_name.replace('.gif', '.jpg'))
                    grid_image.save(grid_output_path, format='JPEG')
                    print(f"Grid image saved to {grid_output_path}")



def recreate_gifs_from_grids(grid_folder, output_folder, frame_size=(128, 128), grid_size=(4, 4)):
    """
    Recreates GIFs from all grid images in a specified folder.

    Args:
    grid_folder (str): Path to the folder containing grid images.
    output_folder (str): Path to save the recreated GIFs.
    frame_size (tuple): The width and height of each frame in the grid.
    grid_size (tuple): The number of columns and rows in the grid.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Process each file in the grid folder
    for file_name in os.listdir(grid_folder):
        if file_name.endswith(".jpg"):  # Assuming grid images are PNGs
            grid_image_path = os.path.join(grid_folder, file_name)
            output_gif_path = os.path.join(output_folder, file_name.replace('.jpg', '.gif'))

            with Image.open(grid_image_path) as grid_img:
                frames = []
                frame_width, frame_height = frame_size
                grid_columns, grid_rows = grid_size

                # Extract frames from the grid image
                for row in range(grid_rows):
                    for col in range(grid_columns):
                        left = col * frame_width
                        top = row * frame_height
                        right = left + frame_width
                        bottom = top + frame_height

                        frame = grid_img.crop((left, top, right, bottom))
                        frames.append(frame)

                # Save the frames as a GIF
                if frames:
                    frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], loop=0)
                    print(f"GIF recreated and saved to {output_gif_path}")




# Example usage
source_folder = 'gifs_homer_simpson_resized_128'
dest_folder_128_16F_gif = 'gifs_homer_simpson_resized_128_16F'
dest_folder_128_16F_grid = 'gifs_homer_simpson_resized_128_16F_Grid'
process_gifs(source_folder, dest_folder_128_16F_gif, dest_folder_128_16F_grid)


# Example usage
# grid_image_path = 'gifs_homer_simpson_resized_128_16F_Grid'  # Specify the path to your grid image
# output_gif_path = 'reconstruct_gifs'  # Specify where to save the recreated GIF
# print("D")
# recreate_gifs_from_grids(grid_image_path, output_gif_path)