from PIL import Image, ImageFilter
import os

from PIL import Image, ImageFilter
import os

def process_gifs(source_folder, dest_folder, target_size=(128, 128), grid_size=4):
    """
    Processes GIFs:
      1) Skips the first frame and processes the rest
      2) Center-crops each frame to a square
      3) Resizes the cropped square to target_size
      4) Optionally applies a filter
      5) Creates a grid (4x4, 2x2, or single images) from those frames
    """

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_name in os.listdir(source_folder):
        if file_name.endswith(".gif"):
            file_path = os.path.join(source_folder, file_name)

            # Open the GIF
            with Image.open(file_path) as img:
                frames = []
                
                try:
                    img.seek(1)  # Skip the first frame
                    while True:
                        # Copy the current frame
                        frame = img.copy()

                        # Convert to RGB
                        frame = frame.convert("RGB")

                        # 1) (Optional) Center-crop to square
                        width, height = frame.size
                        min_dim = min(width, height)
                        left = (width - min_dim) // 2
                        top = (height - min_dim) // 2
                        right = left + min_dim
                        bottom = top + min_dim
                        frame = frame.crop((left, top, right, bottom))
                        
                        # 2) Resize after cropping
                        frame = frame.resize(target_size, Image.LANCZOS)

                        # 3) (Optional) Apply a detail filter
                        frame = frame.filter(ImageFilter.DETAIL)

                        frames.append(frame)

                        # Move to the next frame
                        img.seek(img.tell() + 1)

                except EOFError:
                    print(f"Finished processing '{file_name}' "
                          f"with {len(frames)} frames (excluding first frame).")

                # Build grids / save frames
                if frames:
                    if grid_size == 4:
                        # Create 4x4 grid image
                        grid_image = Image.new(
                            'RGB',
                            (target_size[0] * grid_size, target_size[1] * grid_size)
                        )
                        for i in range(grid_size * grid_size):
                            frame = frames[i % len(frames)]
                            x = (i % grid_size) * target_size[0]
                            y = (i // grid_size) * target_size[1]
                            grid_image.paste(frame, (x, y))
                        
                        # Save
                        grid_output_path = os.path.join(
                            dest_folder,
                            "4x4_grid_image_of_" + file_name.replace('.gif', '.jpg')
                        )
                        grid_image.save(grid_output_path, format='JPEG')
                        print(f"4x4 Grid image saved to {grid_output_path}")

                    elif grid_size == 2:
                        # Select 4 frames from the list
                        if len(frames) == 16:
                            frame_indices = [0, 4, 8, 12]
                        else:
                            frame_indices = [
                                0,
                                len(frames) // 3,
                                2 * len(frames) // 3,
                                len(frames) - 1
                            ]
                        selected_frames = [frames[i] for i in frame_indices]

                        # Build 2x2 grid
                        grid_image = Image.new(
                            'RGB',
                            (target_size[0] * grid_size, target_size[1] * grid_size)
                        )
                        for i, frame in enumerate(selected_frames):
                            x = (i % grid_size) * target_size[0]
                            y = (i // grid_size) * target_size[1]
                            grid_image.paste(frame, (x, y))

                        # Save
                        grid_output_path = os.path.join(
                            dest_folder,
                            "2x2_grid_image_of_" + file_name.replace('.gif', '.jpg')
                        )
                        grid_image.save(grid_output_path, format='JPEG')
                        print(f"2x2 Grid image saved to {grid_output_path}")

                    elif grid_size == 1:
                        # Save each frame as an individual 512x512 image
                        for i, frame in enumerate(frames):
                            # Further upscale if needed:
                            resized_frame = frame.resize((512, 512), Image.LANCZOS)

                            single_frame_output_path = os.path.join(
                                dest_folder,
                                f"frame_{i + 1}_of_{file_name.replace('.gif', '.jpg')}"
                            )
                            resized_frame.save(single_frame_output_path, format='JPEG')
                            print(f"Single frame image saved to {single_frame_output_path}")


# def process_gifs(source_folder, dest_folder, target_size=(128, 128), grid_size = 4):
#     # Ensure the destination folders exist
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)

#     # Loop through all files in the source folder
#     for file_name in os.listdir(source_folder):
#         if file_name.endswith(".gif"):
#             # Construct full file path
#             file_path = os.path.join(source_folder, file_name)
            

#             # Open the GIF
#             with Image.open(file_path) as img:
#                 # Create a list to hold the frames
#                 frames = []
                
#                 # Skip the first frame and process the rest
#                 try:
#                     img.seek(1)  # Skip the first frame
#                     while True:
#                         frame = img.copy().resize(target_size, Image.LANCZOS)  # Resize each frame
#                         frame = frame.convert("RGB")  # Convert to RGB mode
#                         frame = frame.filter(ImageFilter.DETAIL)
#                         # frame = frame.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
#                         frames.append(frame)
#                         img.seek(img.tell() + 1)  # Move to the next frame
#                 except EOFError:
#                     print(f"Finished processing '{file_name}' with {len(frames)} frames (excluding first frame).")
#             # # Open the GIF
#             # with Image.open(file_path) as img:
#             #     # Create a list to hold the frames
#             #     frames = []
                
#             #     # Try to extract the first 16 frames
#             #     try:
#             #         for i in range(16):
#             #             img.seek(i)
#             #             frame = img.copy().resize(target_size, Image.LANCZOS)  # Resize each frame
#             #             frame = frame.convert("RGB")
#             #             frame = frame.filter(ImageFilter.SHARPEN)
#             #             frames.append(frame)
#             #     except EOFError:
#             #         print(f"Warning: '{file_name}' has less than 16 frames.")


#                 if frames:
#                     if grid_size == 4:
#                         # Create 4x4 grid image
#                         grid_image = Image.new('RGB', (target_size[0] * grid_size, target_size[1] * grid_size))
#                         for i in range(grid_size * grid_size):
#                             frame = frames[i % len(frames)]  # Cycle through frames if fewer than 16
#                             x = i % grid_size * target_size[0]
#                             y = i // grid_size * target_size[1]
#                             grid_image.paste(frame, (x, y))
                        
#                         # Save 4x4 grid image
#                         grid_output_path = os.path.join(dest_folder, "4x4_grid_image_of_" + file_name.replace('.gif', '.jpg'))
#                         grid_image.save(grid_output_path, format='JPEG')
#                         print(f"4x4 Grid image saved to {grid_output_path}")

#                     elif grid_size == 2:
#                         # Select 4 evenly spaced frames
#                         frame_indices = [0, 4, 8, 12] if len(frames) == 16 else [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
#                         selected_frames = [frames[i] for i in frame_indices]

#                         # Create 2x2 grid image
#                         grid_image = Image.new('RGB', (target_size[0] * grid_size, target_size[1] * grid_size))
#                         for i in range(grid_size * grid_size):
#                             frame = selected_frames[i]
#                             x = i % grid_size * target_size[0]
#                             y = i // grid_size * target_size[1]
#                             grid_image.paste(frame, (x, y))
                        
#                         # Save 2x2 grid image
#                         grid_output_path = os.path.join(dest_folder, "2x2_grid_image_of_" + file_name.replace('.gif', '.jpg'))
#                         grid_image.save(grid_output_path, format='JPEG')
#                         print(f"2x2 Grid image saved to {grid_output_path}")

#                     elif grid_size == 1:
#                         # Save each frame as an individual 512x512 image
#                         for i, frame in enumerate(frames):
#                             resized_frame = frame.resize((512, 512), Image.LANCZOS).convert("RGB")  # Convert to RGB
#                             single_frame_output_path = os.path.join(dest_folder, f"frame_{i + 1}_of_{file_name.replace('.gif', '.jpg')}")
#                             resized_frame.save(single_frame_output_path, format='JPEG')
#                             print(f"Single frame image saved to {single_frame_output_path}")




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



source_folder = 'gifs_homer_simpson_original'
dest_folder = 'homer_simpson_4x4_images'
process_gifs(source_folder, dest_folder, target_size=(128, 128), grid_size=4)

# dest_folder = 'homer_simpson_2x2_images'
# process_gifs(source_folder, dest_folder, target_size=(256, 256), grid_size=2)

# dest_folder = 'homer_simpson_single_images'
# process_gifs(source_folder, dest_folder, target_size=(512, 512), grid_size=1)

# dest_folder = 'homer_simpson_4x4_1024_images'
# process_gifs(source_folder, dest_folder, target_size=(256, 256), grid_size=4)

# dest_folder = 'homer_simpson_4x4_2048_images'
# process_gifs(source_folder, dest_folder, target_size=(512, 512), grid_size=4)


# Example usage
# grid_image_path = 'gifs_homer_simpson_resized_128_16F_Grid'  # Specify the path to your grid image
# output_gif_path = 'reconstruct_gifs'  # Specify where to save the recreated GIF
# print("D")
# recreate_gifs_from_grids(grid_image_path, output_gif_path)