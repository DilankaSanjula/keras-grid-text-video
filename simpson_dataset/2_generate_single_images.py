from PIL import Image, ImageFilter
import os

def single_images(source_folder, dest_folder, target_size=(512, 512)):

    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        
        if file_name.endswith(".gif"):
            # Construct full file path
            file_path = os.path.join(source_folder, file_name)
            
            with Image.open(file_path) as img:
                frames = []
                total_frames = img.n_frames
                print(total_frames)
                    
                img.seek(3)
                frame = img.copy().resize(target_size, Image.LANCZOS)  # Resize each frame
                frame = frame.convert("RGB")  # Convert to RGB mode
                frame = frame.filter(ImageFilter.DETAIL)

                # Construct the output file path
                output_file_name = f"{os.path.splitext(file_name)[0]}.jpg"
                output_file_path = os.path.join(dest_folder, output_file_name)
                # Save the frame as a JPEG
                frame.save(output_file_path, "JPEG")
                print(f"Saved {output_file_path}")
                


source_folder = 'gifs_homer_simpson_original'

dest_folder = 'homer_single_images'

single_images(source_folder, dest_folder)

