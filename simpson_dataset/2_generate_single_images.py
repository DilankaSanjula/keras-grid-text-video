from PIL import Image
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
                # for i in range(16):
                #     img.seek(i)
                #     frame = img.copy()
                    
                #     frame = frame.resize(target_size, Image.LANCZOS)
                    
                #     frames.append(frame)

                #     # Construct the output file path
                #     output_file_name = f"frame_{i}_of_{os.path.splitext(file_name)[0]}.jpg"
                #     output_file_path = os.path.join(dest_folder, output_file_name)
                #     # Save the frame as a JPEG
                #     frame.convert("RGB").save(output_file_path, "JPEG")
                #     print(f"Saved {output_file_path}")
                


source_folder = 'gifs_homer_simpson_resized_256'

dest_folder = 'homer_simpson_single_images'

single_images(source_folder, dest_folder)

