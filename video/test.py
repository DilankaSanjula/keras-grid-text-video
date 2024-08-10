import os
from PIL import Image
from moviepy.editor import VideoFileClip

resize_index_path = 'images'


def crop_center_square(frame):
    width, height = frame.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    return frame.crop((left, top, right, bottom))

def create_image_grid(frames, grid_size=(4, 4), image_size=(128, 128)):
    grid_width, grid_height = grid_size
    single_width, single_height = image_size
    grid_image = Image.new('RGB', (grid_width * single_width, grid_height * single_height))

    for i, frame in enumerate(frames):
        frame = frame.resize(image_size, Image.ANTIALIAS)
        x = (i % grid_width) * single_width
        y = (i // grid_width) * single_height
        grid_image.paste(frame, (x, y))

    return grid_image

def process_videos_in_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, filename in enumerate(os.listdir(folder_path)):

        try:
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, filename)
                clip = VideoFileClip(video_path)
                duration = clip.duration
                frame_times = [duration * (j + 1) / 17 for j in range(16)]  # 16 frames equally spaced

                frames = []
                for t in frame_times:
                    frame = clip.get_frame(t)
                    frame_image = Image.fromarray(frame)
                    frame_cropped = crop_center_square(frame_image)
                    frames.append(frame_cropped)

                grid_image = create_image_grid(frames)
                grid_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_grid.jpg")
                grid_image.save(grid_image_path)
                #print(f"Processed and saved {grid_image_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


save_folder_train = 'videos'
output_folder_resized_train = 'images'

process_videos_in_folder(save_folder_train, output_folder_resized_train)