import os
import cv2
from moviepy.editor import VideoFileClip

def process_frame(frame, target_size=(512, 512)):
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Resize the frame using OpenCV
    resized_frame = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
    # Convert back to RGB to maintain consistency with moviepy
    return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

def crop_center_square_and_resize(clip, target_size=(512, 512)):
    width, height = clip.size
    min_dim = min(width, height)
    crop_x = (width - min_dim) // 2
    crop_y = (height - min_dim) // 2
    cropped_clip = clip.crop(x1=crop_x, y1=crop_y, width=min_dim, height=min_dim)
    
    # Apply resizing to each frame using the process_frame function
    processed_clip = cropped_clip.fl_image(lambda frame: process_frame(frame, target_size))
    return processed_clip

def process_videos_in_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        try:
            if filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, filename)
                clip = VideoFileClip(video_path)
                
                # Limit the video to the first 4 seconds
                clip = clip.subclip(0, 4)

                # Resize and crop as necessary
                clip_processed = crop_center_square_and_resize(clip)

                # Set the output frame rate to 4 fps to ensure exactly 16 frames over 4 seconds
                clip_processed = clip_processed.set_fps(4)

                resized_video_path = os.path.join(output_folder, filename)
                clip_processed.write_videofile(resized_video_path, codec='libx264', audio_codec='aac')
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage
folder_path = '/content/drive/MyDrive/webvid-10-dataset-2/raw_videos'
output_folder = '/content/drive/MyDrive/webvid-10-dataset-2/resized_videos'
process_videos_in_folder(folder_path, output_folder)  