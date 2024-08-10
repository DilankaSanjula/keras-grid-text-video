import os
import cv2
import numpy as np
from PIL import Image

input_image_path = 'images/test2_grid.jpg'
output_video = 'output_video_3.mp4'
output_dir = 'test'

sub_image_size = (128, 128)

video_duration = 4  # seconds
new_size = (512, 512)
frame_size = (128,128)
fps = 4  # frames per second

input_image = Image.open(input_image_path)
input_image = input_image.resize(new_size)

sub_images = []
image_index = 1
for y in range(0, 512, 128):
    for x in range(0, 512, 128):
        box = (x, y, x + 128, y + 128)
        sub_image = input_image.crop(box)
        sub_images.append(sub_image)
        image_index += 1


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

for frame in sub_images:
    # Convert PIL image to numpy array
    frame_np = np.array(frame)
    # Convert RGB to BGR for OpenCV
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_np)

video_writer.release()
