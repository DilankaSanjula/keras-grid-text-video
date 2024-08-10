import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained ESRGAN model
# Replace 'esrgan_model_path' with the path to your pre-trained model
model = tf.keras.models.load_model('esrgan_model_path')

def preprocess_image(image_path):
    """
    Load and preprocess the image.
    """
    image = Image.open(image_path)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_image(image):
    """
    Postprocess the image to convert it back to [0, 255] and remove batch dimension.
    """
    image = image[0]  # Remove batch dimension
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)

def enhance_resolution(image_path, output_path):
    """
    Enhance the resolution of the image using ESRGAN.
    """
    # Preprocess the input image
    low_res_image = preprocess_image(image_path)

    # Predict the high-resolution image
    high_res_image = model.predict(low_res_image)

    # Postprocess and save the high-resolution image
    high_res_image = postprocess_image(high_res_image)
    high_res_image.save(output_path)
    print(f'Saved enhanced image to {output_path}')

# Paths to input and output images
input_image_path = 'path_to_low_res_image.jpg'
output_image_path = 'path_to_high_res_image.png'

# Enhance the resolution of the input image
enhance_resolution(input_image_path, output_image_path)
