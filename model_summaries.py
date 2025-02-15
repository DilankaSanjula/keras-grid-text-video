import os
import tensorflow as tf
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.decoder import Decoder

# Define a function to print the model details
def print_model_details(model, model_name):
    model.summary()
    print(f"Total Parameters for {model_name}: {model.count_params()}\n")

# Load or initialize models
image_encoder = ImageEncoder(download_weights=False)  # Assuming the encoder weights are managed separately
diffusion_model = DiffusionModel(resolution=512, channels=3, num_text_tokens=77, download_weights=False)  # Set to False to not download weights

# Print details of the image encoder
print_model_details(image_encoder, "Image Encoder")

# Print details of the diffusion model
print_model_details(diffusion_model, "Diffusion Model")