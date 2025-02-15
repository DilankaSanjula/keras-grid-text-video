import os
import tensorflow as tf
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.decoder import Decoder

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
USE_MP = True

# Define a function to print the model details
def print_model_details(model, model_name):
    model.summary()
    print(f"Total Parameters for {model_name}: {model.count_params()}\n")

# Load or initialize models
image_encoder = ImageEncoder(download_weights=False)  # Assuming the encoder weights are managed separately

diffusion_model = DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH, download_weights=False)
# Print details of the image encoder
print_model_details(image_encoder, "Image Encoder")

# Print details of the diffusion model
print_model_details(diffusion_model, "Diffusion Model")