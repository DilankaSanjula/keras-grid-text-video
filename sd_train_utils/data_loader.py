import os
import numpy as np
from tqdm import tqdm

import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras
from sd_train_utils.trainer import Trainer



def create_dataframe(directory, extensions=['.jpg']):
    """
    Get all image paths from a given directory, including subdirectories, and their captions.

    Parameters:
    directory (str): The directory to search for images.
    extensions (list): List of image file extensions to look for.

    Returns:
    pd.DataFrame: DataFrame with columns 'image_path' and 'caption'.
    """
    image_paths = []
    captions = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_name, _ = os.path.splitext(file) 
                caption = file_name.replace('_', ' ') 

                image_path = os.path.join(root, file)
                # Normalize the path to use the appropriate delimiter
                image_path = os.path.normpath(image_path)

                image_paths.append(image_path)
                captions.append(caption)

    data_frame = pd.DataFrame({'image_path': image_paths, 'caption': captions})
    return data_frame