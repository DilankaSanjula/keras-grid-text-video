import numpy as np
import tensorflow as tf
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.decoder import Decoder
from sd_train_utils.data_loader import create_dataframe
from sklearn.model_selection import train_test_split

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth could not be set: {e}")

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512

# Paths
directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_4x4_images'

# Create the dataframe from image directory
data_frame = create_dataframe(directory)
image_paths = np.array(data_frame["image_path"])

# Split the data into training and validation sets
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Load and preprocess images
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [RESOLUTION, RESOLUTION])
    image = (image / 127.5) - 1.0
    return image

def prepare_grid_dataset(image_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (x, x))  # Provide input as both x and y
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = prepare_grid_dataset(train_paths, batch_size=4)
val_dataset = prepare_grid_dataset(val_paths, batch_size=4)

# Initialize the Encoder and Decoder
encoder = ImageEncoder(download_weights=False)
decoder = Decoder(img_height=RESOLUTION, img_width=RESOLUTION, download_weights=False)

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        latents = self.encoder(inputs)
        reconstructed = self.decoder(latents)
        return reconstructed

# Define Loss and Model
reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

def vae_loss(y_true, y_pred):
    return reconstruction_loss_fn(y_true, y_pred)

vae_model = VAE(encoder=encoder, decoder=decoder)
vae_model.compile(optimizer='adam', loss=vae_loss)

# Define callbacks
encoder_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_encoder.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

decoder_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_decoder.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=40,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# Load the best weights for encoder and decoder
vae_model.encoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_encoder.h5")
vae_model.decoder.load_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_decoder.h5")


# Set the learning rate to the last reduced value
vae_model.optimizer.lr.assign(6.25e-05)

# Train the model
vae_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1500,
    callbacks=[encoder_checkpoint, decoder_checkpoint, early_stopping, reduce_lr]
)

# Save the final weights as well
vae_model.encoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_encoder.h5")
vae_model.decoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_decoder.h5")
