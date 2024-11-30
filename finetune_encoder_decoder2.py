import numpy as np
import tensorflow as tf
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.decoder import Decoder
from sd_train_utils.data_loader import create_dataframe
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16

# Set up VGG16 for perceptual loss
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
vgg.trainable = False

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Constants
RESOLUTION = 512
directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/mixed_dataset'

# Load and preprocess images
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [RESOLUTION, RESOLUTION])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return image

def prepare_dataset(image_paths, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

data_frame = create_dataframe(directory)
image_paths = np.array(data_frame["image_path"])
train_dataset = prepare_dataset(image_paths, batch_size=2)

# Initialize the Encoder and Decoder
encoder = ImageEncoder(download_weights=True)
decoder = Decoder(img_height=RESOLUTION, img_width=RESOLUTION, download_weights=True)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        latents = self.encoder(inputs)
        reconstructed = self.decoder(latents)
        return reconstructed

# Define Loss
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    true_features = vgg(y_true)
    pred_features = vgg(y_pred)
    perceptual = tf.reduce_mean(tf.abs(true_features - pred_features))
    ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mse + 0.7 * perceptual + 0.2 * ssim

vae_model = VAE(encoder=encoder, decoder=decoder)
vae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=combined_loss)

class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, encoder_path, decoder_path, monitor="loss", verbose=1):
        super(SaveModelCallback, self).__init__()
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.monitor = monitor
        self.best_loss = np.inf
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.model.encoder.save_weights(self.encoder_path)
            self.model.decoder.save_weights(self.decoder_path)
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} improved to {current_loss:.5f}, saving model to {self.encoder_path} and {self.decoder_path}")

# Train the model
vae_model.fit(
    train_dataset,
    epochs=1000,
    callbacks=[SaveModelCallback("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_encoder.h5",
                                 "/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_decoder.h5")]
)

# Optionally, save the final weights
vae_model.encoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_encoder.h5")
vae_model.decoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_decoder.h5")
