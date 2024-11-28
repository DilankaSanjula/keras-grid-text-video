import numpy as np
import tensorflow as tf
from encoder import ImageEncoder
from decoder import Decoder
from sd_train_utils.data_loader import create_dataframe
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16

# Constants
MAX_PROMPT_LENGTH = 77
RESOLUTION = 512
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
EPOCHS = 1000

# Initialize VGG16 for perceptual loss
vgg = VGG16(include_top=False, weights="imagenet", input_shape=(512, 512, 3))
vgg.trainable = False

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth could not be set: {e}")

# Paths
directory = '/content/drive/MyDrive/stable_diffusion_4x4/dataset/homer_simpson_4x4_images'

# Create the dataframe from image directory
data_frame = create_dataframe(directory)
image_paths = np.array(data_frame["image_path"])

# Split the data into training and validation sets
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Ensure no overlap between training and validation datasets
assert not any(path in train_paths for path in val_paths), "Overlap detected between train and validation datasets"

# Load and preprocess images
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [RESOLUTION, RESOLUTION])
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return image

def prepare_grid_dataset(image_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (x, x))  # Provide input as both x and y
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = prepare_grid_dataset(train_paths, batch_size=BATCH_SIZE)
val_dataset = prepare_grid_dataset(val_paths, batch_size=BATCH_SIZE)

# Initialize Encoder and Decoder
encoder = ImageEncoder()
decoder = Decoder(img_height=RESOLUTION, img_width=RESOLUTION)

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        latents = self.encoder(inputs)
        reconstructed = self.decoder(latents)
        return reconstructed

vae_model = VAE(encoder=encoder, decoder=decoder)

# Define Loss Functions
def perceptual_loss(y_true, y_pred):
    true_features = vgg(y_true)
    pred_features = vgg(y_pred)
    return tf.reduce_mean(tf.abs(true_features - pred_features))

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    perceptual = perceptual_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    return mse + 0.5 * perceptual + 0.3 * ssim

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Initialize accumulated gradients once
accumulated_gradients = [tf.zeros_like(var) for var in vae_model.trainable_variables]

# Gradient Accumulation
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = vae_model(inputs, training=True)
        loss = combined_loss(targets, predictions)
    gradients = tape.gradient(loss, vae_model.trainable_variables)
    return loss, gradients

@tf.function
def apply_gradients(accumulated_gradients):
    optimizer.apply_gradients(zip(accumulated_gradients, vae_model.trainable_variables))

# Callbacks for Saving Weights
best_val_loss = float("inf")  # Initialize best validation loss

def save_best_weights(epoch, val_loss):
    global best_val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        vae_model.encoder.save_weights(
            f"/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_encoder.h5"
        )
        vae_model.decoder.save_weights(
            f"/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/best_vae_decoder.h5"
        )
        print(f"Epoch {epoch + 1}: Saved best weights with val_loss = {val_loss:.5f}")

# Training Loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss = 0
    train_steps = 0

    # Reset accumulated gradients to zero at the start of the epoch
    for i in range(len(accumulated_gradients)):
        accumulated_gradients[i].assign(tf.zeros_like(accumulated_gradients[i]))

    for step, (inputs, targets) in enumerate(train_dataset):
        loss, gradients = train_step(inputs, targets)
        for i in range(len(accumulated_gradients)):
            accumulated_gradients[i].assign_add(gradients[i])

        train_loss += loss
        train_steps += 1

        if (step + 1) % ACCUMULATION_STEPS == 0:
            apply_gradients(accumulated_gradients)
            for i in range(len(accumulated_gradients)):
                accumulated_gradients[i].assign(tf.zeros_like(accumulated_gradients[i]))

    avg_train_loss = train_loss / train_steps

    # Validation
    val_loss = 0
    val_steps = 0
    for inputs, targets in val_dataset:
        predictions = vae_model(inputs, training=False)  # Ensure validation is done in inference mode
        val_loss += combined_loss(targets, predictions)
        val_steps += 1

    avg_val_loss = val_loss / val_steps
    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save weights
    save_best_weights(epoch, avg_val_loss)

    # Early stopping logic
    if avg_val_loss >= best_val_loss and epoch > 40:
        print("Early stopping triggered")
        break

# Save final weights
vae_model.encoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_encoder.h5")
vae_model.decoder.save_weights("/content/drive/MyDrive/stable_diffusion_4x4/decoder_encoder_training/final_vae_decoder.h5")
