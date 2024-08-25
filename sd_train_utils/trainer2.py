from textwrap import wrap
import os
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

class Trainer(tf.keras.Model):
    def __init__(
        self,
        diffusion_model,
        noise_scheduler,
        use_mixed_precision=True,
        max_grad_norm=1.0,
        num_final_layers_to_train=20,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision

        # Apply freezing strategy if needed
        if num_final_layers_to_train is not None:
            self.freeze_layers(num_final_layers_to_train)

    def freeze_layers(self, num_final_layers_to_train):
        """Freeze all layers except the last `num_final_layers_to_train` layers in the diffusion model."""
        total_layers = len(self.diffusion_model.layers)
        layers_to_unfreeze = min(num_final_layers_to_train, total_layers)

        # Freeze all layers initially
        for layer in self.diffusion_model.layers:
            layer.trainable = False

        # Unfreeze the last `layers_to_unfreeze` layers
        for layer in self.diffusion_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

        print(f"Unfroze the last {layers_to_unfreeze} out of {total_layers} layers in the diffusion model.")

    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            # Add noise directly to the images and compute the noisy images
            noise = tf.random.normal(tf.shape(images))

            # Sample a random timestep for each image
            timesteps = tnp.random.randint(
                0, self.noise_scheduler.train_timesteps, (batch_size,)
            )

            # Forward pass through diffusion model
            noisy_images = self.noise_scheduler.add_noise(
                tf.cast(images, noise.dtype), noise, timesteps
            )
            timestep_embedding = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            timestep_embedding = tf.squeeze(timestep_embedding, 1)
            model_pred = self.diffusion_model(
                [noisy_images, timestep_embedding, encoded_text], training=True
            )
            loss = self.compiled_loss(noise, model_pred)
            if self.use_mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradients only for the diffusion model
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients if g is not None]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return the loss and any other metrics you've defined
        return {m.name: m.result() for m in self.metrics}

    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        half = dim // 2
        log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(
            -log_max_period * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Save the diffusion model's weights
        diffusion_model_filepath = filepath + "_diffusion_model.h5"
        self.diffusion_model.save_weights(
            filepath=diffusion_model_filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        print(f"Diffusion model weights saved to {diffusion_model_filepath}")