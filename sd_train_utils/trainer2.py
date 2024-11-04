from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


class Trainer(tf.keras.Model):
    def __init__(
        self,
        diffusion_model,
        vae,
        noise_scheduler,
        use_mixed_precision=True,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.diffusion_model = diffusion_model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        self.vae.trainable = False  # Ensure VAE is trainable
        # No layer freezing - train all layers
        self.unfreeze_all_layers()

    def unfreeze_all_layers(self):
        """Ensure all layers of the diffusion model are trainable."""
        for layer in self.diffusion_model.layers:
            layer.trainable = True
        print("All layers in the diffusion model are unfrozen and trainable.")


    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        batch_size = tf.shape(images)[0]
        RESOLUTION = 512
        # Reshape 4x4 grid images into individual images
        images = tf.reshape(images, [batch_size * 16, RESOLUTION // 4, RESOLUTION // 4, 3])

        with tf.GradientTape() as tape:
            # Forward pass through VAE
            latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
            latents = latents * 0.18215

            # Resize latents to match diffusion model’s input shape
            latents = tf.image.resize(latents, [64, 64])  # Resizing from (16, 16) to (64, 64)

            # Add noise to the latents and compute the noisy latents
            noise = tf.random.normal(tf.shape(latents))

            # Sample a random timestep for each image in the batch
            timesteps = tnp.random.randint(
                0, self.noise_scheduler.train_timesteps, (batch_size * 16,)
            )

            # Apply noise to latents (same noise for all 16 sub-images)
            noisy_latents = self.noise_scheduler.add_noise(tf.cast(latents, noise.dtype), noise, timesteps)

            # Generate timestep embeddings
            timestep_embedding = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            timestep_embedding = tf.squeeze(timestep_embedding, 1)

            # Forward pass through the diffusion model
            model_pred = self.diffusion_model(
                [noisy_latents, timestep_embedding, encoded_text], training=True
            )
            
            # Calculate MSE loss for each individual image in the grid and then average
            losses = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            individual_losses = losses(noise, model_pred)  # Calculate loss per image
            grid_loss = tf.reduce_mean(individual_losses)  # Average loss across the grid images

            if self.use_mixed_precision:
                grid_loss = self.optimizer.get_scaled_loss(grid_loss)

        # Compute gradients only for the diffusion model (VAE is not trainable)
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(grid_loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": grid_loss}


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

    def sample_from_encoder_outputs(self, outputs):
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Save the diffusion model's weights
        diffusion_model_filepath = filepath + "_2x2_diffusion_model.h5"
        self.diffusion_model.save_weights(
            filepath=diffusion_model_filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        print(f"Diffusion model weights saved to {diffusion_model_filepath}")

        # # Save the VAE's weights
        # vae_filepath = filepath + "_vae_both.h5"
        # self.vae.save_weights(
        #     filepath=vae_filepath,
        #     overwrite=overwrite,
        #     save_format=save_format,
        #     options=options,
        # )
        # print(f"VAE model weights saved to {vae_filepath}")
