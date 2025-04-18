from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


tf.config.run_functions_eagerly(True)

class Trainer(tf.keras.Model):
    def __init__(
        self,
        diffusion_model,
        vae,
        noise_scheduler,
        use_mixed_precision=True,
        max_grad_norm=1.0,
        loss_threshold=None,  # Add a parameter for dynamic loss threshold
        **kwargs
    ):
        super().__init__(**kwargs)
        self.diffusion_model = diffusion_model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        self.loss_threshold = loss_threshold
        self.vae.trainable = False
        self.unfreeze_all_layers()

    def unfreeze_all_layers(self):
        """Ensure all layers of the diffusion model are trainable."""
        for layer in self.diffusion_model.layers:
            layer.trainable = True
        print("All layers in the diffusion model are unfrozen and trainable.")

    def log_image_paths_and_losses(image_paths, individual_losses):
        for i in range(len(image_paths)):
            image_path = image_paths[i].numpy().decode('utf-8')
            individual_loss = individual_losses[i].numpy()
            print(f'Image: {image_path}, Loss: {individual_loss}')

    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        image_paths = inputs["image_paths"]  # Access image paths
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            # Forward pass through VAE
            latents = self.sample_from_encoder_outputs(self.vae(images, training=True))
            latents = latents * 0.18215

            # Add noise to the latents
            noise = tf.random.normal(tf.shape(latents))

            # Sample a random timestep for each image
            timesteps = tf.random.uniform(
                [batch_size], minval=0, maxval=self.noise_scheduler.train_timesteps, dtype=tf.int32
            )

            # Add noise to the latents
            noisy_latents = self.noise_scheduler.add_noise(
                tf.cast(latents, noise.dtype), noise, timesteps
            )

            # Generate timestep embeddings
            timestep_embeddings = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            timestep_embeddings = tf.squeeze(timestep_embeddings, 1)

            # Forward pass through the diffusion model
            model_pred = self.diffusion_model(
                [noisy_latents, timestep_embeddings, encoded_text], training=True
            )

            # Calculate individual losses for each sample
            mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            individual_losses = mse_loss(noise, model_pred)

            # Compute the mean loss for backpropagation
            loss = tf.reduce_mean(individual_losses)
            if self.use_mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradients and apply them
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Log image paths and their corresponding losses after processing the batch
        image_paths_list = tf.unstack(image_paths)
        individual_losses_list = tf.unstack(individual_losses)

        for image_path, individual_loss in zip(image_paths_list, individual_losses_list):
            tf.print('Image:', image_path, ', Loss:', individual_loss)

        return {"loss": loss}

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
