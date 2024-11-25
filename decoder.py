from tensorflow import keras
from keras_cv.models.stable_diffusion.__internal__.layers.attention_block import AttentionBlock
from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import PaddedConv2D
from keras_cv.models.stable_diffusion.__internal__.layers.resnet_block import ResnetBlock


class Decoder(keras.Model):
    """Improved VAE Decoder for Stable Diffusion with skip connections and attention."""

    def __init__(self, img_height=512, img_width=512):
        super(Decoder, self).__init__()

        # Input processing
        self.rescale = keras.layers.Rescaling(1.0 / 0.18215)
        self.initial_conv = PaddedConv2D(8, 1)  # Adjust input channels (64, 64, 8)
        self.feature_expansion = PaddedConv2D(512, 3, padding=1)  # Increase features

        # Bottleneck processing
        self.bottleneck_resblock1 = ResnetBlock(512)
        self.bottleneck_attention = AttentionBlock(512)
        self.bottleneck_resblock2 = ResnetBlock(512)

        # Upsample to (128, 128, 512)
        self.upsample1 = keras.layers.Conv2DTranspose(512, 3, strides=2, padding="same")
        self.resblock1_1 = ResnetBlock(512)
        self.resblock1_2 = ResnetBlock(512)

        # Upsample to (256, 256, 256)
        self.upsample2 = keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same")
        self.resblock2_1 = ResnetBlock(256)
        self.resblock2_2 = ResnetBlock(256)

        # Upsample to (512, 512, 128)
        self.upsample3 = keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same")
        self.resblock3_1 = ResnetBlock(128)
        self.resblock3_2 = ResnetBlock(128)

        # Output reconstruction
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.activation = keras.layers.Activation("swish")
        self.output_conv = PaddedConv2D(3, 3, padding=1)  # Final output to RGB (512, 512, 3)

    def call(self, inputs, encoder_features=None):
        """
        Decoder forward pass with optional skip connections.
        Args:
            inputs: Latent tensor from the encoder (64, 64, 4).
            encoder_features: List of features from the encoder for skip connections.
        """
        # Process latent space
        x = self.rescale(inputs)
        x = self.initial_conv(x)  # (64, 64, 8)
        x = self.feature_expansion(x)  # (64, 64, 512)

        # Bottleneck processing
        x = self.bottleneck_resblock1(x)
        x = self.bottleneck_attention(x)
        x = self.bottleneck_resblock2(x)

        # Upsample to (128, 128, 512)
        x = self.upsample1(x)
        if encoder_features:
            x = keras.layers.Concatenate()([x, encoder_features[2]])  # Add skip connection
        x = self.resblock1_1(x)
        x = self.resblock1_2(x)

        # Upsample to (256, 256, 256)
        x = self.upsample2(x)
        if encoder_features:
            x = keras.layers.Concatenate()([x, encoder_features[1]])  # Add skip connection
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)

        # Upsample to (512, 512, 128)
        x = self.upsample3(x)
        if encoder_features:
            x = keras.layers.Concatenate()([x, encoder_features[0]])  # Add skip connection
        x = self.resblock3_1(x)
        x = self.resblock3_2(x)

        # Output reconstruction
        x = self.norm(x)
        x = self.activation(x)
        x = self.output_conv(x)  # Final output to RGB
        return x
