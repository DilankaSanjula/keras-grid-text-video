from tensorflow import keras
from keras_cv.models.stable_diffusion.__internal__.layers.attention_block import AttentionBlock
from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import PaddedConv2D
from keras_cv.models.stable_diffusion.__internal__.layers.resnet_block import ResnetBlock

class ImageEncoder(keras.Model):
    """Improved VAE Encoder for Stable Diffusion with skip connections and attention."""

    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = PaddedConv2D(128, 3, padding=1, strides=1)
        self.resblock1 = ResnetBlock(128)
        self.resblock2 = ResnetBlock(128)

        # Downsample to (256, 256, 128)
        self.downsample1 = PaddedConv2D(128, 3, padding=((0, 1), (0, 1)), strides=2)
        self.resblock3 = ResnetBlock(256)
        self.resblock4 = ResnetBlock(256)

        # Downsample to (128, 128, 256)
        self.downsample2 = PaddedConv2D(256, 3, padding=((0, 1), (0, 1)), strides=2)
        self.resblock5 = ResnetBlock(512)
        self.resblock6 = ResnetBlock(512)

        # Downsample to (64, 64, 512)
        self.downsample3 = PaddedConv2D(512, 3, padding=((0, 1), (0, 1)), strides=2)
        self.resblock7 = ResnetBlock(512)
        self.resblock8 = ResnetBlock(512)

        # Attention block and bottleneck preparation
        self.attention1 = AttentionBlock(512)
        self.resblock9 = ResnetBlock(512)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.activation = keras.layers.Activation("swish")

        # Bottleneck adjustment to (64, 64, 4)
        self.bottleneck_conv1 = PaddedConv2D(8, 3, padding=1)
        self.bottleneck_conv2 = PaddedConv2D(8, 1)
        self.bottleneck_adjustment = keras.layers.Lambda(lambda x: x[..., :4] * 0.18215)

    def call(self, inputs):
        # Initial feature extraction
        x1 = self.initial_conv(inputs)
        x1 = self.resblock1(x1)
        x1 = self.resblock2(x1)

        # Downsample to (256, 256, 128)
        x2 = self.downsample1(x1)
        x2 = self.resblock3(x2)
        x2 = self.resblock4(x2)

        # Downsample to (128, 128, 256)
        x3 = self.downsample2(x2)
        x3 = self.resblock5(x3)
        x3 = self.resblock6(x3)

        # Downsample to (64, 64, 512)
        x4 = self.downsample3(x3)
        x4 = self.resblock7(x4)
        x4 = self.resblock8(x4)

        # Add attention block
        x4 = self.attention1(x4)
        x4 = self.resblock9(x4)

        # Downsample x3 to match x4 dimensions
        x3_downsampled = keras.layers.Conv2D(512, 3, strides=2, padding="same")(x3)  # (64, 64, 512)
        x4 = keras.layers.Concatenate()([x4, x3_downsampled])

        # Downsample x2 to match x4 dimensions
        x2_downsampled = keras.layers.Conv2D(512, 3, strides=4, padding="same")(x2)  # (64, 64, 512)
        x4 = keras.layers.Concatenate()([x4, x2_downsampled])

        # Bottleneck preparation
        x4 = self.norm(x4)
        x4 = self.activation(x4)
        x4 = self.bottleneck_conv1(x4)
        x4 = self.bottleneck_conv2(x4)

        # Final bottleneck adjustment
        return self.bottleneck_adjustment(x4)