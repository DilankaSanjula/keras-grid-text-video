# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow import keras

from keras_cv.models.stable_diffusion.__internal__.layers.attention_block import (  # noqa: E501
    AttentionBlock,
)
from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)
from keras_cv.models.stable_diffusion.__internal__.layers.resnet_block import (
    ResnetBlock,
)


class Decoder(keras.Sequential):
    def __init__(self, img_height=512, img_width=512):
        super().__init__(
            [
                # Input layer
                keras.layers.Input(shape=(img_height // 8, img_width // 8, 4)),

                # Scale latent space back to full range
                keras.layers.Rescaling(1.0 / 0.18215),
                PaddedConv2D(8, 1),  # Adjust input channels (64, 64, 8)
                PaddedConv2D(512, 3, padding=1),  # Increase features

                # Resnet blocks and attention
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),

                # Upsample to (128, 128, 512)
                keras.layers.Conv2DTranspose(512, 3, strides=2, padding="same"),
                ResnetBlock(512),
                ResnetBlock(512),

                # Upsample to (256, 256, 256)
                keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same"),
                ResnetBlock(256),
                ResnetBlock(256),

                # Upsample to (512, 512, 128)
                keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same"),
                ResnetBlock(128),
                ResnetBlock(128),

                # Output reconstruction
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),  # Final output to RGB (512, 512, 3)
            ]
        )