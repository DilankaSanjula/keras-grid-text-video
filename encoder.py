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


class ImageEncoder(keras.Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""

    def __init__(self):
        super().__init__(
            [
                # Input layer
                keras.layers.Input(shape=(None, None, 3)),

                # Initial feature extraction
                PaddedConv2D(128, 3, padding=1, strides=1),  # (512, 512, 128)
                ResnetBlock(128),
                ResnetBlock(128),

                # Downsample to (256, 256, 128)
                PaddedConv2D(128, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(256),
                ResnetBlock(256),

                # Downsample to (128, 128, 256)
                PaddedConv2D(256, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),

                # Downsample to (64, 64, 512)
                PaddedConv2D(512, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),

                # Add attention for better feature encoding
                AttentionBlock(512),
                ResnetBlock(512),

                # Bottleneck preparation
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(8, 3, padding=1),  # Reduce channels to 8
                PaddedConv2D(8, 1),  # Keep dimensions the same

                # Final bottleneck adjustment to (64, 64, 4)
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )
