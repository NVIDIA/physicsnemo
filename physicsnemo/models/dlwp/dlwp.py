# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.layers import get_activation
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

Tensor = torch.Tensor


def _get_same_padding(x: int, k: int, s: int) -> int:
    """Function to compute "same" padding. Inspired from:
    https://github.com/huggingface/pytorch-image-models/blob/0.5.x/timm/models/layers/padding.py
    """
    return max(s * math.ceil(x / s) - s - x + k, 0)


def _pad_periodically_equatorial(
    main_face, left_face, right_face, top_face, bottom_face, nr_rot, size=2
):
    if nr_rot != 0:
        top_face = torch.rot90(top_face, k=nr_rot, dims=(-2, -1))
        bottom_face = torch.rot90(bottom_face, k=nr_rot, dims=(-1, -2))
    padded_data_temp = torch.cat(
        (left_face[..., :, -size:], main_face, right_face[..., :, :size]), dim=-1
    )
    top_pad = torch.cat(
        (top_face[..., :, :size], top_face, top_face[..., :, -size:]), dim=-1
    )  # hacky - extend on the left and right side
    bottom_pad = torch.cat(
        (bottom_face[..., :, :size], bottom_face, bottom_face[..., :, -size:]), dim=-1
    )  # hacky - extend on the left and right side
    padded_data = torch.cat(
        (bottom_pad[..., -size:, :], padded_data_temp, top_pad[..., :size, :]), dim=-2
    )
    return padded_data


def _pad_periodically_polar(
    main_face,
    left_face,
    right_face,
    top_face,
    bottom_face,
    rot_axis_left,
    rot_axis_right,
    size=2,
):
    left_face = torch.rot90(left_face, dims=rot_axis_left)
    right_face = torch.rot90(right_face, dims=rot_axis_right)
    padded_data_temp = torch.cat(
        (bottom_face[..., -size:, :], main_face, top_face[..., :size, :]), dim=-2
    )
    left_pad = torch.cat(
        (left_face[..., :size, :], left_face, left_face[..., -size:, :]), dim=-2
    )  # hacky - extend the left and right
    right_pad = torch.cat(
        (right_face[..., :size, :], right_face, right_face[..., -size:, :]), dim=-2
    )  # hacky - extend the left and right
    padded_data = torch.cat(
        (left_pad[..., :, -size:], padded_data_temp, right_pad[..., :, :size]), dim=-1
    )
    return padded_data


def _cubed_conv_wrapper(faces, equator_conv, polar_conv):
    # compute the required padding
    padding_size = _get_same_padding(
        x=faces[0].size(-1), k=equator_conv.kernel_size[0], s=equator_conv.stride[0]
    )
    padding_size = padding_size // 2
    output = []
    if padding_size != 0:
        for i in range(6):
            if i == 0:
                x = _pad_periodically_equatorial(
                    faces[0],
                    faces[3],
                    faces[1],
                    faces[5],
                    faces[4],
                    nr_rot=0,
                    size=padding_size,
                )
                output.append(equator_conv(x))
            elif i == 1:
                x = _pad_periodically_equatorial(
                    faces[1],
                    faces[0],
                    faces[2],
                    faces[5],
                    faces[4],
                    nr_rot=1,
                    size=padding_size,
                )
                output.append(equator_conv(x))
            elif i == 2:
                x = _pad_periodically_equatorial(
                    faces[2],
                    faces[1],
                    faces[3],
                    faces[5],
                    faces[4],
                    nr_rot=2,
                    size=padding_size,
                )
                output.append(equator_conv(x))
            elif i == 3:
                x = _pad_periodically_equatorial(
                    faces[3],
                    faces[2],
                    faces[0],
                    faces[5],
                    faces[4],
                    nr_rot=3,
                    size=padding_size,
                )
                output.append(equator_conv(x))
            elif i == 4:
                x = _pad_periodically_polar(
                    faces[4],
                    faces[3],
                    faces[1],
                    faces[0],
                    faces[5],
                    rot_axis_left=(-1, -2),
                    rot_axis_right=(-2, -1),
                    size=padding_size,
                )
                output.append(polar_conv(x))
            else:  # i=5
                x = _pad_periodically_polar(
                    faces[5],
                    faces[3],
                    faces[1],
                    faces[4],
                    faces[0],
                    rot_axis_left=(-2, -1),
                    rot_axis_right=(-1, -2),
                    size=padding_size,
                )
                x = torch.flip(x, [-1])
                x = polar_conv(x)
                output.append(torch.flip(x, [-1]))
    else:
        for i in range(6):
            if i in [0, 1, 2, 3]:
                output.append(equator_conv(faces[i]))
            elif i == 4:
                output.append(polar_conv(faces[i]))
            else:  # i=5
                x = torch.flip(faces[i], [-1])
                x = polar_conv(x)
                output.append(torch.flip(x, [-1]))

    return output


def _cubed_non_conv_wrapper(faces, layer):
    output = [layer(faces[i]) for i in range(6)]
    return output


@dataclass
class MetaData(ModelMetaData):
    name: str = "DLWP"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class DLWP(Module):
    """A Convolutional model for Deep Learning Weather Prediction that
    works on Cubed-sphere grids.

    This model expects the input to be of shape [N, C, 6, Res, Res]

    Parameters
    ----------
    nr_input_channels : int
        Number of channels in the input
    nr_output_channels : int
        Number of channels in the output
    nr_initial_channels : int
        Number of channels in the initial convolution. This governs the overall channels
        in the model.
    activation_fn : str
        Activation function for the convolutions
    depth : int
        Depth for the U-Net
    clamp_activation : Tuple of ints, floats or None
        The min and max value used for torch.clamp()

    Example
    -------
    >>> model = physicsnemo.models.dlwp.DLWP(
    ... nr_input_channels=2,
    ... nr_output_channels=4,
    ... )
    >>> input = torch.randn(4, 2, 6, 64, 64) # [N, C, F, Res, Res]
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 4, 6, 64, 64])

    Note
    ----
    Reference: Weyn, Jonathan A., et al. "Sub‐seasonal forecasting with a large ensemble
     of deep‐learning weather prediction models." Journal of Advances in Modeling Earth
     Systems 13.7 (2021): e2021MS002502.
    """

    def __init__(
        self,
        nr_input_channels: int,
        nr_output_channels: int,
        nr_initial_channels: int = 64,
        activation_fn: str = "leaky_relu",
        depth: int = 2,
        clamp_activation: Tuple[Union[float, int, None], Union[float, int, None]] = (
            None,
            10.0,
        ),
    ):
        super().__init__(meta=MetaData())

        self.nr_input_channels = nr_input_channels
        self.nr_output_channels = nr_output_channels
        self.nr_initial_channels = nr_initial_channels
        self.activation_fn = get_activation(activation_fn)
        self.depth = depth
        self.clamp_activation = clamp_activation

        # define layers
        # define non-convolutional layers
        self.avg_pool = nn.AvgPool2d(2)
        self.upsample_layer = nn.Upsample(scale_factor=2)

        # define layers
        self.equatorial_downsample = []
        self.equatorial_upsample = []
        self.equatorial_mid_layers = []
        self.polar_downsample = []
        self.polar_upsample = []
        self.polar_mid_layers = []

        for i in range(depth):
            if i == 0:
                ins = self.nr_input_channels
            else:
                ins = self.nr_initial_channels * (2 ** (i - 1))
            outs = self.nr_initial_channels * (2 ** (i))
            self.equatorial_downsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.polar_downsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.equatorial_downsample.append(nn.Conv2d(outs, outs, kernel_size=3))
            self.polar_downsample.append(nn.Conv2d(outs, outs, kernel_size=3))

        for i in range(2):
            if i == 0:
                ins = outs
                outs = ins * 2
            else:
                ins = outs
                outs = ins // 2
            self.equatorial_mid_layers.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.polar_mid_layers.append(nn.Conv2d(ins, outs, kernel_size=3))

        for i in range(depth - 1, -1, -1):
            if i == 0:
                outs = self.nr_initial_channels
                outs_final = outs
            else:
                outs = self.nr_initial_channels * (2 ** (i))
                outs_final = outs // 2
            ins = outs * 2
            self.equatorial_upsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.polar_upsample.append(nn.Conv2d(ins, outs, kernel_size=3))
            self.equatorial_upsample.append(nn.Conv2d(outs, outs_final, kernel_size=3))
            self.polar_upsample.append(nn.Conv2d(outs, outs_final, kernel_size=3))

        self.equatorial_downsample = nn.ModuleList(self.equatorial_downsample)
        self.polar_downsample = nn.ModuleList(self.polar_downsample)
        self.equatorial_mid_layers = nn.ModuleList(self.equatorial_mid_layers)
        self.polar_mid_layers = nn.ModuleList(self.polar_mid_layers)
        self.equatorial_upsample = nn.ModuleList(self.equatorial_upsample)
        self.polar_upsample = nn.ModuleList(self.polar_upsample)

        self.equatorial_last = nn.Conv2d(outs, self.nr_output_channels, kernel_size=1)
        self.polar_last = nn.Conv2d(outs, self.nr_output_channels, kernel_size=1)

    # define activation layers
    def activation(self, x: Tensor):
        x = self.activation_fn(x)
        if any(isinstance(c, (float, int)) for c in self.clamp_activation):
            x = torch.clamp(
                x, min=self.clamp_activation[0], max=self.clamp_activation[1]
            )
        return x

    def forward(self, cubed_sphere_input):
        # do some input checks
        if cubed_sphere_input.size(-3) != 6:
            raise ValueError("The input must have 6 faces.")
        if cubed_sphere_input.size(-2) != cubed_sphere_input.size(-1):
            raise ValueError("The input must have equal height and width")

        # split the cubed_sphere_input into individual faces
        faces = torch.split(
            cubed_sphere_input, split_size_or_sections=1, dim=2
        )  # split along face dim

        faces = [torch.squeeze(face, dim=2) for face in faces]

        encoder_states = []

        for i, (equatorial_layer, polar_layer) in enumerate(
            zip(self.equatorial_downsample, self.polar_downsample)
        ):
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)
            if i % 2 != 0:
                encoder_states.append(faces)
                faces = _cubed_non_conv_wrapper(faces, self.avg_pool)

        for i, (equatorial_layer, polar_layer) in enumerate(
            zip(self.equatorial_mid_layers, self.polar_mid_layers)
        ):
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)

        j = 0
        for i, (equatorial_layer, polar_layer) in enumerate(
            zip(self.equatorial_upsample, self.polar_upsample)
        ):
            if i % 2 == 0:
                encoder_faces = encoder_states[len(encoder_states) - j - 1]
                faces = _cubed_non_conv_wrapper(faces, self.upsample_layer)
                faces = [
                    torch.cat((face_1, face_2), dim=1)
                    for face_1, face_2 in zip(faces, encoder_faces)
                ]
                j += 1
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)

        faces = _cubed_conv_wrapper(faces, self.equatorial_last, self.polar_last)
        output = torch.stack(faces, dim=2)

        return output
