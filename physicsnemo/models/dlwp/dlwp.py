# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import get_activation


def _get_same_padding(x: int, k: int, s: int) -> int:
    r"""
    Compute the "same" padding size for a 1D convolution dimension.

    Parameters
    ----------
    x : int
        Input size.
    k : int
        Kernel size.
    s : int
        Stride size.

    Returns
    -------
    int
        Padding size for "same" output resolution.
    """
    return max(s * math.ceil(x / s) - s - x + k, 0)


def _pad_periodically_equatorial(
    main_face: Float[torch.Tensor, "batch channels height width"],
    left_face: Float[torch.Tensor, "batch channels height width"],
    right_face: Float[torch.Tensor, "batch channels height width"],
    top_face: Float[torch.Tensor, "batch channels height width"],
    bottom_face: Float[torch.Tensor, "batch channels height width"],
    nr_rot: int,
    size: int = 2,
) -> Float[torch.Tensor, "batch channels height_out width_out"]:
    r"""
    Periodically pad a cubed-sphere equatorial face using adjacent faces.

    Parameters
    ----------
    main_face : torch.Tensor
        Equatorial face tensor of shape :math:`(B, C, H, W)`.
    left_face : torch.Tensor
        Left neighbor face tensor of shape :math:`(B, C, H, W)`.
    right_face : torch.Tensor
        Right neighbor face tensor of shape :math:`(B, C, H, W)`.
    top_face : torch.Tensor
        Top neighbor face tensor of shape :math:`(B, C, H, W)`.
    bottom_face : torch.Tensor
        Bottom neighbor face tensor of shape :math:`(B, C, H, W)`.
    nr_rot : int
        Number of 90-degree rotations applied to the polar faces.
    size : int, optional, default=2
        Padding size applied along spatial dimensions.

    Returns
    -------
    torch.Tensor
        Padded face tensor of shape :math:`(B, C, H + 2p, W + 2p)`.
    """
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
    main_face: Float[torch.Tensor, "batch channels height width"],
    left_face: Float[torch.Tensor, "batch channels height width"],
    right_face: Float[torch.Tensor, "batch channels height width"],
    top_face: Float[torch.Tensor, "batch channels height width"],
    bottom_face: Float[torch.Tensor, "batch channels height width"],
    rot_axis_left: tuple[int, int],
    rot_axis_right: tuple[int, int],
    size: int = 2,
) -> Float[torch.Tensor, "batch channels height_out width_out"]:
    r"""
    Periodically pad a cubed-sphere polar face using adjacent faces.

    Parameters
    ----------
    main_face : torch.Tensor
        Polar face tensor of shape :math:`(B, C, H, W)`.
    left_face : torch.Tensor
        Left neighbor face tensor of shape :math:`(B, C, H, W)`.
    right_face : torch.Tensor
        Right neighbor face tensor of shape :math:`(B, C, H, W)`.
    top_face : torch.Tensor
        Top neighbor face tensor of shape :math:`(B, C, H, W)`.
    bottom_face : torch.Tensor
        Bottom neighbor face tensor of shape :math:`(B, C, H, W)`.
    rot_axis_left : tuple[int, int]
        Rotation axes for the left neighbor face.
    rot_axis_right : tuple[int, int]
        Rotation axes for the right neighbor face.
    size : int, optional, default=2
        Padding size applied along spatial dimensions.

    Returns
    -------
    torch.Tensor
        Padded face tensor of shape :math:`(B, C, H + 2p, W + 2p)`.
    """
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


def _cubed_conv_wrapper(
    faces: Sequence[Float[torch.Tensor, "batch channels height width"]],
    equator_conv: nn.Conv2d,
    polar_conv: nn.Conv2d,
) -> list[Float[torch.Tensor, "batch channels height_out width_out"]]:
    r"""
    Apply face-wise convolution with cubed-sphere padding.

    Parameters
    ----------
    faces : Sequence[torch.Tensor]
        Sequence of six faces, each of shape :math:`(B, C, H, W)`.
    equator_conv : torch.nn.Conv2d
        Convolution applied to equatorial faces (indices 0-3).
    polar_conv : torch.nn.Conv2d
        Convolution applied to polar faces (indices 4-5).

    Returns
    -------
    list[torch.Tensor]
        List of six convolved faces, each of shape :math:`(B, C', H', W')`.
    """
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


def _cubed_non_conv_wrapper(
    faces: Sequence[Float[torch.Tensor, "batch channels height width"]],
    layer: Callable[
        [Float[torch.Tensor, "batch channels height width"]],
        Float[torch.Tensor, "batch channels height width"],
    ],
) -> list[Float[torch.Tensor, "batch channels height width"]]:
    r"""
    Apply a non-convolutional layer to each cubed-sphere face.

    Parameters
    ----------
    faces : Sequence[torch.Tensor]
        Sequence of six faces, each of shape :math:`(B, C, H, W)`.
    layer : Callable[[torch.Tensor], torch.Tensor]
        Callable applied independently to each face tensor.

    Returns
    -------
    list[torch.Tensor]
        List of transformed faces, each of shape :math:`(B, C', H', W')`.
    """
    return [layer(face) for face in faces]


@dataclass
class MetaData(ModelMetaData):
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
    r"""
    Convolutional U-Net for Deep Learning Weather Prediction on cubed-sphere grids.

    This model operates on cubed-sphere data with six faces and applies face-aware
    padding so that convolutions respect cubed-sphere connectivity.

    Based on `Weyn et al. (2021) <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002502>`_.

    Parameters
    ----------
    nr_input_channels : int
        Number of input channels :math:`C_{in}`.
    nr_output_channels : int
        Number of output channels :math:`C_{out}`.
    nr_initial_channels : int, optional, default=64
        Number of channels in the first convolution block :math:`C_{init}`. Defaults to 64.
    activation_fn : str, optional, default="leaky_relu"
        Activation name resolved with :func:`~physicsnemo.nn.get_activation`. Defaults to "leaky_relu".
    depth : int, optional, default=2
        Depth of the U-Net encoder/decoder stacks. Defaults to 2.
    clamp_activation : Tuple[float | int | None, float | int | None], optional, default=(None, 10.0)
        Minimum and maximum bounds applied via ``torch.clamp`` after activation. Defaults to (None, 10.0).

    Forward
    -------
    cubed_sphere_input : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, F, H, W)` with :math:`F=6` faces.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, F, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models import DLWP
    >>> model = DLWP(nr_input_channels=2, nr_output_channels=4)
    >>> inputs = torch.randn(4, 2, 6, 64, 64)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([4, 4, 6, 64, 64])
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
    ) -> None:
        r"""Initialize the DLWP model."""
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
    def activation(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels height width"]:
        r"""
        Apply activation and optional clamping to a face tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input face tensor of shape :math:`(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Activated face tensor of shape :math:`(B, C, H, W)`.
        """
        x = self.activation_fn(x)
        if any(isinstance(c, (float, int)) for c in self.clamp_activation):
            x = torch.clamp(
                x, min=self.clamp_activation[0], max=self.clamp_activation[1]
            )
        return x

    def forward(
        self,
        cubed_sphere_input: Float[torch.Tensor, "batch channels faces height width"],
    ) -> Float[torch.Tensor, "batch channels_out faces height width"]:
        r"""Apply the DLWP forward pass to cubed-sphere input data."""
        # Input validation (skip under torch.compile)
        if not torch.compiler.is_compiling():
            if cubed_sphere_input.ndim != 5:
                raise ValueError(
                    "Expected input tensor of shape (B, C, F, H, W) but got tensor of "
                    f"shape {tuple(cubed_sphere_input.shape)}"
                )
            batch, channels, faces_count, height, width = cubed_sphere_input.shape
            if channels != self.nr_input_channels:
                raise ValueError(
                    f"Expected input tensor with {self.nr_input_channels} channels but "
                    f"got {channels} channels"
                )
            if faces_count != 6:
                raise ValueError(
                    "Expected input tensor of shape (B, C, 6, H, W) but got tensor of "
                    f"shape {tuple(cubed_sphere_input.shape)}"
                )
            if height != width:
                raise ValueError(
                    "Expected input tensor of shape (B, C, F, H, H) but got tensor of "
                    f"shape {tuple(cubed_sphere_input.shape)}"
                )

        # Split cubed-sphere input into individual faces
        faces = torch.split(
            cubed_sphere_input, split_size_or_sections=1, dim=2
        )  # (B, C, 1, H, W)
        faces = [torch.squeeze(face, dim=2) for face in faces]  # (B, C, H, W)

        encoder_states = []

        # Encoder: per-face convolutions with downsampling
        for i, (equatorial_layer, polar_layer) in enumerate(
            zip(self.equatorial_downsample, self.polar_downsample)
        ):
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)
            if i % 2 != 0:
                encoder_states.append(faces)
                faces = _cubed_non_conv_wrapper(faces, self.avg_pool)

        # Bottleneck convolutions
        for i, (equatorial_layer, polar_layer) in enumerate(
            zip(self.equatorial_mid_layers, self.polar_mid_layers)
        ):
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)

        j = 0
        # Decoder: upsample, concatenate skip connections, and convolve
        for i, (equatorial_layer, polar_layer) in enumerate(
            zip(self.equatorial_upsample, self.polar_upsample)
        ):
            if i % 2 == 0:
                encoder_faces = encoder_states[len(encoder_states) - j - 1]
                faces = _cubed_non_conv_wrapper(faces, self.upsample_layer)
                faces = [
                    torch.cat((face_1, face_2), dim=1)  # (B, 2*C, H, W)
                    for face_1, face_2 in zip(faces, encoder_faces)
                ]
                j += 1
            faces = _cubed_conv_wrapper(faces, equatorial_layer, polar_layer)
            faces = _cubed_non_conv_wrapper(faces, self.activation)

        # Final face-wise projection and reassembly
        faces = _cubed_conv_wrapper(faces, self.equatorial_last, self.polar_last)
        output = torch.stack(faces, dim=2)

        return output
