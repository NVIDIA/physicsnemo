# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import importlib
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", "0.10.0", hard_fail=False)

if TE_AVAILABLE:
    te = importlib.import_module("transformer_engine.pytorch")
else:
    te = None


class ReshapedLayerNorm(te.LayerNorm if te else nn.LayerNorm):
    r"""LayerNorm that normalizes over channels for spatial tensors.

    Reshapes and transposes the input tensor before applying layer normalization,
    then restores the original shape. This enables layer normalization over the
    channel dimension while preserving the spatial structure.

    .. note::
        When ``transformer_engine`` is installed, this class inherits from
        :class:`transformer_engine.pytorch.LayerNorm`. Otherwise, it inherits
        from :class:`torch.nn.LayerNorm`.

    Parameters
    ----------
    normalized_shape : int | Sequence[int]
        Input shape over which to normalize. If a single integer, treated as a
        singleton list.
    eps : float, optional, default=1e-5
        Value added to denominator for numerical stability.
    elementwise_affine : bool, optional, default=True
        Whether to learn affine parameters (scale and shift).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, *spatial)` where :math:`C` is the
        number of channels and :math:`*spatial` are spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Normalized tensor with same shape as input.
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(
        self, x: Float[torch.Tensor, "batch channels *spatial"]
    ) -> Float[torch.Tensor, "batch channels *spatial"]:
        """Forward pass applying reshaped layer normalization."""
        shape = x.shape
        x = x.view(shape[0], shape[1], -1).transpose(1, 2).contiguous()
        x = super().forward(x)
        x = x.transpose(1, 2).contiguous().view(shape)
        return x


class Conv3DBlock(Module):
    r"""3D convolutional block with optional normalization and activation.

    Applies a 3D convolution followed by optional normalization and activation
    functions. Used as a building block in encoder and decoder paths.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    kernel_size : int | tuple[int, ...], optional, default=3
        Size of the convolving kernel.
    stride : int | tuple[int, ...], optional, default=1
        Stride of the convolution.
    padding : int | tuple[int, ...], optional, default=1
        Padding added to all sides of the input.
    dilation : int | tuple[int, ...], optional, default=1
        Spacing between kernel elements.
    groups : int, optional, default=1
        Number of blocked connections from input to output channels.
    bias : bool, optional, default=True
        If ``True``, adds a learnable bias to the output.
    padding_mode : Literal["zeros", "reflect", "replicate", "circular"], optional, default="zeros"
        Padding mode for convolutions.
    activation : str | None, optional, default="relu"
        Activation function name from :mod:`torch.nn.functional`.
    normalization : Literal["groupnorm", "batchnorm", "layernorm"] | None, optional, default="groupnorm"
        Normalization type. If ``None``, no normalization is applied.
    normalization_args : dict | None, optional, default=None
        Additional arguments for the normalization layer.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, D', H', W')`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 1,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        activation: str | None = "relu",
        normalization: Literal["groupnorm", "batchnorm", "layernorm"]
        | None = "groupnorm",
        normalization_args: dict | None = None,
    ):
        super().__init__()
        # Initialize convolution layer
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # Initialize activation function
        if activation:
            if hasattr(F, activation):
                self.activation = getattr(F, activation)
            else:
                raise ValueError(f"Activation type '{activation}' is not supported.")
        else:
            self.activation = nn.Identity()

        # Initialize normalization layer
        if normalization:
            if normalization == "groupnorm":
                default_args = {"num_groups": 1, "num_channels": out_channels}
                norm_args = {
                    **default_args,
                    **(normalization_args if normalization_args else {}),
                }
                self.norm = nn.GroupNorm(**norm_args)
            elif normalization == "batchnorm":
                self.norm = nn.BatchNorm3d(out_channels)
            elif normalization == "layernorm":
                self.norm = ReshapedLayerNorm(out_channels)
            else:
                raise ValueError(
                    f"Normalization type '{normalization}' is not supported."
                )
        else:
            self.norm = nn.Identity()

    def forward(
        self, x: Float[torch.Tensor, "batch c_in depth height width"]
    ) -> Float[torch.Tensor, "batch c_out depth_out height_out width_out"]:
        """Forward pass through the convolutional block."""
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvTranspose3D(Module):
    r"""3D transposed convolutional block with optional normalization and activation.

    Applies a transposed 3D convolution for upsampling, followed by optional
    normalization and activation functions. Used in the decoder path.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    kernel_size : int | tuple[int, ...], optional, default=3
        Size of the convolving kernel.
    stride : int | tuple[int, ...], optional, default=2
        Stride of the convolution.
    padding : int | tuple[int, ...], optional, default=1
        Padding added to all sides of the input.
    output_padding : int | tuple[int, ...], optional, default=1
        Additional size added to one side of the output shape.
    groups : int, optional, default=1
        Number of blocked connections from input to output channels.
    bias : bool, optional, default=True
        If ``True``, adds a learnable bias to the output.
    dilation : int | tuple[int, ...], optional, default=1
        Spacing between kernel elements.
    padding_mode : Literal["zeros", "reflect", "replicate", "circular"], optional, default="zeros"
        Padding mode for convolutions.
    activation : str | None, optional, default=None
        Activation function name from :mod:`torch.nn.functional`.
    normalization : Literal["groupnorm", "batchnorm", "layernorm"] | None, optional, default=None
        Normalization type. If ``None``, no normalization is applied.
    normalization_args : dict | None, optional, default=None
        Additional arguments for the normalization layer.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Upsampled output tensor of shape :math:`(B, C_{out}, D', H', W')`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 2,
        padding: int | tuple[int, ...] = 1,
        output_padding: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, ...] = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        activation: str | None = None,
        normalization: Literal["groupnorm", "batchnorm", "layernorm"] | None = None,
        normalization_args: dict | None = None,
    ):
        super().__init__()
        # Initialize transposed convolution layer
        self.conv3d_transpose = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        # Initialize activation function
        if activation:
            if hasattr(F, activation):
                self.activation = getattr(F, activation)
            else:
                raise ValueError(f"Activation type '{activation}' is not supported.")
        else:
            self.activation = nn.Identity()

        # Initialize normalization layer
        if normalization:
            if normalization == "groupnorm":
                default_args = {"num_groups": 1, "num_channels": out_channels}
                norm_args = {
                    **default_args,
                    **(normalization_args if normalization_args else {}),
                }
                self.norm = nn.GroupNorm(**norm_args)
            elif normalization == "batchnorm":
                self.norm = nn.BatchNorm3d(out_channels)
            elif normalization == "layernorm":
                self.norm = ReshapedLayerNorm(out_channels)
            else:
                raise ValueError(
                    f"Normalization type '{normalization}' is not supported."
                )
        else:
            self.norm = nn.Identity()

    def forward(
        self, x: Float[torch.Tensor, "batch c_in depth height width"]
    ) -> Float[torch.Tensor, "batch c_out depth_out height_out width_out"]:
        """Forward pass through the transposed convolutional block."""
        x = self.conv3d_transpose(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Pool3D(Module):
    r"""3D pooling block.

    Applies a specified 3D pooling operation (average or max) for downsampling.

    Parameters
    ----------
    pooling_type : Literal["AvgPool3d", "MaxPool3d"], optional, default="AvgPool3d"
        Type of pooling: ``"AvgPool3d"`` or ``"MaxPool3d"``.
    kernel_size : int | tuple[int, ...], optional, default=2
        Size of the pooling window.
    stride : int | tuple[int, ...] | None, optional, default=None
        Stride of the pooling. If ``None``, uses ``kernel_size``.
    padding : int | tuple[int, ...], optional, default=0
        Implicit zero padding on both sides of the input.
    dilation : int | tuple[int, ...], optional, default=1
        Spacing between kernel points (only for ``MaxPool3d``).
    ceil_mode : bool, optional, default=False
        If ``True``, use ceil instead of floor to compute output shape.
    count_include_pad : bool, optional, default=True
        For ``AvgPool3d`` only: include zero-padding in averaging.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Pooled output tensor of shape :math:`(B, C, D', H', W')`.
    """

    def __init__(
        self,
        pooling_type: Literal["AvgPool3d", "MaxPool3d"] = "AvgPool3d",
        kernel_size: int | tuple[int, ...] = 2,
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__()

        # Validate pooling type and initialize pooling layer
        if pooling_type not in ["AvgPool3d", "MaxPool3d"]:
            raise ValueError(
                f"Invalid pooling_type '{pooling_type}'. Please choose from ['AvgPool3d', 'MaxPool3d'] or implement additional types."
            )

        # Initialize the corresponding pooling layer
        if pooling_type == "AvgPool3d":
            self.pooling = nn.AvgPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        elif pooling_type == "MaxPool3d":
            self.pooling = nn.MaxPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )

    def forward(
        self, x: Float[torch.Tensor, "batch channels depth height width"]
    ) -> Float[torch.Tensor, "batch channels depth_out height_out width_out"]:
        """Forward pass through the pooling layer."""
        return self.pooling(x)


class Attention3DBlock(Module):
    r"""Attention gate for skip connections in 3D U-Net architectures.

    Applies an attention mechanism to modulate skip connection features based
    on the decoder's gating signal. Uses LayerNorm for normalization.

    Parameters
    ----------
    F_g : int
        Number of channels in the decoder's gating features (query) :math:`C_g`.
    F_l : int
        Number of channels in the encoder's skip features (key/value) :math:`C_l`.
    F_int : int
        Number of intermediate channels :math:`C_{int}` for attention computation.

    Forward
    -------
    g : torch.Tensor
        Gating signal from decoder of shape :math:`(B, C_g, D, H, W)`.
    x : torch.Tensor
        Skip connection features from encoder of shape :math:`(B, C_l, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Attended skip features of shape :math:`(B, C_l, D, H, W)`.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Project gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            ReshapedLayerNorm(F_int),
        )

        # Project skip connection features
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            ReshapedLayerNorm(F_int),
        )

        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            ReshapedLayerNorm(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        g: Float[torch.Tensor, "batch c_g depth height width"],
        x: Float[torch.Tensor, "batch c_l depth height width"],
    ) -> Float[torch.Tensor, "batch c_l depth height width"]:
        """Forward pass computing attention-weighted skip features."""
        # Compute attention
        g1 = self.W_g(g)  # (B, F_int, D, H, W)
        x1 = self.W_x(x)  # (B, F_int, D, H, W)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B, 1, D, H, W)
        return x * psi  # Element-wise multiplication with attention mask


class Encoder3DBlock(Module):
    r"""U-Net encoder block with multi-scale feature extraction.

    Sequentially applies convolutional blocks with pooling operations to
    progressively downsample and extract features at multiple scales.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    feature_map_channels : Sequence[int]
        Channel sizes for each conv block. Length must equal
        ``model_depth * num_conv_blocks``.
    kernel_size : int | tuple[int, ...], optional, default=3
        Size of the convolving kernel.
    stride : int | tuple[int, ...], optional, default=1
        Stride of the convolution.
    model_depth : int, optional, default=4
        Number of depth levels (conv-pool repetitions).
    num_conv_blocks : int, optional, default=2
        Number of convolutional blocks per depth level.
    activation : str | None, optional, default="relu"
        Activation function name.
    padding : int, optional, default=1
        Padding for convolutions.
    padding_mode : Literal["zeros", "reflect", "replicate", "circular"], optional, default="zeros"
        Padding mode for convolutions.
    pooling_type : Literal["AvgPool3d", "MaxPool3d"], optional, default="AvgPool3d"
        Type of pooling.
    pool_size : int, optional, default=2
        Pooling window size.
    normalization : Literal["groupnorm", "batchnorm", "layernorm"] | None, optional, default="groupnorm"
        Normalization type. If ``None``, no normalization is applied.
    normalization_args : dict | None, optional, default=None
        Additional normalization arguments.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Encoded features at the deepest level.
    """

    def __init__(
        self,
        in_channels: int,
        feature_map_channels: Sequence[int],
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        model_depth: int = 4,
        num_conv_blocks: int = 2,
        activation: str | None = "relu",
        padding: int = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        pooling_type: Literal["AvgPool3d", "MaxPool3d"] = "AvgPool3d",
        pool_size: int = 2,
        normalization: Literal["groupnorm", "batchnorm", "layernorm"]
        | None = "groupnorm",
        normalization_args: dict | None = None,
    ):
        super().__init__()

        if len(feature_map_channels) != model_depth * num_conv_blocks:
            raise ValueError(
                "The length of feature_map_channels should be equal to model_depth * num_conv_blocks"
            )

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for depth in range(model_depth):
            for i in range(num_conv_blocks):
                self.layers.append(
                    Conv3DBlock(
                        in_channels=current_channels,
                        out_channels=feature_map_channels[depth * num_conv_blocks + i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        padding_mode=padding_mode,
                        activation=activation,
                        normalization=normalization,
                        normalization_args=normalization_args,
                    )
                )
                current_channels = feature_map_channels[depth * num_conv_blocks + i]

            if (
                depth < model_depth - 1
            ):  # Add pooling between levels but not at the last level
                self.layers.append(
                    Pool3D(pooling_type=pooling_type, kernel_size=pool_size)
                )

    def forward(
        self, x: Float[torch.Tensor, "batch c_in depth height width"]
    ) -> Float[torch.Tensor, "batch c_out depth_out height_out width_out"]:
        """Forward pass through the encoder block."""
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder3DBlock(Module):
    r"""U-Net decoder block with upsampling and skip connection support.

    Sequentially applies transposed convolutions for upsampling and regular
    convolutions for feature processing. Designed to concatenate features
    from the encoder (skip connections) externally.

    Parameters
    ----------
    out_channels : int
        Number of output channels :math:`C_{out}`.
    feature_map_channels : Sequence[int]
        Channel sizes for each layer. Length must equal
        ``model_depth * num_conv_blocks + 1``.
    kernel_size : int | tuple[int, ...], optional, default=3
        Size of the convolving kernel.
    stride : int | tuple[int, ...], optional, default=1
        Stride of the convolution.
    model_depth : int, optional, default=3
        Number of depth levels.
    num_conv_blocks : int, optional, default=2
        Number of convolutional blocks per depth level.
    conv_activation : str | None, optional, default="relu"
        Activation for convolutional layers.
    conv_transpose_activation : str | None, optional, default=None
        Activation for transposed convolutional layers.
    padding : int, optional, default=1
        Padding for convolutions.
    padding_mode : Literal["zeros", "reflect", "replicate", "circular"], optional, default="zeros"
        Padding mode for convolutions.
    normalization : Literal["groupnorm", "batchnorm", "layernorm"] | None, optional, default="groupnorm"
        Normalization type. If ``None``, no normalization is applied.
    normalization_args : dict | None, optional, default=None
        Additional normalization arguments.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)`.

    Outputs
    -------
    torch.Tensor
        Decoded output tensor of shape :math:`(B, C_{out}, D', H', W')`.
    """

    def __init__(
        self,
        out_channels: int,
        feature_map_channels: Sequence[int],
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        model_depth: int = 3,
        num_conv_blocks: int = 2,
        conv_activation: str | None = "relu",
        conv_transpose_activation: str | None = None,
        padding: int = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        normalization: Literal["groupnorm", "batchnorm", "layernorm"]
        | None = "groupnorm",
        normalization_args: dict | None = None,
    ):
        super().__init__()

        if len(feature_map_channels) != model_depth * num_conv_blocks + 1:
            raise ValueError(
                "The length of feature_map_channels in the decoder block should be equal to model_depth * num_conv_blocks + 1"
            )

        self.layers = nn.ModuleList()
        current_channels = feature_map_channels[0]
        feature_map_channels = feature_map_channels[1:]

        for depth in range(model_depth):
            for i in range(num_conv_blocks):
                if i == 0:
                    self.layers.append(
                        ConvTranspose3D(
                            in_channels=current_channels,
                            out_channels=current_channels,
                            activation=conv_transpose_activation,
                        )
                    )
                    current_channels += feature_map_channels[
                        depth * num_conv_blocks + i
                    ]

                self.layers.append(
                    Conv3DBlock(
                        in_channels=current_channels,
                        out_channels=feature_map_channels[depth * num_conv_blocks + i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        padding_mode=padding_mode,
                        activation=conv_activation,
                        normalization=normalization,
                        normalization_args=normalization_args,
                    )
                )
                current_channels = feature_map_channels[depth * num_conv_blocks + i]

        # Final convolution
        self.layers.append(
            Conv3DBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                activation=None,
                normalization=None,
            )
        )

    def forward(
        self, x: Float[torch.Tensor, "batch c_in depth height width"]
    ) -> Float[torch.Tensor, "batch c_out depth_out height_out width_out"]:
        """Forward pass through the decoder block."""
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = True
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class UNet(Module):
    r"""3D U-Net model with encoder-decoder architecture and skip connections.

    Implements the U-Net architecture for volumetric data, featuring an encoder
    path for multi-scale feature extraction, a decoder path for upsampling, and
    skip connections to preserve spatial information. Optionally supports
    attention gates on skip connections.

    Based on the original U-Net paper:
    `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>`_.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    kernel_size : int | tuple[int, ...], optional, default=3
        Size of the convolving kernel.
    stride : int | tuple[int, ...], optional, default=1
        Stride of the convolution.
    model_depth : int, optional, default=5
        Number of levels in the U-Net (including bottleneck).
    feature_map_channels : Sequence[int], optional
        Channel sizes for each conv block. Length must equal
        ``model_depth * num_conv_blocks``.
    num_conv_blocks : int, optional, default=2
        Number of convolutional blocks per level.
    conv_activation : str | None, optional, default="relu"
        Activation function for convolutional layers.
    conv_transpose_activation : str | None, optional, default=None
        Activation function for transposed convolutional layers.
    padding : int, optional, default=1
        Padding for convolutions.
    padding_mode : Literal["zeros", "reflect", "replicate", "circular"], optional, default="zeros"
        Padding mode for convolutions.
    pooling_type : Literal["AvgPool3d", "MaxPool3d"], optional, default="MaxPool3d"
        Pooling type.
    pool_size : int, optional, default=2
        Pooling window size.
    normalization : Literal["groupnorm", "batchnorm", "layernorm"] | None, optional, default="groupnorm"
        Normalization type. If ``None``, no normalization is applied.
    normalization_args : dict | None, optional, default=None
        Additional normalization arguments.
    use_attn_gate : bool, optional, default=False
        Whether to use attention gates on skip connections.
    attn_decoder_feature_maps : Sequence[int] | None, optional, default=None
        Decoder channel sizes for attention (required if ``use_attn_gate=True``).
    attn_feature_map_channels : Sequence[int] | None, optional, default=None
        Encoder channel sizes for attention (required if ``use_attn_gate=True``).
    attn_intermediate_channels : int | None, optional, default=None
        Intermediate channels for attention computation.
    gradient_checkpointing : bool, optional, default=True
        Whether to use gradient checkpointing to reduce memory.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)` where :math:`B` is
        batch size, :math:`D` is depth, :math:`H` is height, and :math:`W` is width.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, D, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.unet import UNet
    >>> model = UNet(
    ...     in_channels=1,
    ...     out_channels=1,
    ...     model_depth=3,
    ...     feature_map_channels=[16, 16, 32, 32, 64, 64],
    ... )
    >>> x = torch.randn(2, 1, 32, 32, 32)
    >>> output = model(x)
    >>> output.shape
    torch.Size([2, 1, 32, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        model_depth: int = 5,
        feature_map_channels: Sequence[int] = (
            64,
            64,
            128,
            128,
            256,
            256,
            512,
            512,
            1024,
            1024,
        ),
        num_conv_blocks: int = 2,
        conv_activation: str | None = "relu",
        conv_transpose_activation: str | None = None,
        padding: int = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        pooling_type: Literal["AvgPool3d", "MaxPool3d"] = "MaxPool3d",
        pool_size: int = 2,
        normalization: Literal["groupnorm", "batchnorm", "layernorm"]
        | None = "groupnorm",
        normalization_args: dict | None = None,
        use_attn_gate: bool = False,
        attn_decoder_feature_maps: Sequence[int] | None = None,
        attn_feature_map_channels: Sequence[int] | None = None,
        attn_intermediate_channels: int | None = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attn_gate = use_attn_gate
        self.gradient_checkpointing = gradient_checkpointing

        # Construct the encoder
        self.encoder = Encoder3DBlock(
            in_channels=in_channels,
            feature_map_channels=feature_map_channels,
            kernel_size=kernel_size,
            stride=stride,
            model_depth=model_depth,
            num_conv_blocks=num_conv_blocks,
            activation=conv_activation,
            padding=padding,
            padding_mode=padding_mode,
            pooling_type=pooling_type,
            pool_size=pool_size,
            normalization=normalization,
            normalization_args=normalization_args,
        )

        # Construct the decoder
        if num_conv_blocks > 1:
            decoder_feature_maps = feature_map_channels[::-1][
                1:
            ]  # Reverse and discard the first channel
        else:
            decoder_feature_maps = feature_map_channels[::-1]
        self.decoder = Decoder3DBlock(
            out_channels=out_channels,
            feature_map_channels=decoder_feature_maps,
            kernel_size=kernel_size,
            stride=stride,
            model_depth=model_depth - 1,
            num_conv_blocks=num_conv_blocks,
            conv_activation=conv_activation,
            conv_transpose_activation=conv_transpose_activation,
            padding=padding,
            padding_mode=padding_mode,
            normalization=normalization,
            normalization_args=normalization_args,
        )

        # Initialize attention blocks for each skip connection
        if self.use_attn_gate:
            if attn_decoder_feature_maps is None:
                raise ValueError(
                    "attn_decoder_feature_maps is required when use_attn_gate=True"
                )
            if attn_feature_map_channels is None:
                raise ValueError(
                    "attn_feature_map_channels is required when use_attn_gate=True"
                )
            if attn_intermediate_channels is None:
                raise ValueError(
                    "attn_intermediate_channels is required when use_attn_gate=True"
                )
            self.attention_blocks = nn.ModuleList(
                [
                    Attention3DBlock(
                        F_g=attn_decoder_feature_maps[i],
                        F_l=attn_feature_map_channels[i],
                        F_int=attn_intermediate_channels,
                    )
                    for i in range(model_depth - 1)
                ]
            )

    def _checkpointed_forward(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient checkpointing to a layer if enabled.

        Parameters
        ----------
        layer : nn.Module
            The layer to apply.
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the layer.
        """
        if self.gradient_checkpointing:
            return checkpoint.checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(
        self, x: Float[torch.Tensor, "batch in_channels depth height width"]
    ) -> Float[torch.Tensor, "batch out_channels depth height width"]:
        """Forward pass through the U-Net."""
        # Validate input shape
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected 5D input tensor (B, C, D, H, W), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected input with {self.in_channels} channels, "
                    f"got {x.shape[1]} in tensor with shape {tuple(x.shape)}"
                )

        skip_features = []
        # Encoding path
        for layer in self.encoder.layers:
            if isinstance(layer, Pool3D):
                skip_features.append(x)
            # Apply checkpointing if enabled
            x = self._checkpointed_forward(layer, x)

        # Decoding path
        skip_features = skip_features[::-1]  # Reverse the skip features
        concats = 0  # Track number of concats
        for layer in self.decoder.layers:
            if isinstance(layer, ConvTranspose3D):
                x = self._checkpointed_forward(layer, x)
                if self.use_attn_gate:
                    # Apply attention to the skip connection
                    skip_att = self.attention_blocks[concats](x, skip_features[concats])
                    x = torch.cat([x, skip_att], dim=1)
                else:
                    x = torch.cat([x, skip_features[concats]], dim=1)
                concats += 1
            else:
                # Apply checkpointing for other layers
                x = self._checkpointed_forward(layer, x)

        return x
