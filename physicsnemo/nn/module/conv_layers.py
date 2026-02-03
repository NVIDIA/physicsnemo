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

import math
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.module import Module
from physicsnemo.nn.module.utils.utils import _validate_amp
from physicsnemo.nn.module.utils.weight_init import _weight_init


class CubeEmbedding(nn.Module):
    """
    3D Image Cube Embedding
    Args:
        img_size (tuple[int]): Image size [T, Lat, Lon].
        patch_size (tuple[int]): Patch token size [T, Lat, Lon].
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
    """

    def __init__(
        self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class ConvBlock(nn.Module):
    """
    Conv2d block
    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        num_groups (int): Number of groups to separate the channels into for group normalization.
        num_residuals (int, optinal): Number of Conv2d operator. Default: 2
        upsample (int, optinal): 1: Upsample, 0: Conv, -1: Downsample. Default: 0
    """

    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2, upsample=0):
        super().__init__()
        if upsample == 1:
            self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif upsample == -1:
            self.conv = nn.Conv2d(
                in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1
            )
        elif upsample == 0:
            self.conv = nn.Conv2d(
                in_chans, out_chans, kernel_size=(3, 3), stride=1, padding=1
            )

        blk = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)
        x_skip = x
        x = self.b(x)
        return x + x_skip


def _get_same_padding(x: int, k: int, s: int) -> int:
    r"""
    Function to compute "same" padding.

    Inspired from: `timm padding <https://github.com/huggingface/pytorch-image-models/blob/0.5.x/timm/models/layers/padding.py>`_

    Parameters
    ----------
    x : int
        Input dimension size.
    k : int
        Kernel size.
    s : int
        Stride.

    Returns
    -------
    int
        Padding value to achieve "same" padding.
    """
    return max(s * math.ceil(x / s) - s - x + k, 0)


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of conv2d. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: Literal[
            "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
        ] = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
        fused_conv_bias: bool = False,
        amp_mode: bool = False,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")
        if not kernel and fused_conv_bias:
            print(
                "Warning: Kernel is required when fused_conv_bias is enabled. Setting fused_conv_bias to False."
            )
            fused_conv_bias = False

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.fused_conv_bias = fused_conv_bias
        self.amp_mode = amp_mode
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                _weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        weight, bias, resample_filter = self.weight, self.bias, self.resample_filter
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
            if (
                self.resample_filter is not None
                and self.resample_filter.dtype != x.dtype
            ):
                resample_filter = self.resample_filter.to(x.dtype)

        w = weight if weight is not None else None
        b = bias if bias is not None else None
        f = resample_filter if resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x, w, padding=max(w_pad - f_pad, 0), bias=b
                )
            else:
                x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                    bias=b,
                )
            else:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:  # ask in corrdiff channel whether w will ever be none
                if self.fused_conv_bias:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, bias=b)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None and not self.fused_conv_bias:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class ConvLayer(Module):
    r"""
    Generalized Convolution Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dimension : int
        Dimensionality of the input (1, 2, or 3).
    kernel_size : int
        Kernel size for the convolution.
    stride : int, optional, default=1
        Stride for the convolution.
    activation_fn : nn.Module, optional, default=nn.Identity()
        Activation function to use.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, *)` where :math:`*` represents
        spatial dimensions matching ``dimension``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *)` where spatial dimensions
        depend on stride and padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dimension = dimension
        self.activation_fn = activation_fn

        if self.dimension == 1:
            self.conv = nn.Conv1d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif self.dimension == 2:
            self.conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif self.dimension == 3:
            self.conv = nn.Conv3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        else:
            raise ValueError("Only 1D, 2D and 3D dimensions are supported")

        self._reset_parameters()

    def _exec_activation_fn(
        self,
        x: Float[Tensor, "batch channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch channels ..."]:  # noqa: F722
        r"""
        Executes activation function on the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C, *)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, C, *)`.
        """
        return self.activation_fn(x)

    def _reset_parameters(self) -> None:
        r"""
        Initialization for network parameters.
        """
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(
        self,
        x: Float[Tensor, "batch in_channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch out_channels ..."]:  # noqa: F722
        r"""Forward pass with same padding."""
        ### Input validation
        if not torch.compiler.is_compiling():
            input_length = len(x.size()) - 2  # exclude channel and batch dims
            if input_length != self.dimension:
                raise ValueError(
                    f"Expected {self.dimension}D input tensor (excluding batch and channel dims), "
                    f"got {input_length}D tensor with shape {tuple(x.shape)}"
                )

        input_length = len(x.size()) - 2  # exclude channel and batch dims

        # Apply same padding based on dimensionality
        if input_length == 1:
            iw = x.size()[-1:][0]
            pad_w = _get_same_padding(iw, self.kernel_size, self.stride)
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2], mode="constant", value=0.0)
        elif input_length == 2:
            ih, iw = x.size()[-2:]
            pad_h, pad_w = (
                _get_same_padding(ih, self.kernel_size, self.stride),
                _get_same_padding(iw, self.kernel_size, self.stride),
            )
            # F.pad expects padding in reverse dimension order: [left, right, top, bottom]
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                mode="constant",
                value=0.0,
            )
        else:
            _id, ih, iw = x.size()[-3:]
            pad_d, pad_h, pad_w = (
                _get_same_padding(_id, self.kernel_size, self.stride),
                _get_same_padding(ih, self.kernel_size, self.stride),
                _get_same_padding(iw, self.kernel_size, self.stride),
            )
            # F.pad expects padding in reverse dimension order: [left, right, top, bottom, front, back]
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_d // 2,
                    pad_d - pad_d // 2,
                ],
                mode="constant",
                value=0.0,
            )

        # Apply convolution
        x = self.conv(x)

        # Apply activation if not identity
        if self.activation_fn is not nn.Identity():
            x = self._exec_activation_fn(x)

        return x


class TransposeConvLayer(Module):
    r"""
    Generalized Transposed Convolution Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dimension : int
        Dimensionality of the input (1, 2, or 3).
    kernel_size : int
        Kernel size for the convolution.
    stride : int, optional, default=1
        Stride for the convolution.
    activation_fn : nn.Module, optional, default=nn.Identity()
        Activation function to use.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, *)` where :math:`*` represents
        spatial dimensions matching ``dimension``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *)` where spatial dimensions
        are upsampled based on stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int,
        stride: int = 1,
        activation_fn=nn.Identity(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dimension = dimension
        self.activation_fn = activation_fn

        if dimension == 1:
            self.trans_conv = nn.ConvTranspose1d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif dimension == 2:
            self.trans_conv = nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif dimension == 3:
            self.trans_conv = nn.ConvTranspose3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        else:
            raise ValueError("Only 1D, 2D and 3D dimensions are supported")

        self._reset_parameters()

    def _exec_activation_fn(
        self,
        x: Float[Tensor, "batch channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch channels ..."]:  # noqa: F722
        r"""
        Executes activation function on the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C, *)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, C, *)`.
        """
        return self.activation_fn(x)

    def _reset_parameters(self) -> None:
        r"""
        Initialization for network parameters.
        """
        nn.init.constant_(self.trans_conv.bias, 0)
        nn.init.xavier_uniform_(self.trans_conv.weight)

    def forward(
        self,
        x: Float[Tensor, "batch in_channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch out_channels ..."]:  # noqa: F722
        r"""Forward pass with transposed convolution and cropping."""
        ### Input validation
        if not torch.compiler.is_compiling():
            input_length = len(x.size()) - 2  # exclude channel and batch dims
            if input_length != self.dimension:
                raise ValueError(
                    f"Expected {self.dimension}D input tensor (excluding batch and channel dims), "
                    f"got {input_length}D tensor with shape {tuple(x.shape)}"
                )

        orig_x = x
        input_length = len(orig_x.size()) - 2  # exclude channel and batch dims

        # Apply transposed convolution
        x = self.trans_conv(x)

        # Crop output to match expected output size (same padding logic)
        if input_length == 1:
            iw = orig_x.size()[-1:][0]
            pad_w = _get_same_padding(iw, self.kernel_size, self.stride)
            x = x[
                :,
                :,
                pad_w // 2 : x.size(-1) - (pad_w - pad_w // 2),
            ]
        elif input_length == 2:
            ih, iw = orig_x.size()[-2:]
            pad_h, pad_w = (
                _get_same_padding(
                    ih,
                    self.kernel_size,
                    self.stride,
                ),
                _get_same_padding(iw, self.kernel_size, self.stride),
            )
            x = x[
                :,
                :,
                pad_h // 2 : x.size(-2) - (pad_h - pad_h // 2),
                pad_w // 2 : x.size(-1) - (pad_w - pad_w // 2),
            ]
        else:
            _id, ih, iw = orig_x.size()[-3:]
            pad_d, pad_h, pad_w = (
                _get_same_padding(_id, self.kernel_size, self.stride),
                _get_same_padding(ih, self.kernel_size, self.stride),
                _get_same_padding(iw, self.kernel_size, self.stride),
            )
            x = x[
                :,
                :,
                pad_d // 2 : x.size(-3) - (pad_d - pad_d // 2),
                pad_h // 2 : x.size(-2) - (pad_h - pad_h // 2),
                pad_w // 2 : x.size(-1) - (pad_w - pad_w // 2),
            ]

        # Apply activation if not identity
        if self.activation_fn is not nn.Identity():
            x = self._exec_activation_fn(x)

        return x


class ConvGRULayer(Module):
    r"""
    Convolutional GRU layer.

    Parameters
    ----------
    in_features : int
        Input features/channels.
    hidden_size : int
        Hidden layer features/channels.
    dimension : int
        Spatial dimension of the input.
    activation_fn : nn.Module, optional, default=nn.ReLU()
        Activation Function to use.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, *)` where :math:`*` represents
        spatial dimensions.
    hidden : torch.Tensor
        Hidden state tensor of shape :math:`(B, H, *)` where :math:`H` is
        ``hidden_size``.

    Outputs
    -------
    torch.Tensor
        Next hidden state of shape :math:`(B, H, *)`.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        dimension: int,
        activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.conv_1 = ConvLayer(
            in_channels=in_features + hidden_size,
            out_channels=2 * hidden_size,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )
        self.conv_2 = ConvLayer(
            in_channels=in_features + hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )

    def _exec_activation_fn(
        self,
        x: Float[Tensor, "batch channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch channels ..."]:  # noqa: F722
        r"""
        Executes activation function on the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C, *)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, C, *)`.
        """
        return self.activation_fn(x)

    def forward(
        self,
        x: Float[Tensor, "batch in_features ..."],  # noqa: F722
        hidden: Float[Tensor, "batch hidden_size ..."],  # noqa: F722
    ) -> Float[Tensor, "batch hidden_size ..."]:  # noqa: F722
        r"""Forward pass implementing GRU update."""
        ### Input validation
        if not torch.compiler.is_compiling():
            if x.shape[1] != self.in_features:
                raise ValueError(
                    f"Expected input with {self.in_features} features, "
                    f"got {x.shape[1]} features in tensor with shape {tuple(x.shape)}"
                )
            if hidden.shape[1] != self.hidden_size:
                raise ValueError(
                    f"Expected hidden state with {self.hidden_size} features, "
                    f"got {hidden.shape[1]} features in tensor with shape {tuple(hidden.shape)}"
                )
            if x.shape[0] != hidden.shape[0] or x.shape[2:] != hidden.shape[2:]:
                raise ValueError(
                    f"Input and hidden state must have matching batch size and spatial dims. "
                    f"Got input shape {tuple(x.shape)} and hidden shape {tuple(hidden.shape)}"
                )

        # Concatenate input and hidden state
        concat = torch.cat((x, hidden), dim=1)  # (B, in_features + hidden_size, *)

        # Compute reset and update gates
        conv_concat = self.conv_1(concat)  # (B, 2 * hidden_size, *)
        conv_r, conv_z = torch.split(conv_concat, self.hidden_size, 1)

        reset_gate = torch.special.expit(conv_r)  # (B, hidden_size, *)
        update_gate = torch.special.expit(conv_z)  # (B, hidden_size, *)

        # Compute candidate hidden state
        concat = torch.cat((x, torch.mul(hidden, reset_gate)), dim=1)
        n = self._exec_activation_fn(self.conv_2(concat))  # (B, hidden_size, *)

        # Compute next hidden state
        h_next = torch.mul((1 - update_gate), n) + torch.mul(update_gate, hidden)

        return h_next


class ConvResidualBlock(Module):
    r"""
    Convolutional ResNet Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dimension : int
        Dimensionality of the input.
    stride : int, optional, default=1
        Stride of the convolutions.
    gated : bool, optional, default=False
        Residual Gate activation.
    layer_normalization : bool, optional, default=False
        Whether to apply layer normalization.
    begin_activation_fn : bool, optional, default=True
        Whether to use activation function in the beginning.
    activation_fn : nn.Module, optional, default=nn.ReLU()
        Activation function to use.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, *)` where :math:`*` represents
        spatial dimensions matching ``dimension``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *)` with residual connection.

    Raises
    ------
    ValueError
        If stride > 2 (not supported).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        stride: int = 1,
        gated: bool = False,
        layer_normalization: bool = False,
        begin_activation_fn: bool = True,
        activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dimension = dimension
        self.gated = gated
        self.layer_normalization = layer_normalization
        self.begin_activation_fn = begin_activation_fn
        self.activation_fn = activation_fn

        if self.stride == 1:
            self.conv_1 = ConvLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                dimension=self.dimension,
            )
        elif self.stride == 2:
            self.conv_1 = ConvLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=self.stride,
                dimension=self.dimension,
            )
        else:
            raise ValueError("stride > 2 is not supported")

        if not self.gated:
            self.conv_2 = ConvLayer(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                dimension=self.dimension,
            )
        else:
            self.conv_2 = ConvLayer(
                in_channels=self.out_channels,
                out_channels=2 * self.out_channels,
                kernel_size=3,
                stride=1,
                dimension=self.dimension,
            )

    def _exec_activation_fn(
        self,
        x: Float[Tensor, "batch channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch channels ..."]:  # noqa: F722
        r"""
        Executes activation function on the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C, *)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, C, *)`.
        """
        return self.activation_fn(x)

    def forward(
        self,
        x: Float[Tensor, "batch in_channels ..."],  # noqa: F722
    ) -> Float[Tensor, "batch out_channels ..."]:  # noqa: F722
        r"""Forward pass with residual connection."""
        ### Input validation
        if not torch.compiler.is_compiling():
            input_length = len(x.size()) - 2  # exclude channel and batch dims
            if input_length != self.dimension:
                raise ValueError(
                    f"Expected {self.dimension}D input tensor (excluding batch and channel dims), "
                    f"got {input_length}D tensor with shape {tuple(x.shape)}"
                )

        orig_x = x

        # Apply layer normalization and activation at the beginning if specified
        if self.begin_activation_fn:
            if self.layer_normalization:
                layer_norm = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
                x = layer_norm(x)
            x = self._exec_activation_fn(x)

        # First convolutional layer
        x = self.conv_1(x)

        # Apply layer normalization after first convolution
        if self.layer_normalization:
            layer_norm = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
            x = layer_norm(x)

        # Second activation and convolution
        x = self._exec_activation_fn(x)
        x = self.conv_2(x)

        # Apply gating if specified
        if self.gated:
            x_1, x_2 = torch.split(x, x.size(1) // 2, 1)
            x = x_1 * torch.special.expit(x_2)

        # Adjust skip connection if spatial dimensions differ (due to stride)
        if orig_x.size(-1) > x.size(-1):  # Check if widths are different
            if len(orig_x.size()) - 2 == 1:
                iw = orig_x.size()[-1:][0]
                pad_w = _get_same_padding(iw, 2, 2)
                pool = torch.nn.AvgPool1d(
                    2, 2, padding=pad_w // 2, count_include_pad=False
                )
            elif len(orig_x.size()) - 2 == 2:
                ih, iw = orig_x.size()[-2:]
                pad_h, pad_w = (
                    _get_same_padding(
                        ih,
                        2,
                        2,
                    ),
                    _get_same_padding(iw, 2, 2),
                )
                pool = torch.nn.AvgPool2d(
                    2, 2, padding=(pad_h // 2, pad_w // 2), count_include_pad=False
                )
            elif len(orig_x.size()) - 2 == 3:
                _id, ih, iw = orig_x.size()[-3:]
                pad_d, pad_h, pad_w = (
                    _get_same_padding(_id, 2, 2),
                    _get_same_padding(ih, 2, 2),
                    _get_same_padding(iw, 2, 2),
                )
                pool = torch.nn.AvgPool3d(
                    2,
                    2,
                    padding=(pad_d // 2, pad_h // 2, pad_w // 2),
                    count_include_pad=False,
                )
            else:
                raise ValueError("Only 1D, 2D and 3D dimensions are supported")
            orig_x = pool(orig_x)

        # Adjust skip connection channels if needed
        in_channels = int(orig_x.size(1))
        if self.out_channels > in_channels:
            orig_x = F.pad(
                orig_x,
                (len(orig_x.size()) - 2) * (0, 0)
                + (self.out_channels - self.in_channels, 0),
            )
        elif self.out_channels < in_channels:
            pass

        return orig_x + x
