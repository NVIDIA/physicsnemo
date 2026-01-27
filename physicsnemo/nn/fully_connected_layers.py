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

from typing import Callable, Literal, Union

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.nn.utils.utils import _validate_amp
from physicsnemo.nn.utils.weight_init import _weight_init

from .activations import Identity
from .weight_fact import WeightFactLinear
from .weight_norm import WeightNormLinear


class FCLayer(Module):
    r"""Densely connected neural network layer.

    A single fully connected layer with optional activation, weight normalization,
    and weight factorization.

    Parameters
    ----------
    in_features : int
        Size of input features :math:`D_{in}`.
    out_features : int
        Size of output features :math:`D_{out}`.
    activation_fn : Union[nn.Module, Callable[[Tensor], Tensor], None], optional, default=None
        Activation function to use. Can be ``None`` for no activation.
    weight_norm : bool, optional, default=False
        Applies weight normalization to the layer.
    weight_fact : bool, optional, default=False
        Applies weight factorization to the layer.
    activation_par : Union[nn.Parameter, None], optional, default=None
        Learnable scaling parameter for adaptive activations.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D_{in})` where :math:`*` denotes any
        number of leading batch dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(*, D_{out})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        weight_norm: bool = False,
        weight_fact: bool = False,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__()

        if activation_fn is None:
            self.activation_fn = Identity()
        else:
            self.activation_fn = activation_fn
        self.weight_norm = weight_norm
        self.weight_fact = weight_fact
        self.activation_par = activation_par

        # Ensure weight_norm and weight_fact are not both True
        if weight_norm and weight_fact:
            raise ValueError(
                "Cannot apply both weight normalization and weight factorization together, please select one."
            )

        if weight_norm:
            self.linear = WeightNormLinear(in_features, out_features, bias=True)
        elif weight_fact:
            self.linear = WeightFactLinear(in_features, out_features, bias=True)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset fully connected layer weights to Xavier uniform initialization."""
        if not self.weight_norm and not self.weight_fact:
            nn.init.constant_(self.linear.bias, 0)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer."""
        x = self.linear(x)

        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)

        return x


class ConvFCLayer(Module):
    r"""Base class for 1x1 convolutional layers acting on image channels.

    This abstract base class provides activation handling for convolutional
    layers that act like fully connected layers over the channel dimension.

    Parameters
    ----------
    activation_fn : Union[nn.Module, Callable[[Tensor], Tensor], None], optional, default=None
        Activation function to use. Can be ``None`` for no activation.
    activation_par : Union[nn.Parameter, None], optional, default=None
        Learnable scaling parameter for adaptive activations.

    Forward
    -------
    x : torch.Tensor
        Input tensor (shape depends on subclass).

    Outputs
    -------
    torch.Tensor
        Output tensor with activation applied.
    """

    def __init__(
        self,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__()
        if activation_fn is None:
            self.activation_fn = Identity()
        else:
            self.activation_fn = activation_fn
        self.activation_par = activation_par

    def apply_activation(self, x: Tensor) -> Tensor:
        r"""Apply activation function with optional learnable scaling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor
            Tensor with activation applied, same shape as input.
        """
        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):
    r"""Channel-wise fully connected layer using 1D convolutions.

    Applies a 1x1 convolution followed by an optional activation function.
    This is equivalent to a fully connected layer operating on the channel
    dimension of 1D signals.

    Parameters
    ----------
    in_features : int
        Number of input channels :math:`C_{in}`.
    out_features : int
        Number of output channels :math:`C_{out}`.
    activation_fn : Union[nn.Module, Callable[[Tensor], Tensor], None], optional, default=None
        Activation function to use. Can be ``None`` for no activation.
    activation_par : Union[nn.Parameter, None], optional, default=None
        Learnable scaling parameter for adaptive activations.
    weight_norm : bool, optional, default=False
        Weight normalization (not currently supported, raises ``NotImplementedError``).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, L)` where :math:`B` is batch size
        and :math:`L` is sequence length.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, L)`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
        weight_norm: bool = False,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.reset_parameters()

        if weight_norm:
            raise NotImplementedError("Weight norm not supported for Conv FC layers")

    def reset_parameters(self) -> None:
        """Reset layer weights to Xavier uniform initialization."""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the 1D convolutional layer."""
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):
    r"""Channel-wise fully connected layer using 2D convolutions.

    Applies a 1x1 convolution followed by an optional activation function.
    This is equivalent to a fully connected layer operating on the channel
    dimension of 2D images.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    activation_fn : Union[nn.Module, Callable[[Tensor], Tensor], None], optional, default=None
        Activation function to use. Can be ``None`` for no activation.
    activation_par : Union[nn.Parameter, None], optional, default=None
        Learnable scaling parameter for adaptive activations.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch size,
        :math:`H` is height, and :math:`W` is width.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer weights to Xavier uniform initialization."""
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the 2D convolutional layer."""
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):
    r"""Channel-wise fully connected layer using 3D convolutions.

    Applies a 1x1x1 convolution followed by an optional activation function.
    This is equivalent to a fully connected layer operating on the channel
    dimension of 3D volumes.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    activation_fn : Union[nn.Module, Callable[[Tensor], Tensor], None], optional, default=None
        Activation function to use. Can be ``None`` for no activation.
    activation_par : Union[nn.Parameter, None], optional, default=None
        Learnable scaling parameter for adaptive activations.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, D, H, W)` where :math:`B` is batch
        size, :math:`D` is depth, :math:`H` is height, and :math:`W` is width.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, D, H, W)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[nn.Module, Callable[[Tensor], Tensor], None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer weights to Xavier uniform initialization."""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the 3D convolutional layer."""
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class ConvNdFCLayer(ConvFCLayer):
    r"""Channel-wise fully connected layer with N-dimensional convolutions.

    Applies a kernel-1 convolution followed by an optional activation function.
    For dimensions 1, 2, or 3, use :class:`Conv1dFCLayer`, :class:`Conv2dFCLayer`,
    or :class:`Conv3dFCLayer` instead for better performance.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    activation_fn : Union[nn.Module, None], optional, default=None
        Activation function to use. Can be ``None`` for no activation.
    activation_par : Union[nn.Parameter, None], optional, default=None
        Learnable scaling parameter for adaptive activations.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, *spatial)` where :math:`B` is
        batch size and :math:`*spatial` represents arbitrary spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *spatial)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[nn.Module, None] = None,
        activation_par: Union[nn.Parameter, None] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = ConvNdKernel1Layer(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer weights by recursively applying Xavier initialization."""
        self.conv.apply(self.initialise_parameters)

    def initialise_parameters(self, model: nn.Module) -> None:
        """Initialize weights and biases for a module.

        Parameters
        ----------
        model : nn.Module
            Module to initialize.
        """
        if hasattr(model, "bias"):
            nn.init.constant_(model.bias, 0)
        if hasattr(model, "weight"):
            nn.init.xavier_uniform_(model.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the N-dimensional convolutional layer."""
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class ConvNdKernel1Layer(Module):
    r"""Kernel-1 convolution layer for N-dimensional inputs.

    Implements a 1x1 convolution by reshaping the input to 1D, applying
    a 1D convolution, and reshaping back. For dimensions 1, 2, or 3, use
    the specialized layer classes for better performance.

    Parameters
    ----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, *spatial)` where :math:`B` is
        batch size and :math:`*spatial` represents arbitrary spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, *spatial)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the N-dimensional kernel-1 convolution."""
        dims = list(x.size())
        dims[1] = self.out_channels
        x = self.conv(x.view(dims[0], self.in_channels, -1)).view(dims)
        return x


class Linear(Module):
    r"""Fully connected (dense) layer with customizable initialization.

    The layer's weights and biases can be initialized using custom strategies
    like ``"kaiming_normal"``, and scaled by ``init_weight`` and ``init_bias``.

    Parameters
    ----------
    in_features : int
        Size of each input sample :math:`D_{in}`.
    out_features : int
        Size of each output sample :math:`D_{out}`.
    bias : bool, optional, default=True
        If ``True``, adds a learnable bias to the output. If ``False``, the layer
        will not learn an additive bias.
    init_mode : str, optional, default="kaiming_normal"
        The initialization mode for weights and biases. Supported modes are:
        ``"xavier_uniform"``, ``"xavier_normal"``, ``"kaiming_uniform"``,
        ``"kaiming_normal"``.
    init_weight : float, optional, default=1
        A scaling factor to multiply with the initialized weights.
    init_bias : float, optional, default=0
        A scaling factor to multiply with the initialized biases.
    amp_mode : bool, optional, default=False
        Whether mixed-precision (AMP) training is enabled.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D_{in})` where :math:`*` denotes any
        number of leading batch dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(*, D_{out})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: Literal[
            "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
        ] = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
        amp_mode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.amp_mode = amp_mode
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            _weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear layer."""
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
        x = x @ weight.t()
        if self.bias is not None:
            x = x.add_(bias)
        return x
