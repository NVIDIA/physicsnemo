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

from typing import Callable, Union

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.nn.utils.utils import _validate_amp
from physicsnemo.nn.utils.weight_init import _weight_init

from .activations import Identity
from .weight_fact import WeightFactLinear
from .weight_norm import WeightNormLinear


class FCLayer(nn.Module):
    """Densely connected NN layer

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    weight_norm : bool, optional
        Applies weight normalization to the layer, by default False
    weight_fact : bool, optional
        Applies weight factorization to the layer, by default False
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
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
        """Reset fully connected weights"""
        if not self.weight_norm and not self.weight_fact:
            nn.init.constant_(self.linear.bias, 0)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)

        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)

        return x


class ConvFCLayer(nn.Module):
    """Base class for 1x1 Conv layer for image channels

    Parameters
    ----------
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
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
        """Applied activation / learnable activations

        Parameters
        ----------
        x : Tensor
            Input tensor
        """
        if self.activation_par is None:
            x = self.activation_fn(x)
        else:
            x = self.activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 1d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
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
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 2d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
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
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with 3d convolutions

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
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
        """Reset layer weights"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class ConvNdFCLayer(ConvFCLayer):
    """Channel-wise FC like layer with convolutions of arbitrary dimensions
    CAUTION: if n_dims <= 3, use specific version for that n_dims instead

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
    activation_fn : Union[nn.Module, None], optional
        Activation function to use. Can be None for no activation, by default None
    activation_par : Union[nn.Parameter, None], optional
        Additional parameters for the activation function, by default None
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

    def reset_parameters(self):
        self.conv.apply(self.initialise_parameters)  # recursively apply initialisations

    def initialise_parameters(self, model):
        """Reset layer weights"""
        if hasattr(model, "bias"):
            nn.init.constant_(model.bias, 0)
        if hasattr(model, "weight"):
            nn.init.xavier_uniform_(model.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class ConvNdKernel1Layer(nn.Module):
    """Channel-wise FC like layer for convolutions of arbitrary dimensions
    CAUTION: if n_dims <= 3, use specific version for that n_dims instead

    Parameters
    ----------
    in_features : int
        Size of input features
    out_features : int
        Size of output features
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
        dims = list(x.size())
        dims[1] = self.out_channels
        x = self.conv(x.view(dims[0], self.in_channels, -1)).view(dims)
        return x


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
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

    def forward(self, x):
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
