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

r"""Multi-layer perceptron components for FIGConvNet.

This module provides MLP building blocks used throughout the FIGConvNet
architecture for feature transformation and projection.

The main classes are:

- :class:`LinearBlock`: Single linear layer with normalization and activation
- :class:`ResidualLinearBlock`: Linear block with residual connection
- :class:`MLP`: Multi-layer perceptron with configurable depth
- :class:`MLPBlock`: Two-layer MLP block with residual connection
"""

# ruff: noqa: F722
from typing import List

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class LinearBlock(nn.Module):
    r"""Single linear layer with layer normalization and activation.

    A simple building block consisting of:
    1. Linear projection (without bias)
    2. Layer normalization
    3. Activation function

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : type[nn.Module], optional, default=nn.GELU
        Activation function class.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(..., C_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(..., C_{out})`.

    Examples
    --------
    >>> import torch
    >>> block = LinearBlock(64, 128)
    >>> x = torch.randn(4, 100, 64)
    >>> out = block(x)
    >>> out.shape
    torch.Size([4, 100, 128])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            activation(),
        )

    def forward(self, x: Float[Tensor, "... C1"]) -> Float[Tensor, "... C2"]:
        r"""Apply linear block transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(..., C_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(..., C_{out})`.
        """
        return self.block(x)


class ResidualLinearBlock(nn.Module):
    r"""Linear block with residual connection.

    Applies a two-layer MLP with a skip connection:

    .. math::

        \text{output} = \text{activation}(\text{norm}_2(\text{fc}_2(\text{activation}(\text{norm}_1(\text{fc}_1(x))))) + \text{shortcut}(x))

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int, optional, default=None
        Number of hidden channels. Defaults to ``in_channels`` if not specified.
    activation : type[nn.Module], optional, default=nn.GELU
        Activation function class.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(..., C_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(..., C_{out})`.

    Examples
    --------
    >>> import torch
    >>> block = ResidualLinearBlock(64, 64)
    >>> x = torch.randn(4, 100, 64)
    >>> out = block(x)
    >>> out.shape
    torch.Size([4, 100, 64])

    Note
    ----
    The shortcut is an identity mapping when ``in_channels == out_channels``,
    otherwise a linear projection is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = None,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels

        self.blocks = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            activation(),
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
        )

        # Shortcut: identity if same dimensions, otherwise linear projection
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels)
        )
        self.activation = activation()

    def forward(self, x: Float[Tensor, "... C1"]) -> Float[Tensor, "... C2"]:
        r"""Apply residual block transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(..., C_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(..., C_{out})`.
        """
        out = self.blocks(x)
        # Add skip connection and apply activation
        out = self.activation(out + self.shortcut(x))
        return out


class MLP(nn.Module):
    r"""Multi-layer perceptron with configurable architecture.

    A flexible MLP that supports:
    - Arbitrary number of hidden layers
    - Optional residual connections
    - Configurable activation function

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : List[int]
        List of hidden layer sizes. An empty list creates a single-layer MLP.
    use_residual : bool, optional, default=False
        Whether to use residual connections in hidden layers.
    activation : type[nn.Module], optional, default=nn.GELU
        Activation function class.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(..., C_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(..., C_{out})`.

    Examples
    --------
    >>> import torch
    >>> # Create a 3-layer MLP: 64 -> 128 -> 256 -> 32
    >>> mlp = MLP(64, 32, hidden_channels=[128, 256])
    >>> x = torch.randn(4, 100, 64)
    >>> out = mlp(x)
    >>> out.shape
    torch.Size([4, 100, 32])

    >>> # With residual connections
    >>> mlp_res = MLP(64, 64, hidden_channels=[64, 64], use_residual=True)
    >>> out_res = mlp_res(x)
    >>> out_res.shape
    torch.Size([4, 100, 64])

    Note
    ----
    When ``use_residual=True``, residual connections are applied to all layers
    except the final output layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        use_residual: bool = False,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Build channel progression: [in_channels, *hidden_channels, out_channels]
        channels = [in_channels] + hidden_channels + [out_channels]

        for i in range(len(channels) - 1):
            if use_residual and i < len(channels) - 2:
                # Use residual blocks for all but the last layer
                self.layers.append(
                    ResidualLinearBlock(
                        channels[i],
                        channels[i + 1],
                        activation=activation,
                    )
                )
            else:
                # Use simple linear blocks
                self.layers.append(
                    LinearBlock(channels[i], channels[i + 1], activation=activation)
                )

    def forward(self, x: Float[Tensor, "... C1"]) -> Float[Tensor, "... C2"]:
        r"""Apply MLP transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(..., C_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(..., C_{out})`.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MLPBlock(nn.Module):
    r"""Two-layer MLP block with residual connection.

    A compact MLP block consisting of:
    1. Linear projection to hidden dimension
    2. Layer normalization and activation
    3. Linear projection to output dimension
    4. Layer normalization
    5. Residual connection from input
    6. Final activation

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int, optional, default=None
        Number of hidden channels. Defaults to ``in_channels``.
    out_channels : int, optional, default=None
        Number of output channels. Defaults to ``in_channels``.
    activation : type[nn.Module], optional, default=nn.GELU
        Activation function class.

    Attributes
    ----------
    in_channels : int
        Number of input channels.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(..., C_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(..., C_{out})`.

    Examples
    --------
    >>> import torch
    >>> block = MLPBlock(in_channels=64, hidden_channels=128, out_channels=64)
    >>> x = torch.randn(4, 100, 64)
    >>> out = block(x)
    >>> out.shape
    torch.Size([4, 100, 64])

    Note
    ----
    This block always includes a residual connection. When input and output
    dimensions differ, a linear projection is applied to the shortcut.

    See Also
    --------
    :class:`ResidualLinearBlock`
    :class:`MLP`
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        # Default hidden and output channels to input channels
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels

        # First layer: project to hidden dimension
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)

        # Second layer: project to output dimension
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # Shortcut: project input to output dimension if needed
        self.shortcut = nn.Linear(in_channels, out_channels)

        self.activation = activation()

    def forward(self, x: Float[Tensor, "... C1"]) -> Float[Tensor, "... C2"]:
        r"""Apply MLPBlock transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(..., C_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(..., C_{out})`.
        """
        # Main path: fc1 -> norm1 -> activation -> fc2 -> norm2
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))

        # Add skip connection and apply final activation
        out = self.activation(out + self.shortcut(x))

        return out
