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

# ruff: noqa: F722
from typing import List

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class LinearBlock(nn.Module):
    """Simple linear block with ReLU and dropout

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    activation : type[nn.Module]
        Activation function, default nn.GELU
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
        return self.block(x)


class ResidualLinearBlock(nn.Module):
    """MLPBlock."""

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
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels)
        )
        self.activation = activation()

    def forward(self, x):
        out = self.blocks(x)
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out


class MLP(nn.Module):
    """Multi-layer perceptron

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    hidden_channels : int
        Number of inernal channels in the MLP.
    use_residual : bool, optional
        Whether to use residual connections, default False.
    activation : type[nn.Module]
        Activation function, default nn.GELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        use_residual: bool = False,
        activation: type[nn.Module] = nn.GELU,
    ):
        """
        :param channels: list of channels
        :param dropout: dropout rate
        """
        super().__init__()

        self.layers = nn.ModuleList()
        channels = [in_channels] + hidden_channels + [out_channels]
        for i in range(len(channels) - 1):
            if use_residual and i < len(channels) - 2:
                self.layers.append(
                    ResidualLinearBlock(
                        channels[i],
                        channels[i + 1],
                        activation=activation,
                    )
                )
            else:
                self.layers.append(
                    LinearBlock(channels[i], channels[i + 1], activation=activation)
                )

    def forward(self, x: Float[Tensor, "... C1"]) -> Float[Tensor, "... C2"]:
        """
        Forward pass
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MLPBlock(nn.Module):
    """MLPBlock."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out
