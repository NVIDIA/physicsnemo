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

"""Multi-layer perceptron (MLP) module with optional Transformer Engine support."""

import torch
from torch import nn

from physicsnemo.core.version_check import OptionalImport

from .activations import get_activation

# Check for Transformer Engine availability
te = OptionalImport("transformer_engine.pytorch")


class Mlp(nn.Module):
    """Multi-layer perceptron with configurable architecture and optional Transformer Engine support.

    A flexible MLP that supports:
    - Arbitrary number of hidden layers with configurable dimensions
    - Optional Transformer Engine linear layers for optimized performance
    - Configurable activation function
    - Optional dropout

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int | list[int] | None, optional
        Hidden layer dimension(s). Can be:
        - ``int``: Single hidden layer with this dimension
        - ``list[int]``: Multiple hidden layers with specified dimensions
        - ``None``: Single hidden layer with ``in_features`` dimension
        Default is ``None``.
    out_features : int | None, optional
        Number of output features. If ``None``, defaults to ``in_features``.
        Default is ``None``.
    act_layer : nn.Module | str, optional
        Activation function. Can be:
        - ``str``: Name of activation (e.g., ``"gelu"``, ``"relu"``, ``"silu"``)
        - ``type``: Activation class to instantiate (e.g., ``nn.GELU``)
        - ``nn.Module``: Pre-instantiated activation module
        Default is ``nn.GELU``.
    drop : float, optional
        Dropout rate applied after each layer. Default is ``0.0``.
    use_te : bool, optional
        Whether to use Transformer Engine linear layers for optimized performance.
        Requires Transformer Engine to be installed. Default is ``False``.

    Examples
    --------
    >>> import torch
    >>> # Simple MLP with single hidden layer
    >>> mlp = Mlp(in_features=64, hidden_features=128, out_features=32)
    >>> x = torch.randn(2, 64)
    >>> out = mlp(x)
    >>> out.shape
    torch.Size([2, 32])

    >>> # MLP with multiple hidden layers
    >>> mlp = Mlp(in_features=64, hidden_features=[128, 256, 128], out_features=32)
    >>> x = torch.randn(2, 64)
    >>> out = mlp(x)
    >>> out.shape
    torch.Size([2, 32])

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | list[int] | None = None,
        out_features: int | None = None,
        act_layer: nn.Module | type[nn.Module] | str = nn.GELU,
        drop: float = 0.0,
        use_te: bool = False,
    ):
        super().__init__()

        self.use_te = use_te

        # Set default output features
        out_features = out_features or in_features

        # Normalize hidden_features to list
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        elif hidden_features is None:
            hidden_features = [in_features]

        # Process activation layer
        # If it's a string, get the activation by name
        # If it's a type (class), instantiate it
        # If it's already an instance, use it directly
        if isinstance(act_layer, str):
            act_layer = get_activation(act_layer)
        elif isinstance(act_layer, nn.Module):
            pass
        else:
            act_layer = act_layer()
            if not isinstance(act_layer, nn.Module):
                raise ValueError(
                    f"Activation layer must be a string or a module, got {type(act_layer)}"
                )

        # Select linear layer type based on use_te
        linear_layer = te.Linear if use_te else nn.Linear

        # Build layers
        layers: list[nn.Module] = []
        input_dim = in_features

        for hidden_dim in hidden_features:
            layers.append(linear_layer(input_dim, hidden_dim))
            layers.append(act_layer)
            if drop != 0:
                layers.append(nn.Dropout(drop))
            input_dim = hidden_dim

        # Add the final output layer
        layers.append(linear_layer(input_dim, out_features))
        if drop != 0:
            layers.append(nn.Dropout(drop))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(*, in_features)`` where ``*`` denotes
            any number of batch dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(*, out_features)``.
        """
        return self.layers(x)
