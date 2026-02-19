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

from typing import Callable, Sequence

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.module import Module


class MLP(Module):
    r"""Multi-layer perceptron with configurable architecture.

    Constructs a fully-connected feedforward neural network with optional
    batch normalization, dropout, spectral normalization, and a user-specified
    activation function. The architecture is defined by a sequence of layer
    sizes, where each entry specifies the number of neurons in that layer.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        Number of units in each layer, including input and output.
        For example, ``[8, 32, 16, 4]`` creates an MLP with input
        dimension 8, two hidden layers of sizes 32 and 16, and output
        dimension 4.
    activation_function : Callable[[torch.Tensor], torch.Tensor] | None
        Activation applied after each hidden layer. Can be any
        ``torch.nn`` module or callable. Defaults to ``nn.SiLU()``.
    bias : bool
        Whether to include a bias term in the linear layers.
        Default ``True``.
    dropout : float
        Dropout probability after each activation (except the last layer).
        Set to ``0.0`` to disable. Default ``0.0``.
    use_batchnorm : bool
        If ``True``, applies ``BatchNorm1d`` after each linear layer
        (except the last). Default ``False``.
    spectral_norm : bool
        If ``True``, applies spectral normalization to all linear layer
        weights, constraining the spectral norm to 1. Default ``False``.

    Forward
    -------
    x : Float[torch.Tensor, "batch input_dim"]
        Input tensor of shape :math:`(B, D_{in})` where :math:`D_{in}`
        is ``layer_sizes[0]``.

    Outputs
    -------
    Float[torch.Tensor, "batch output_dim"]
        Output tensor of shape :math:`(B, D_{out})` where :math:`D_{out}`
        is ``layer_sizes[-1]``.

    Notes
    -----
    No activation, dropout, or batch normalization is applied after the
    final output layer.

    Examples
    --------
    >>> mlp = MLP([10, 64, 32, 3], activation_function=nn.ReLU(), dropout=0.1)
    >>> x = torch.randn(5, 10)
    >>> y = mlp(x)
    >>> y.shape
    torch.Size([5, 3])
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        spectral_norm: bool = False,
    ):
        if activation_function is None:
            activation_function = nn.SiLU()

        super().__init__()

        ### Store constructor arguments
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.bias = bias
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.spectral_norm = spectral_norm

        ### Build layers
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                bias=bias,
            )
            if spectral_norm:
                linear_layer = nn.utils.parametrizations.spectral_norm(
                    module=linear_layer,
                    name="weight",
                )

            layers.append(linear_layer)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation/dropout after last layer
                layers.append(activation_function)
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(
        self, x: Float[torch.Tensor, "batch input_dim"]
    ) -> Float[torch.Tensor, "batch output_dim"]:
        r"""Forward pass through all layers.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch input_dim"]
            Input tensor of shape :math:`(B, D_{in})`.

        Returns
        -------
        Float[torch.Tensor, "batch output_dim"]
            Output tensor of shape :math:`(B, D_{out})`.
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D input (B, D_in), got {x.ndim}D tensor "
                    f"with shape {tuple(x.shape)}"
                )
            if x.shape[-1] != self.layer_sizes[0]:
                raise ValueError(
                    f"Expected input dim {self.layer_sizes[0]}, got {x.shape[-1]}"
                )

        return self.layers(x)
