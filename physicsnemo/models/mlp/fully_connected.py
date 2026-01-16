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

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.nn import FCLayer, get_activation


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    # Inference
    onnx: bool = True
    onnx_runtime: bool = True
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class FullyConnected(Module):
    r"""A densely-connected MLP architecture.

    This model constructs a multi-layer perceptron with configurable depth,
    width, activation functions, and optional skip connections. It uses
    :class:`~physicsnemo.nn.FCLayer` for each hidden layer.

    Parameters
    ----------
    in_features : int, optional, default=512
        Size of input features :math:`D_{in}`.
    layer_size : int, optional, default=512
        Size of every hidden layer :math:`D_{hidden}`.
    out_features : int, optional, default=512
        Size of output features :math:`D_{out}`.
    num_layers : int, optional, default=6
        Number of hidden layers.
    activation_fn : Union[str, List[str]], optional, default="silu"
        Activation function to use. Can be a single string or a list of strings
        (one per layer). Supported values include ``"silu"``, ``"relu"``, ``"gelu"``.
    skip_connections : bool, optional, default=False
        Add skip connections every 2 hidden layers.
    adaptive_activations : bool, optional, default=False
        Use an adaptive activation function with learnable scaling parameter.
    weight_norm : bool, optional, default=False
        Use weight normalization on fully connected layers.
    weight_fact : bool, optional, default=False
        Use weight factorization on fully connected layers.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})` where :math:`B` is the batch size.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> import physicsnemo.models.mlp
    >>> model = physicsnemo.models.mlp.FullyConnected(in_features=32, out_features=64)
    >>> x = torch.randn(128, 32)
    >>> output = model(x)
    >>> output.shape
    torch.Size([128, 64])
    """

    def __init__(
        self,
        in_features: int = 512,
        layer_size: int = 512,
        out_features: int = 512,
        num_layers: int = 6,
        activation_fn: Union[str, List[str]] = "silu",
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = False,
        weight_fact: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())

        self.in_features = in_features
        self.out_features = out_features
        self.skip_connections = skip_connections

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * num_layers
        if len(activation_fn) < num_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                num_layers - len(activation_fn)
            )
        activation_fn = [get_activation(a) for a in activation_fn]

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(num_layers):
            self.layers.append(
                FCLayer(
                    layer_in_features,
                    layer_size,
                    activation_fn[i],
                    weight_norm,
                    weight_fact,
                    activation_par,
                )
            )
            layer_in_features = layer_size

        self.final_layer = FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=None,
            weight_norm=False,
            weight_fact=False,
            activation_par=None,
        )

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        """Forward pass through the MLP."""
        # Validate input shape
        if not torch.compiler.is_compiling():
            if x.ndim < 2:
                raise ValueError(
                    f"Expected input tensor with at least 2 dimensions, "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input with {self.in_features} features (last dimension), "
                    f"got {x.shape[-1]} in tensor with shape {tuple(x.shape)}"
                )

        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x
