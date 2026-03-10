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

import itertools

import torch
from torch import nn

from physicsnemo.core.version_check import OptionalImport

from .activations import get_activation

# Check for Transformer Engine availability
te = OptionalImport("transformer_engine.pytorch")


class Mlp(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Supports arbitrary depth, dropout, batch normalization, spectral
    normalization, bias control, and optional Transformer Engine linear
    layers.

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
    act_layer : nn.Module | type[nn.Module] | str, optional
        Activation function. Can be:
        - ``str``: Name of activation (e.g., ``"gelu"``, ``"relu"``, ``"silu"``)
        - ``type``: Activation class to instantiate (e.g., ``nn.GELU``)
        - ``nn.Module``: Pre-instantiated activation module
        Default is ``nn.GELU``.
    drop : float, optional
        Dropout rate applied after each hidden layer. Default is ``0.0``.
    final_dropout : bool, optional
        Whether to apply dropout after the final linear layer. Default is ``True``.
    bias : bool, optional
        Whether to include bias terms in the linear layers. Default is ``True``.
    use_batchnorm : bool, optional
        If ``True``, applies ``BatchNorm1d`` after each linear layer
        (including the output layer). Default is ``False``.
    spectral_norm : bool, optional
        If ``True``, applies spectral normalization to all linear layer
        weights, constraining the spectral norm to 1. Default is ``False``.
    use_te : bool, optional
        Whether to use Transformer Engine linear layers for optimized performance.
        Requires Transformer Engine to be installed. Default is ``False``.

    Examples
    --------
    >>> import torch
    >>> mlp = Mlp(in_features=64, hidden_features=128, out_features=32)
    >>> x = torch.randn(2, 64)
    >>> out = mlp(x)
    >>> out.shape
    torch.Size([2, 32])

    >>> mlp = Mlp(in_features=64, hidden_features=[128, 256, 128], out_features=32)
    >>> x = torch.randn(2, 64)
    >>> out = mlp(x)
    >>> out.shape
    torch.Size([2, 32])

    >>> # With batch normalization and spectral normalization
    >>> mlp = Mlp(
    ...     in_features=10,
    ...     hidden_features=[32, 16],
    ...     out_features=4,
    ...     use_batchnorm=True,
    ...     spectral_norm=True,
    ... )
    >>> mlp(torch.randn(8, 10)).shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | list[int] | None = None,
        out_features: int | None = None,
        act_layer: nn.Module | type[nn.Module] | str = nn.GELU,
        drop: float = 0.0,
        final_dropout: bool = True,
        bias: bool = True,
        use_batchnorm: bool = False,
        spectral_norm: bool = False,
        use_te: bool = False,
    ):
        super().__init__()

        self.use_te = use_te

        out_features = out_features or in_features

        if hidden_features is None:
            hidden_features = [in_features]
        elif isinstance(hidden_features, int):
            hidden_features = [hidden_features]

        ### Resolve activation
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

        linear_layer = te.Linear if use_te else nn.Linear

        ### Build layers
        layers: list[nn.Module] = []
        dims = [in_features, *hidden_features, out_features]
        n_layers = len(dims) - 1

        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            is_last = i == n_layers - 1
            linear = linear_layer(in_dim, out_dim, bias=bias)
            if spectral_norm:
                linear = nn.utils.parametrizations.spectral_norm(linear, name="weight")
            layers.append(linear)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            if not is_last:
                layers.append(act_layer)
            if drop != 0 and (not is_last or final_dropout):
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
