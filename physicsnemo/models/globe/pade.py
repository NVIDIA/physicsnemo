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
from physicsnemo.models.globe.mlp import MLP


class Pade(Module):
    r"""Padé-approximant neural network for rational function learning.

    Implements a rational neural network of the form

    .. math::

        f(x) = \frac{\operatorname{sgn}(\varphi_n(x))\;
        |\varphi_n(x)|^{N}}{1 + |\varphi_d(x)|^{D}}

    where :math:`\varphi_n` and :math:`\varphi_d` are learnable MLPs
    (numerator and denominator sub-networks), and :math:`N, D` are the
    numerator and denominator orders. This rational structure provides
    strong inductive bias for approximating Green's functions and other
    physics kernels that exhibit algebraic singularities near sources and
    algebraic decay at infinity.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        Number of units in each layer, including input and output.
        For example, ``[8, 32, 16, 4]`` creates a network with input
        dimension 8, two hidden layers, and output dimension 4.
    activation_function : Callable[[torch.Tensor], torch.Tensor] | None, optional, default=None
        Activation applied after each hidden layer. Should satisfy
        ``f(x) -> 0`` as ``x -> -inf`` and ``f(x) -> x`` as
        ``x -> +inf`` for proper asymptotic behavior (e.g.,
        ``nn.SiLU()``, ``nn.Mish()``). When ``None``, defaults to
        ``nn.SiLU()``.
    dropout : float, optional, default=0.0
        Dropout probability.
    use_batchnorm : bool, optional, default=False
        Whether to apply ``BatchNorm1d``.
    spectral_norm : bool, optional, default=False
        Whether to apply spectral normalization.
    numerator_order : int, optional, default=1
        Power to raise the numerator to.
    denominator_order : int, optional, default=2
        Power to raise the denominator to.
    share_denominator_across_channels : bool, optional, default=True
        If ``True``, uses a single scalar denominator for all output
        channels.
    use_separate_mlps : bool, optional, default=True
        If ``True``, uses separate MLPs for numerator and denominator.
        If ``False``, uses a single MLP with split outputs.

    Forward
    -------
    x : Float[torch.Tensor, "batch input_dim"]
        Input tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    Float[torch.Tensor, "batch output_dim"]
        Padé-approximant output of shape :math:`(B, D_{out})` where
        :math:`D_{out}` is ``layer_sizes[-1]``.

    Notes
    -----
    When :math:`N = D` (default 2/2), the function asymptotes to a
    constant in any far-field direction, ideal for bounded far-field
    decay. The denominator :math:`1 + |\varphi_d|^D \geq 1` prevents
    singularities and limits the Lipschitz constant.

    Examples
    --------
    >>> pade = Pade([10, 64, 64, 3], numerator_order=2, denominator_order=2)
    >>> x = torch.randn(32, 10)
    >>> y = pade(x)
    >>> y.shape
    torch.Size([32, 3])
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        spectral_norm: bool = False,
        numerator_order: int = 1,
        denominator_order: int = 2,
        share_denominator_across_channels: bool = True,
        use_separate_mlps: bool = True,
    ):
        if activation_function is None:
            activation_function = nn.SiLU()

        super().__init__()

        self.register_buffer("_one", torch.tensor(1.0))

        ### Store constructor arguments
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.spectral_norm = spectral_norm
        self.numerator_order = numerator_order
        self.denominator_order = denominator_order
        self.share_denominator_across_channels = share_denominator_across_channels
        self.use_separate_mlps = use_separate_mlps

        ### Create the MLPs
        if use_separate_mlps:
            numerator_layer_sizes = list(layer_sizes)[:-1] + [
                self.numerator_output_size
            ]
            denominator_layer_sizes = list(layer_sizes)[:-1] + [
                self.denominator_output_size
            ]

            self.numerator_mlp = MLP(
                layer_sizes=numerator_layer_sizes,
                activation_function=activation_function,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                spectral_norm=spectral_norm,
            )
            self.denominator_mlp = MLP(
                layer_sizes=denominator_layer_sizes,
                activation_function=activation_function,
                bias=False,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                spectral_norm=spectral_norm,
            )
        else:
            combined_output_size = (
                self.numerator_output_size + self.denominator_output_size
            )
            combined_layer_sizes = list(layer_sizes)[:-1] + [combined_output_size]

            self.combined_mlp = MLP(
                layer_sizes=combined_layer_sizes,
                activation_function=activation_function,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                spectral_norm=spectral_norm,
            )

    @property
    def numerator_output_size(self) -> int:
        r"""Number of numerator output channels (equals ``layer_sizes[-1]``).

        Returns
        -------
        int
            Output dimension of the numerator MLP.
        """
        return self.layer_sizes[-1]

    @property
    def denominator_output_size(self) -> int:
        r"""Number of denominator output channels.

        Returns ``1`` if ``share_denominator_across_channels`` is ``True``,
        otherwise ``layer_sizes[-1]``.

        Returns
        -------
        int
            Output dimension of the denominator MLP.
        """
        return 1 if self.share_denominator_across_channels else self.layer_sizes[-1]

    def forward(
        self, x: Float[torch.Tensor, "batch input_dim"]
    ) -> Float[torch.Tensor, "batch output_dim"]:
        r"""Evaluate the Padé-approximant network.

        Computes

        .. math::

            f(x) = \frac{\operatorname{sgn}(\varphi_n(x))\;
            |\varphi_n(x)|^{N}}{1 + |\varphi_d(x)|^{D}}

        Parameters
        ----------
        x : Float[torch.Tensor, "batch input_dim"]
            Input tensor of shape :math:`(B, D_{in})`.

        Returns
        -------
        Float[torch.Tensor, "batch output_dim"]
            Rational-function output of shape :math:`(B, D_{out})`.
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D input (B, D_in), got {x.ndim}D tensor "
                    f"with shape {tuple(x.shape)}"
                )

        if self.use_separate_mlps:
            raw_numerator: torch.Tensor | float = (
                self._one if self.numerator_order == 0 else self.numerator_mlp(x)
            )
            raw_denominator: torch.Tensor | float = (
                self._one if self.denominator_order == 0 else self.denominator_mlp(x)
            )
        else:
            raw_numerator, raw_denominator = self.combined_mlp(x).split(
                self.numerator_output_size, dim=-1
            )

        def apply_power(x: torch.Tensor, order: int, even: bool) -> torch.Tensor:
            if even:
                return x.abs().pow(order)
            else:
                return x.sign() * x.abs().pow(order)

        numerator = apply_power(raw_numerator, self.numerator_order, even=False)
        denominator = apply_power(raw_denominator, self.denominator_order, even=True)

        # Compute Padé approximant: numerator / (1 + denominator)
        return numerator / (1 + denominator)
