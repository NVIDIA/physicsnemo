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

"""GALE_FA (Geometry-Aware Latent Embeddings with FLARE self-Attention) attention layer.

This module provides the GALE_FA attention mechanism, 
an alternative to the GALE attention mechanism of the GeoTransolver.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.nn import ConcreteDropout
from physicsnemo.nn.module.physics_attention import _project_input
from physicsnemo.experimental.nn.flare_attention import _flare_self_attention
from physicsnemo.experimental.models.geotransolver.gale import (
    _gale_cross_init,
    _mix_self_and_cross,
)

te = OptionalImport("transformer_engine.pytorch", "0.1.0")


class GALE_FA(nn.Module):
    r"""GALE_FA: Geometry-Aware Latent Embeddings with FLARE self-Attention attention layer.
    Adopted:
    - FLARE attention: Fast Low-rank Attention Routing Engine
        paper: https://arxiv.org/abs/2508.12594
    - GeoTransolver context:
        paper: https://arxiv.org/abs/2512.20399

    GALE_FA is an alternative to the GALE attention mechanism of the GeoTransolver 
    It supports cross-attention with a context vector, built from geometry and global embeddings.
    GALE_FA combines FLARE self-attention on learned physical state slices with cross-attention
    to geometry-aware context, using a learnable mixing weight to blend the two.

    Parameters
    ----------
    dim : int
        Input dimension of the features.
    heads : int, optional
        Number of attention heads. Default is 8.
    dim_head : int, optional
        Dimension of each attention head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    n_global_queries : int, optional
        Number of learned global queries. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is False.
    context_dim : int, optional
        Dimension of the context vector for cross-attention. Default is 0.
    concrete_dropout : bool, optional
        Whether to use learned concrete dropout instead of standard dropout.
        Default is ``False``.
    state_mixing_mode : str, optional
        How to blend self-attention and cross-attention outputs.         ``"weighted"`` uses
        a learnable sigmoid-gated weighted sum. ``"concat_project"``
        concatenates the two along the head dimension and projects back with a
        linear layer. Default is ``"weighted"``.

    Forward
    -------
    x : tuple[torch.Tensor, ...]
        Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B` is
        batch size, :math:`N` is number of tokens, and :math:`C` is number of channels.
    context : tuple[torch.Tensor, ...] | None, optional
        Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)` where
        :math:`H` is number of heads, :math:`S_c` is number of context slices, and
        :math:`D_c` is context dimension. If ``None``, only self-attention is applied.
        Default is ``None``.

    Outputs
    -------
    list[torch.Tensor]
        List of output tensors, each of shape :math:`(B, N, C)`, same shape as inputs.

    Notes
    -----
    The mixing between self-attention and cross-attention is controlled by a learnable
    parameter ``state_mixing`` which is passed through a sigmoid function to ensure
    the mixing weight stays in :math:`[0, 1]`.

    See Also
    --------
    :class:`GALE` : Original GeoTransolver GALE attention class.
    :class:`GALE_block` : Transformer block that calls GALE or GALE_FA attention.

    Examples
    --------
    >>> import torch
    >>> gale_fa = GALE_FA(dim=256, heads=8, dim_head=32, context_dim=32)
    >>> x = (torch.randn(2, 100, 256),)  # Single input tensor in tuple
    >>> context = torch.randn(2, 8, 64, 32)  # Context for cross-attention
    >>> outputs = gale_fa(x, context)
    >>> len(outputs)
    1
    >>> outputs[0].shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        n_global_queries: int = 64,
        use_te: bool = True,
        context_dim: int = 0,
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
    ):
        if use_te:
            raise ValueError(
                "GALE_FA does not support Transformer Engine backend. "
                "Use use_te=False; TE disables FlashAttention for differing q/k sizes in FLARE attention."
            )
        super().__init__()
        self.use_te = use_te
        self.heads = heads
        self.dim_head = dim_head
        self.scale = 1.0
        # It is recommended by the FLARE authors to use self.scale = 1 if self.dim_head <= 8 else (self.dim_head ** -0.5)
        # but we use self.scale = 1.0 because the recommended scaling is not tested yet.
        inner_dim = dim_head * heads

        linear_layer = te.Linear if self.use_te else nn.Linear

        # Global queries for FLARE self-attention
        self.q_global = nn.Parameter(torch.randn(1, heads, n_global_queries, dim_head))

        # Linear projections for self-attention
        self.in_project_x = linear_layer(dim, inner_dim)
        self.self_k = linear_layer(dim_head, dim_head)
        self.self_v = linear_layer(dim_head, dim_head)

        if context_dim > 0:
            _gale_cross_init(self, dim_head, context_dim, use_te, state_mixing_mode)

        # Linear projection for output
        self.out_linear = linear_layer(inner_dim, dim)
        if concrete_dropout:
            self.out_dropout = ConcreteDropout(
                in_features=dim,
                init_p=max(dropout, 0.05),
            )
        else:
            self.out_dropout = nn.Dropout(dropout)


    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        r"""Forward pass of the GALE_FA module.

        Applies GALE_FA attention to the input features.

        Parameters
        ----------
        x : tuple[torch.Tensor, ...]
            Tuple of input tensors, each of shape :math:`(B, N, C)` where :math:`B`
            is batch size, :math:`N` is number of tokens, and :math:`C` is number
            of channels.
        context : torch.Tensor | None, optional
            Context tensor for cross-attention of shape :math:`(B, H, S_c, D_c)`
            where :math:`H` is number of heads, :math:`S_c` is number of context
            slices, and :math:`D_c` is context dimension. If ``None``, only
            self-attention is applied. Default is ``None``.

        Returns
        -------
        list[torch.Tensor]
            List of output tensors, each of shape :math:`(B, N, C)``, same shape
            as inputs.
        """

        # Input projection: (B, N, C) -> (B, N, H, D) -> (B, H, N, D)
        x_mid = [
            _project_input(
                _x, self.in_project_x, self.heads, self.dim_head,
                "B N (H D) -> B N H D",
            ).permute(0, 2, 1, 3)
            for _x in x
        ]

        # FLARE self-attention per input
        self_attention = [
            _flare_self_attention(
                _x_mid, self.q_global, self.self_k, self.self_v, self.scale,
            )
            for _x_mid in x_mid
        ]

        # Cross-attention with context and state mixing
        if context is not None:
            q = [self.cross_q(_x_mid) for _x_mid in x_mid]
            k = self.cross_k(context)
            v = self.cross_v(context)
            cross_attention = [
                F.scaled_dot_product_attention(_q, k, v, scale=self.scale)
                for _q in q
            ]
            outputs = [
                _mix_self_and_cross(
                    sa, ca, self.state_mixing_mode,
                    state_mixing=getattr(self, "state_mixing", None),
                    concat_project=getattr(self, "concat_project", None),
                )
                for sa, ca in zip(self_attention, cross_attention)
            ]
        else:
            outputs = self_attention

        # Back to token layout: (B, H, N, D) -> (B, N, H, D)
        outputs = [_y.permute(0, 2, 1, 3) for _y in outputs]
        outputs = [rearrange(_out, "b n h d -> b n (h d)") for _out in outputs]
        outputs = [self.out_linear(_out) for _out in outputs]
        return [self.out_dropout(_out) for _out in outputs]

