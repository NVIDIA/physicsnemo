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

"""FLARE (Fast Low-rank Attention Routing Engine) attention layer.

This module provides the FLARE attention mechanism,
an alternative to the PhysicsAttention attention mechanism of the Transolver.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec, OptionalImport
from physicsnemo.nn.module.physics_attention import _project_input

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
te = OptionalImport("transformer_engine.pytorch", "0.1.0")


def _flare_self_attention(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    scale: float,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel.

    Computes low-rank attention via learned global queries: first aggregate
    token values into global slots, then distribute back to tokens.

    Parameters
    ----------
    x_mid : torch.Tensor
        Projected input of shape :math:`(B, H, N, D)`.
    q_global : nn.Parameter
        Learned global queries of shape :math:`(1, H, S, D)`.
    self_k : nn.Module
        Key projection applied to ``x_mid``.
    self_v : nn.Module
        Value projection applied to ``x_mid``.
    scale : float
        Attention scale factor.

    Returns
    -------
    torch.Tensor
        Self-attended output of shape :math:`(B, H, N, D)`.
    """
    G = q_global.to(dtype=x_mid.dtype).expand(x_mid.shape[0], -1, -1, -1)
    k = self_k(x_mid)
    v = self_v(x_mid)
    z = F.scaled_dot_product_attention(G, k, v, scale=scale)
    return F.scaled_dot_product_attention(k, G, z, scale=scale)


class FLARE(nn.Module):
    r"""FLARE: Fast Low-rank Attention Routing Engine attention layer.
    Adopted:
    - FLARE attention: Fast Low-rank Attention Routing Engine
        paper: https://arxiv.org/abs/2508.12594

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

    Forward
    -------
    x : torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])
    Outputs
    -------
    torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])

    Examples
    --------
    >>> import torch
    >>> flare = FLARE(dim=256, heads=8, dim_head=32)
    >>> x = torch.randn(2, 100, 256)
    >>> outputs = flare(x)
    >>> outputs.shape
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
    ):
        if use_te:
            raise ValueError(
                "FLARE does not support Transformer Engine backend. "
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

        # te attention
        if self.use_te:
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=dropout,
                qkv_format="bshd",
                softmax_scale=self.scale
            )

        # Linear projection for output
        self.out_linear = linear_layer(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Forward pass of the FLARE module.

        Applies FLARE attention to the input features.

        Parameters
        ----------
        x : torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])
            Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
            :math:`N` is number of points, and :math:`C` is number of channels.

        Returns
        -------
        torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])
            Output tensor of shape :math:`(B, N, C)`, same shape as inputs.
        """

        x_mid = _project_input(
            x, self.in_project_x, self.heads, self.dim_head,
            "B N (H D) -> B N H D",
        )
        x_mid = x_mid.permute(0, 2, 1, 3)  # (B, N, H, D) -> (B, H, N, D)

        y = _flare_self_attention(
            x_mid, self.q_global, self.self_k, self.self_v, self.scale,
        )

        out_x = y.permute(0, 2, 1, 3)  # (B, H, N, D) -> (B, N, H, D)
        out_x = rearrange(out_x, "b n h d -> b n (h d)")
        out_x = self.out_linear(out_x)
        return self.out_dropout(out_x)
