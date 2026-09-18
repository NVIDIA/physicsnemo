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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.version_check import OptionalImport

from .physics_attention import _project_input

te = OptionalImport("transformer_engine.pytorch")


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


def _flare_self_attention_te(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    attn_fn: nn.Module,
    heads: int,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel on the Transformer Engine backend.

    Same computation as :func:`_flare_self_attention`, but the two attention
    passes run through a Transformer Engine ``DotProductAttention`` module.  Both
    passes are treated as cross-attention because the global-query and token
    sequences have different lengths.  ``DotProductAttention`` consumes ``bshd``
    inputs and returns the head dimensions flattened, so each pass is reshaped
    back to ``bshd``/``bhnd`` around the call.

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
    attn_fn : nn.Module
        Transformer Engine ``DotProductAttention`` module configured with
        ``qkv_format="bshd"`` and ``attention_type="cross"``.
    heads : int
        Number of attention heads :math:`H`, used to un-flatten the attention
        output.

    Returns
    -------
    torch.Tensor
        Self-attended output of shape :math:`(B, H, N, D)`.
    """
    G = q_global.to(dtype=x_mid.dtype).expand(x_mid.shape[0], -1, -1, -1)
    G = rearrange(G, "b h s d -> b s h d")
    k = rearrange(self_k(x_mid), "b h n d -> b n h d")
    v = rearrange(self_v(x_mid), "b h n d -> b n h d")
    z = attn_fn(G, k, v)
    z = rearrange(z, "b s (h d) -> b s h d", h=heads)
    y = attn_fn(k, G, z)
    return rearrange(y, "b n (h d) -> b h n d", h=heads)


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
    use_te : bool, optional, default=False
        Whether to use Transformer Engine backend when available.

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
        use_te: bool = False,
    ):
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

        # Transformer Engine cross-attention supports the unequal global and
        # token sequence lengths used by both FLARE attention passes. Keep
        # dropout in out_dropout so TE and PyTorch use the same dropout site.
        if self.use_te:
            self.attn_fn = te.DotProductAttention(
                num_attention_heads=self.heads,
                kv_channels=self.dim_head,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                attention_type="cross",
                qkv_format="bshd",
                softmax_scale=self.scale,
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
            x,
            self.in_project_x,
            self.heads,
            self.dim_head,
            "B N (H D) -> B N H D",
        )
        x_mid = x_mid.permute(0, 2, 1, 3)  # (B, N, H, D) -> (B, H, N, D)

        if self.use_te:
            y = _flare_self_attention_te(
                x_mid,
                self.q_global,
                self.self_k,
                self.self_v,
                self.attn_fn,
                self.heads,
            )
        else:
            y = _flare_self_attention(
                x_mid,
                self.q_global,
                self.self_k,
                self.self_v,
                self.scale,
            )

        out_x = y.permute(0, 2, 1, 3)  # (B, H, N, D) -> (B, N, H, D)
        out_x = rearrange(out_x, "b n h d -> b n (h d)")
        out_x = self.out_linear(out_x)
        return self.out_dropout(out_x)


class FLAREPlusPlus(nn.Module):
    r"""FLARE++ attention with input-conditioned routing queries.

    FLARE++ first uses learned seeds to summarize the current input into a set
    of routing queries. Those queries then gather values from the input and
    scatter the gathered information back to the input tokens. All three steps
    use scaled dot-product attention, so the cost is linear in the number of
    input tokens when the number of routing queries is fixed.

    PhysicsNeMo ``ShardTensor`` inputs may shard the token dimension. The two
    encoder calls then use globally normalized distributed SDPA, and the
    decoder preserves the token sharding without gathering the full sequence.

    For architecture details, see the `FLARE++ paper
    <https://arxiv.org/abs/2608.11519>`_.

    Parameters
    ----------
    dim : int
        Number of input and output channels.
    heads : int, optional
        Number of attention heads. Default is 8.
    dim_head : int, optional
        Number of channels per attention head. Default is 64.
    dropout : float, optional
        Dropout applied after the output projection. Default is 0.0.
    n_global_queries : int, optional
        Number of learned seeds and synthesized routing queries. Default is 64.
    use_te : bool, optional
        Transformer Engine is not currently supported for FLARE++. Default is
        ``False``.
    attn_scale : float | None, optional
        Scale applied to attention scores. ``None`` uses the standard
        ``1 / sqrt(dim_head)`` scale. Default is ``None``.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, N, C)`.

    Examples
    --------
    >>> import torch
    >>> attention = FLAREPlusPlus(dim=256, heads=8, dim_head=32)
    >>> output = attention(torch.randn(2, 100, 256))
    >>> output.shape
    torch.Size([2, 100, 256])
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        n_global_queries: int = 64,
        use_te: bool = False,
        attn_scale: float | None = None,
    ) -> None:
        super().__init__()
        if use_te:
            raise ValueError(
                "FLAREPlusPlus does not support Transformer Engine; set use_te=False."
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if dim_head <= 0:
            raise ValueError(f"dim_head must be positive, got {dim_head}")
        if n_global_queries <= 0:
            raise ValueError(
                f"n_global_queries must be positive, got {n_global_queries}"
            )
        if attn_scale is None:
            attn_scale = dim_head**-0.5
        if not math.isfinite(attn_scale) or attn_scale <= 0.0:
            raise ValueError(
                f"attn_scale must be a positive finite value, got {attn_scale}"
            )

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.n_global_queries = n_global_queries
        self.use_te = False
        self.scale = float(attn_scale)
        inner_dim = heads * dim_head

        self.q_seed = nn.Parameter(torch.randn(1, heads, n_global_queries, dim_head))
        # A fused projection is mathematically identical to four independent
        # projections and executes them in one matrix multiplication. The
        # chunks are query-synthesis K/V followed by physical K/V.
        self.in_projection = nn.Linear(dim, 4 * inner_dim)
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def _compute_attention(
        self, x: Float[torch.Tensor, "B N C"]
    ) -> tuple[
        Float[torch.Tensor, "B H N D"],
        Float[torch.Tensor, "B H N D"],
    ]:
        """Return FLARE++ head outputs and physical keys for backend reuse."""
        query_k, query_v, physical_k, physical_v = self.in_projection(x).chunk(
            4, dim=-1
        )
        query_k, query_v, physical_k, physical_v = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads, d=self.dim_head)
            for tensor in (query_k, query_v, physical_k, physical_v)
        )

        seeds = self.q_seed.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        queries = F.scaled_dot_product_attention(
            seeds, query_k, query_v, scale=self.scale
        )
        routed_values = F.scaled_dot_product_attention(
            queries, physical_k, physical_v, scale=self.scale
        )
        output = F.scaled_dot_product_attention(
            physical_k, queries, routed_values, scale=self.scale
        )
        return output, physical_k

    def _project_output(
        self, output: Float[torch.Tensor, "B H N D"]
    ) -> Float[torch.Tensor, "B N C"]:
        """Merge attention heads and apply the output projection."""
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.out_dropout(self.out_linear(output))

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Apply FLARE++ dynamic routing to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, N, C)`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected a 3D input tensor (B, N, C), got shape {tuple(x.shape)}"
                )

        output, _ = self._compute_attention(x)
        return self._project_output(output)
