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

from physicsnemo.core.version_check import OptionalImport

from .physics_attention import _project_input

te = OptionalImport("transformer_engine.pytorch")


def _flare_encode(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    scale: float,
) -> tuple[
    Float[torch.Tensor, "B H N D"],
    Float[torch.Tensor, "B H S D"],
    Float[torch.Tensor, "B H S D"],
]:
    r"""FLARE encode pass: gather token values into the global slots.

    First half of :func:`_flare_self_attention`, exposed separately so
    callers can transform the latent slots between the encode and decode
    passes (e.g. a latent-bottleneck context read). The matching decode
    pass is ``F.scaled_dot_product_attention(k, G, z, scale=scale)``.

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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Projected keys of shape :math:`(B, H, N, D)`, expanded global
        queries of shape :math:`(B, H, S, D)`, and latent slots of shape
        :math:`(B, H, S, D)`.
    """
    G = q_global.to(dtype=x_mid.dtype).expand(x_mid.shape[0], -1, -1, -1)
    k = self_k(x_mid)
    v = self_v(x_mid)
    z = F.scaled_dot_product_attention(G, k, v, scale=scale)
    return k, G, z


def _flare_encode_te(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    attn_fn: nn.Module,
    heads: int,
) -> tuple[
    Float[torch.Tensor, "B N H D"],
    Float[torch.Tensor, "B S H D"],
    Float[torch.Tensor, "B S H D"],
]:
    r"""FLARE encode pass on the Transformer Engine backend.

    Same computation as :func:`_flare_encode`, with the keys, global
    queries, and latent slots returned in the ``bshd`` layout consumed by
    the Transformer Engine ``DotProductAttention`` decode call
    ``attn_fn(k, G, z)``.

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
        Number of attention heads :math:`H`, used to un-flatten the
        attention output.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Projected keys of shape :math:`(B, N, H, D)`, expanded global
        queries of shape :math:`(B, S, H, D)`, and latent slots of shape
        :math:`(B, S, H, D)`.
    """
    G = q_global.to(dtype=x_mid.dtype).expand(x_mid.shape[0], -1, -1, -1)
    G = rearrange(G, "b h s d -> b s h d")
    k = rearrange(self_k(x_mid), "b h n d -> b n h d")
    v = rearrange(self_v(x_mid), "b h n d -> b n h d")
    z = attn_fn(G, k, v)
    z = rearrange(z, "b s (h d) -> b s h d", h=heads)
    return k, G, z


def _flare_self_attention(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    scale: float,
    context: Float[torch.Tensor, "B H S_c D_c"] | None = None,
    cross_q: nn.Module | None = None,
    cross_k: nn.Module | None = None,
    cross_v: nn.Module | None = None,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel.

    Computes low-rank attention via learned global queries: first aggregate
    token values into global slots, then distribute back to tokens.  When the
    caller passes a ``context``, the global slots additionally cross-attend
    to it between the two passes, adding the read back to the latent stream.

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
    context : torch.Tensor or None, optional
        Context of shape :math:`(B, H, S_c, D_c)` read at the latent
        bottleneck. Default is ``None`` (no context read).
    cross_q : nn.Module or None, optional
        Query projection applied to the latent slots; required when
        ``context`` is not ``None``. Default is ``None``.
    cross_k : nn.Module or None, optional
        Key projection applied to ``context``; required when ``context`` is
        not ``None``. Default is ``None``.
    cross_v : nn.Module or None, optional
        Value projection applied to ``context``; required when ``context``
        is not ``None``. Default is ``None``.

    Returns
    -------
    torch.Tensor
        Self-attended output of shape :math:`(B, H, N, D)`.
    """
    k, G, z = _flare_encode(x_mid, q_global, self_k, self_v, scale)
    if context is not None:
        z = z + F.scaled_dot_product_attention(
            cross_q(z), cross_k(context), cross_v(context), scale=scale
        )
    return F.scaled_dot_product_attention(k, G, z, scale=scale)


def _flare_self_attention_te(
    x_mid: Float[torch.Tensor, "B H N D"],
    q_global: nn.Parameter,
    self_k: nn.Module,
    self_v: nn.Module,
    attn_fn: nn.Module,
    heads: int,
    context: Float[torch.Tensor, "B H S_c D_c"] | None = None,
    cross_q: nn.Module | None = None,
    cross_k: nn.Module | None = None,
    cross_v: nn.Module | None = None,
) -> Float[torch.Tensor, "B H N D"]:
    r"""FLARE two-pass self-attention kernel on the Transformer Engine backend.

    Same computation as :func:`_flare_self_attention`, but the two attention
    passes and the optional context read run through a Transformer Engine
    ``DotProductAttention`` module.  All passes run as cross-attention
    because the global-query, token, and context sequences have different
    lengths.  The ``DotProductAttention`` module consumes ``bshd`` inputs and
    returns the head dimensions flattened, so the kernel reshapes each pass
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
    context : torch.Tensor or None, optional
        Context of shape :math:`(B, H, S_c, D_c)` read at the latent
        bottleneck. Default is ``None`` (no context read).
    cross_q : nn.Module or None, optional
        Query projection applied to the latent slots; required when
        ``context`` is not ``None``. Default is ``None``.
    cross_k : nn.Module or None, optional
        Key projection applied to ``context``; required when ``context`` is
        not ``None``. Default is ``None``.
    cross_v : nn.Module or None, optional
        Value projection applied to ``context``; required when ``context``
        is not ``None``. Default is ``None``.

    Returns
    -------
    torch.Tensor
        Self-attended output of shape :math:`(B, H, N, D)`.
    """
    k, G, z = _flare_encode_te(x_mid, q_global, self_k, self_v, attn_fn, heads)
    if context is not None:
        k_ctx = rearrange(cross_k(context), "b h s d -> b s h d")
        v_ctx = rearrange(cross_v(context), "b h s d -> b s h d")
        z_ctx = attn_fn(cross_q(z), k_ctx, v_ctx)
        z = z + rearrange(z_ctx, "b s (h d) -> b s h d", h=heads)
    y = attn_fn(k, G, z)
    return rearrange(y, "b n (h d) -> b h n d", h=heads)


class FLARE(nn.Module):
    r"""FLARE: Fast Low-rank Attention Routing Engine attention layer.
    Adopted:
    - FLARE attention: Fast Low-rank Attention Routing Engine
        paper: https://arxiv.org/abs/2508.12594

    Optionally, when constructed with ``context_dim > 0``, the layer
    cross-attends to an external context sequence at the latent bottleneck:
    the encoded latents query the context and add the read back to the
    latent stream before decoding.

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
    context_dim : int, optional
        Dimension :math:`D_c` of an optional external context sequence. When
        greater than 0, the layer creates projections for a cross-attention
        read of the context at the latent bottleneck, between the encode and
        decode passes. Reading the context from the ``n_global_queries``
        latent tokens instead of the :math:`N` point tokens reduces the
        context-attention cost by a factor ``n_global_queries`` :math:`/ N`.
        Default is 0 (no context read).

    Forward
    -------
    x : torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])
    context : torch.Tensor[Batch, Heads, N_context, D_context] ([B, H, S_c, D_c]), optional
        Context read at the latent bottleneck; requires ``context_dim > 0``.
        When ``None``, the layer skips the read and reduces to plain FLARE
        attention.
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

    With a context read at the latent bottleneck:

    >>> flare = FLARE(dim=256, heads=8, dim_head=32, context_dim=16)
    >>> context = torch.randn(2, 8, 12, 16)
    >>> outputs = flare(x, context)
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
        context_dim: int = 0,
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

        # Linear projections for the latent-bottleneck context read
        self.context_dim = context_dim
        if context_dim > 0:
            self.cross_q = linear_layer(dim_head, dim_head)
            self.cross_k = linear_layer(context_dim, dim_head)
            self.cross_v = linear_layer(context_dim, dim_head)
        else:
            self.cross_q = None
            self.cross_k = None
            self.cross_v = None

        # Transformer Engine cross-attention supports the unequal global,
        # token, and context sequence lengths used by the FLARE attention
        # passes and the optional context read. Keep dropout in out_dropout
        # so TE and PyTorch use the same dropout site.
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

    def forward(
        self,
        x: Float[torch.Tensor, "B N C"],
        context: Float[torch.Tensor, "B H S_c D_c"] | None = None,
    ) -> Float[torch.Tensor, "B N C"]:
        r"""Forward pass of the FLARE module.

        Applies FLARE attention to the input features, with an optional
        cross-attention read of ``context`` at the latent bottleneck.

        Parameters
        ----------
        x : torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])
            Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
            :math:`N` is number of points, and :math:`C` is number of channels.
        context : torch.Tensor | None, optional
            Context tensor of shape :math:`(B, H, S_c, D_c)` where :math:`H`
            is number of heads, :math:`S_c` is number of context tokens, and
            :math:`D_c` is ``context_dim``. Requires ``context_dim > 0``.
            When ``None``, the layer skips the context read. Default is
            ``None``.

        Returns
        -------
        torch.Tensor[Batch, N_points, N_Channels] ([B, N, C])
            Output tensor of shape :math:`(B, N, C)`, same shape as inputs.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if context is not None:
                if self.context_dim == 0:
                    raise ValueError(
                        "Received a context but the layer has no context "
                        "projections; construct FLARE with context_dim > 0 "
                        "to enable the context read."
                    )
                if (
                    context.ndim != 4
                    or context.shape[1] != self.heads
                    or context.shape[-1] != self.context_dim
                ):
                    raise ValueError(
                        f"Expected context of shape (B, {self.heads}, S_c, "
                        f"{self.context_dim}), got tensor with shape "
                        f"{tuple(context.shape)}"
                    )

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
                context=context,
                cross_q=self.cross_q,
                cross_k=self.cross_k,
                cross_v=self.cross_v,
            )
        else:
            y = _flare_self_attention(
                x_mid,
                self.q_global,
                self.self_k,
                self.self_v,
                self.scale,
                context=context,
                cross_q=self.cross_q,
                cross_k=self.cross_k,
                cross_v=self.cross_v,
            )

        out_x = y.permute(0, 2, 1, 3)  # (B, H, N, D) -> (B, N, H, D)
        out_x = rearrange(out_x, "b n h d -> b n (h d)")
        out_x = self.out_linear(out_x)
        return self.out_dropout(out_x)
