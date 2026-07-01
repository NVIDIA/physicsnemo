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
"""Temporal self-attention over the time axis of :math:`(B, T, X, C)` tensors."""

import math
from typing import Optional

import einops
import torch
from jaxtyping import Float

from physicsnemo.nn.module.rope import RotaryPositionEmbedding1D


def mask_causal(
    attn: Float[torch.Tensor, "batch time_q time_k space heads"],
    linear: bool = True,
    window: Optional[int] = None,
) -> Float[torch.Tensor, "batch time_q time_k space heads"]:
    r"""Apply a causal mask to a :math:`(B, T_q, T_k, X, H)` attention tensor.

    Masks out positions where :math:`T_q < T_k` (future frames). Uses zero-fill
    for linear attention and ``-inf``-fill for softmax attention.

    Parameters
    ----------
    attn : torch.Tensor
        Attention logit tensor of shape :math:`(B, T_q, T_k, X, H)`.
    linear : bool, optional, default=True
        If ``True``, fill masked positions with ``0.0``; otherwise fill with
        ``-inf`` (for softmax attention).
    window : int or None, optional, default=None
        When set, additionally restricts each query frame to a lookback of
        ``window`` frames (including itself). ``None`` gives unbounded causal
        attention.

    Returns
    -------
    torch.Tensor
        Masked attention tensor of the same shape :math:`(B, T_q, T_k, X, H)`.
    """
    tq, tk = attn.shape[1], attn.shape[2]
    # Upper-triangular mask: True where t_k > t_q (future frames to mask out).
    mask = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).triu(diagonal=1)
    if window is not None:
        # Lower-triangular mask for frames older than `window` steps.
        too_old = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).tril(
            diagonal=-window
        )
        mask = mask | too_old
    return attn.masked_fill(
        mask.view(1, tq, tk, 1, 1), 0.0 if linear else float("-inf")
    )


class TemporalAttention(torch.nn.Module):
    r"""Temporal self-attention over the time dimension of :math:`(B, T, X, C)` tensors.

    Each spatial location attends independently across the time axis, complementing
    per-frame spatial attention in a factorized video DiT block. Supports rotary
    position embeddings on the time axis via
    :class:`~physicsnemo.nn.module.rope.RotaryPositionEmbedding1D`, an optional
    softmax-free (linear) attention variant, and causal / sliding-window masking.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension :math:`C`, split evenly across ``num_heads``.
    num_heads : int
        Number of attention heads.
    use_rope : bool, optional, default=True
        Apply rotary position embeddings to queries and keys along the time axis.
    rope_base : int, optional, default=100
        Base frequency :math:`\theta` for the RoPE sinusoidal schedule.
    max_seq_len : int, optional, default=100
        Maximum temporal sequence length for the RoPE cache pre-computation.
    linear_attention : bool, optional, default=True
        If ``True``, skip softmax — attention weights are raw dot-products and
        causal masking uses zero-fill instead of ``-inf``.
    causal_window : int or None, optional, default=None
        When set (and the forward pass is causal), restricts each frame to attend
        to itself and the previous ``causal_window - 1`` frames only.

    Forward
    -------
    x : torch.Tensor
        Input latents of shape :math:`(B, T, X, C)`.
    is_causal : bool, optional, default=False
        If ``True``, apply causal masking via :func:`mask_causal`.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, C)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.healda.temporal_attention import TemporalAttention
    >>> layer = TemporalAttention(hidden_size=64, num_heads=4)
    >>> x = torch.randn(2, 8, 16, 64)
    >>> out = layer(x, is_causal=False)
    >>> out.shape
    torch.Size([2, 8, 16, 64])
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        use_rope: bool = True,
        rope_base: int = 100,
        max_seq_len: int = 100,
        linear_attention: bool = True,
        causal_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._time_parallel_group = None
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.head_dim = hidden_size // num_heads
        self.linear_attention = linear_attention
        self.causal_window = causal_window

        self.rope: Optional[RotaryPositionEmbedding1D]
        if self.use_rope:
            self.rope = RotaryPositionEmbedding1D(
                head_dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_base
            )
        else:
            self.rope = None

    @torch.compile
    def forward(
        self,
        x: Float[torch.Tensor, "batch time space hidden_size"],
        is_causal: bool = False,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        r"""Compute temporal self-attention over the time axis.

        Parameters
        ----------
        x : torch.Tensor
            Input latents of shape :math:`(B, T, X, C)`.
        is_causal : bool, optional, default=False
            If ``True``, apply causal masking so each frame only attends to
            itself and prior frames (optionally within ``causal_window``).

        Returns
        -------
        torch.Tensor
            Output latents of shape :math:`(B, T, X, C)`.
        """
        # Project to queries, keys, values: (B, T, X, 3*C) -> 3 x (B, T, X, H, C_h)
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv,
            "b t x (n heads c) -> n b t x heads c",
            n=3,
            heads=self.num_heads,
        )

        if self.rope is not None:
            # RotaryPositionEmbedding1D rotates the second-to-last axis; move T there.
            q = einops.rearrange(q, "b t x h c -> b x h t c")
            k = einops.rearrange(k, "b t x h c -> b x h t c")
            q, k = self.rope(q, k)
            q = einops.rearrange(q, "b x h t c -> b t x h c")
            k = einops.rearrange(k, "b x h t c -> b t x h c")

        attn = torch.einsum(
            "b q x h c, b k x h c -> b q k x h", q, k / math.sqrt(k.shape[-1])
        )

        if is_causal:
            attn = mask_causal(
                attn, linear=self.linear_attention, window=self.causal_window
            )
        if not self.linear_attention:
            attn = attn.softmax(2)

        out = einops.einsum(attn, v, "b q k x h, b k x h c -> b q x h c")
        out = einops.rearrange(out, "b t x h c -> b t x (h c)")
        out = self.proj(out)
        return out
