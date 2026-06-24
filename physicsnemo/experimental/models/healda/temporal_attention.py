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
"""Temporal (video) self-attention over the time axis of ``(b, t, x, c)`` tensors.

Copied as-is from the healda model. Used by the factorized video DiT block: each
spatial location attends across time, complementing the per-frame spatial
attention. Supports rotary position embeddings on the time axis, an optional
linear (softmax-free) variant, and causal / sliding-window masking.
"""

import math

import einops
import torch
from jaxtyping import Float


class RotaryPositionEmbedding(torch.nn.Module):
    """Rotary Position Embedding (RoPE) for the time axis of ``(b, t, x, h, d)``."""

    def __init__(self, head_dim, base: int = 10000, max_seq_len: int = 24):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._precompute_freqs()

    def _precompute_freqs(self):
        position = torch.arange(self.max_seq_len).float()
        dim_indices = torch.arange(self.head_dim // 2).float()
        dim_indices = dim_indices[None, :]
        freqs = 1.0 / (self.base ** (2 * dim_indices / self.head_dim))
        freqs = position[:, None] * freqs
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    @torch.compile
    def forward(
        self, x: Float[torch.Tensor, "batch time space heads head_dim"]
    ) -> Float[torch.Tensor, "batch time space heads head_dim"]:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        return self._apply_rotary_pos_emb(x, freqs_cos, freqs_sin)

    def _apply_rotary_pos_emb(self, x, freqs_cos, freqs_sin):
        # x: [b, t, x, heads, head_dim]; freqs_*: [t, head_dim//2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos = freqs_cos[None, :, None, None, :]
        sin = freqs_sin[None, :, None, None, :]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = torch.stack([out1, out2], dim=-1)
        out = out.reshape(x.shape)
        return out


def mask_causal(attn, linear: bool = True, window: int | None = None):
    """Apply a causal mask to a ``[b tq tk x h]`` attention tensor (out-of-place).

    Zero-fill for linear attention, -inf fill for softmax attention. Masks out
    positions where tq < tk (supports asymmetric q/k lengths). When ``window`` is
    given, additionally restrict each query to a sliding lookback of ``window``
    frames (including itself). ``window=None`` is unbounded causal attention.
    """
    tq, tk = attn.shape[1], attn.shape[2]
    mask = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).triu(diagonal=1)
    if window is not None:
        too_old = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).tril(
            diagonal=-window
        )
        mask = mask | too_old
    return attn.masked_fill(
        mask.view(1, tq, tk, 1, 1), 0.0 if linear else float("-inf")
    )


class TemporalAttention(torch.nn.Module):
    """Temporal self-attention over the time dimension of ``(b, t, x, c)`` tensors.

    Args:
        embed_dim: Hidden dimension (split across heads).
        num_heads: Number of attention heads.
        use_rope: Apply rotary position embeddings to q/k along the time axis.
        rope_base: Base frequency for RoPE.
        max_seq_len: Maximum sequence length for RoPE cache.
        linear_attention: If True, skip softmax -- attention weights are raw
            dot-products (causal masking uses zero-fill instead of -inf).
        temporal_attn_legacy_scaling_bug: When True *and* linear_attention is True,
            normalize q.k by sequence length instead of head dim. Reproduces a bug
            from older checkpoints; set False for correct behaviour.
        causal_window: When set (and the forward pass is causal), restrict each
            frame to attend to itself and the previous ``causal_window - 1`` frames.
    """

    def __init__(
        self,
        *,
        embed_dim,
        num_heads,
        use_rope=True,
        rope_base=100,
        max_seq_len=100,
        linear_attention=True,
        temporal_attn_legacy_scaling_bug=False,
        causal_window: int | None = None,
    ) -> None:
        super().__init__()
        self._time_parallel_group = None
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.proj = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.head_dim = embed_dim // num_heads
        self.linear_attention = linear_attention
        self.temporal_attn_legacy_scaling_bug = temporal_attn_legacy_scaling_bug
        self.causal_window = causal_window

        if self.use_rope:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim, base=rope_base, max_seq_len=max_seq_len
            )
        else:
            self.rope = None

    @torch.compile
    def forward(
        self,
        x: Float[torch.Tensor, "batch time space channels"],
        is_causal: bool = False,
    ) -> Float[torch.Tensor, "batch time space channels"]:
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv,
            "b t x (n heads c) -> n b t x heads c",
            n=3,
            heads=self.num_heads,
        )

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.linear_attention:
            scale_dim = (
                k.shape[1] if self.temporal_attn_legacy_scaling_bug else k.shape[-1]
            )
        else:
            scale_dim = k.shape[-1]

        attn = torch.einsum(
            "b q x h c, b k x h c -> b q k x h", q, k / math.sqrt(scale_dim)
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
