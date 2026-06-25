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

r"""Rotary position embedding (RoPE) modules and primitives.

Overview
--------
Rotary Position Embedding (RoPE) encodes token position by *rotating* query
and key vectors before the attention dot-product. Because the dot-product of
a rotated query and a rotated key depends only on the *relative* angle between
them, RoPE gives attention position-awareness without adding any learned
parameters. No positional vectors are added to the token features — instead, the
position is woven into the rotation of each head's Q/K projections.

This module exposes two levels of API:

**Ready-to-use modules** (bring-your-own-attention):
  - :class:`RotaryPositionEmbedding2D` — axial 2D RoPE for global attention over
    a flattened :math:`h \times w` token grid (ViT / SDPA style, shape
    :math:`(B, \text{heads}, h \cdot w, head\_dim)`).
  - :class:`RotaryPositionEmbedding1D` — standard 1D sequence RoPE for general
    transformers (shape :math:`(B, \text{heads}, \text{seq}, head\_dim)`).

**Low-level functional helpers** (:func:`build_axial_rope_cos_sin_2d`,
:func:`build_rope_cos_sin_1d`, :func:`apply_rotary_pos_emb`):
  Used internally by the modules above and by attention implementations that
  need direct control over the table layout (e.g. NATTEN windowed attention,
  which keeps explicit spatial ``(h, w)`` dimensions, or domain-parallel
  paths that shard the tables across GPUs).

Choosing the right API
----------------------
* Writing a custom attention block that takes a *flattened* sequence from a 2D
  grid?  Use :class:`RotaryPositionEmbedding2D`.
* Writing a general-sequence transformer?  Use :class:`RotaryPositionEmbedding1D`.
* Implementing NATTEN windowed attention or need sharded / domain-parallel
  tables?  Use the functional helpers directly (see
  :class:`~physicsnemo.nn.module.dit_layers.RopeNatten2DSelfAttention` for a
  reference implementation).

Math (axial 2D RoPE)
--------------------
``head_dim`` is split in half: the first half rotates by row index, the second
by column index. Each axis has ``head_dim/4`` rotation pairs sharing a frequency
:math:`\theta_k = \text{base}^{-2k/(head\_dim/2)}` for
:math:`k = 0 \ldots head\_dim/4 - 1`. For an adjacent channel pair
:math:`(x_a, x_b)` at angle :math:`\phi`, the rotation is
:math:`(x_a \cos\phi - x_b \sin\phi,\ x_a \sin\phi + x_b \cos\phi)`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from jaxtyping import Float

from physicsnemo.core import Module


def build_axial_rope_cos_sin_2d(
    h: int,
    w: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Precompute axial 2D RoPE cos/sin tables for an :math:`h \times w` token grid.

    The first ``head_dim/2`` channels are rotated by the row index, the last
    ``head_dim/2`` by the column index. Within each axis-half, frequency
    :math:`\theta_k = \text{theta}^{-2k/(head\_dim/2)}` drives the adjacent
    channel pair ``(2k, 2k+1)``.

    Parameters
    ----------
    h : int
        Token grid height.
    w : int
        Token grid width.
    head_dim : int
        Per-head channel dimension. Must be divisible by 4 (half per axis, then
        adjacent pairs within each half).
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    device : torch.device, optional
        Device for the generated tables.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(h, w, head\_dim)` in fp32.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by 4 for axial 2D RoPE "
            f"(half per axis, then adjacent pairs within each half)."
        )
    half = head_dim // 2  # channels per axis

    # Frequencies for one axis: head_dim/4 unique values, each shared across an
    # adjacent channel pair via repeat_interleave below.
    k = torch.arange(0, half, 2, dtype=torch.float32, device=device)
    freqs = theta ** (-k / half)  # (head_dim/4,)

    row_idx = torch.arange(h, dtype=torch.float32, device=device)
    row_ang = row_idx[:, None] * freqs[None, :]  # (h, head_dim/4)
    col_idx = torch.arange(w, dtype=torch.float32, device=device)
    col_ang = col_idx[:, None] * freqs[None, :]  # (w, head_dim/4)

    # repeat_interleave(2) sends [a, b, c, ...] -> [a, a, b, b, c, c, ...] so that
    # the adjacent channel pair (2k, 2k+1) shares frequency theta_k.
    cos_row = row_ang.cos().repeat_interleave(2, dim=-1)  # (h, half)
    sin_row = row_ang.sin().repeat_interleave(2, dim=-1)
    cos_col = col_ang.cos().repeat_interleave(2, dim=-1)  # (w, half)
    sin_col = col_ang.sin().repeat_interleave(2, dim=-1)

    cos = torch.cat(
        [
            cos_row[:, None, :].expand(h, w, half),
            cos_col[None, :, :].expand(h, w, half),
        ],
        dim=-1,
    )  # (h, w, head_dim)
    sin = torch.cat(
        [
            sin_row[:, None, :].expand(h, w, half),
            sin_col[None, :, :].expand(h, w, half),
        ],
        dim=-1,
    )
    return cos.contiguous(), sin.contiguous()


def build_rope_cos_sin_1d(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Precompute 1D RoPE cos/sin tables for a length-``seq_len`` sequence.

    The standard sequence RoPE: every channel rotates by the token position,
    with ``head_dim/2`` frequencies :math:`\theta_k = \text{theta}^{-2k/head\_dim}`
    for :math:`k = 0 \ldots head\_dim/2 - 1`, each driving the adjacent channel
    pair ``(2k, 2k+1)``.

    Parameters
    ----------
    seq_len : int
        Number of positions in the sequence.
    head_dim : int
        Per-head channel dimension. Must be even (rotation acts on adjacent
        channel pairs).
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    device : torch.device, optional
        Device for the generated tables.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(seq\_len, head\_dim)` in fp32.
    """
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be even for 1D RoPE "
            f"(rotation acts on adjacent channel pairs)."
        )

    # head_dim/2 unique frequencies, each shared across an adjacent channel pair
    # via repeat_interleave below.
    k = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freqs = theta ** (-k / head_dim)  # (head_dim/2,)

    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    ang = pos[:, None] * freqs[None, :]  # (seq_len, head_dim/2)

    cos = ang.cos().repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = ang.sin().repeat_interleave(2, dim=-1)
    return cos.contiguous(), sin.contiguous()


def apply_rotary_pos_emb(
    x: Float[torch.Tensor, "..."],
    cos: Float[torch.Tensor, "..."],
    sin: Float[torch.Tensor, "..."],
) -> Float[torch.Tensor, "..."]:
    r"""Apply precomputed RoPE cos/sin tables to a query or key tensor.

    Rotates each adjacent channel pair :math:`(x_a, x_b)` in
    ``x`` by the angle encoded in the corresponding position of ``cos``/``sin``:

    .. math::

        (x_a,\, x_b) \;\mapsto\;
        (x_a \cos\phi - x_b \sin\phi,\;\; x_a \sin\phi + x_b \cos\phi)

    This is the standard *rotate-half* formulation
    ``x * cos + rotate_half(x) * sin``.  The arithmetic is promoted to fp32
    regardless of ``x``'s dtype (the sign-flipped term accumulates error in
    half precision) and cast back before returning.

    Call this directly when you manage the cos/sin tables
    yourself — for example, inside a custom NATTEN or domain-parallel attention
    block where you build the tables with :func:`build_axial_rope_cos_sin_2d`
    or :func:`build_rope_cos_sin_1d` and need to apply them independently to
    queries and keys.  If you are using :class:`RotaryPositionEmbedding2D` or
    :class:`RotaryPositionEmbedding1D`, those modules call this function
    internally and you do not need to invoke it yourself.

    Parameters
    ----------
    x : torch.Tensor
        Query or key tensor of shape :math:`(\ldots, \text{positions}, head\_dim)`.
    cos, sin : torch.Tensor
        Rotation tables broadcastable to ``x`` over the trailing
        ``(positions, head_dim)`` dimensions (e.g. shape
        :math:`(\text{positions}, head\_dim)`), as produced by
        :func:`build_axial_rope_cos_sin_2d` or :func:`build_rope_cos_sin_1d`.

    Returns
    -------
    torch.Tensor
        Rotated tensor of the same shape and dtype as ``x``.
    """
    in_dtype = x.dtype
    x = x.float()

    # rotate_half: swap adjacent channel pairs with a sign flip, mapping
    # (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...). Stacking (-x_odd, x_even)
    # along a new trailing axis and flattening interleaves them back into the
    # original (2k, 2k+1) channel order.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotate_half = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    return (x * cos + rotate_half * sin).to(in_dtype)


class RotaryPositionEmbedding2D(Module):
    r"""Axial 2D rotary position embedding for flattened-sequence attention.

    Encodes the 2D spatial position :math:`(row, col)` of
    each token by rotating its query and key vectors before the attention
    dot-product.  The first half of ``head_dim`` is rotated by the row index;
    the second half by the column index.  Because only the *relative* rotation
    between query and key enters the dot-product, attention scores are
    automatically sensitive to relative 2D position — no learned positional
    vectors are added to the token features.

    Use it when you are building a *custom attention module* that operates
    on a *flattened 2D token grid* in the standard
    :math:`(B, \text{heads}, N, head\_dim)` layout where
    :math:`N = h \times w`.  Typical examples:

    * Vision-transformer (ViT) style full-sequence
      :func:`torch.nn.functional.scaled_dot_product_attention`.
    * Custom ``timm``-style transformer blocks.
    * Any attention block that receives a flat token sequence but should
      respect 2D spatial geometry.

    When *not* to use this class:

    * *NATTEN windowed attention* keeps the spatial axes explicit
      :math:`(B, h, w, \text{heads}, head\_dim)`, so it needs tables with that
      layout; use the functional helpers or
      :class:`~physicsnemo.nn.module.dit_layers.RopeNatten2DSelfAttention`
      directly.
    * *Domain-parallel / sharded* attention needs tables that can be sliced
      along the ``h`` or ``w`` dimension; again use the functional helpers.

    Parameters
    ----------
    head_dim : int
        Per-head channel dimension. Must be divisible by 4 (half per spatial
        axis, then adjacent channel pairs within each half).
    latent_hw : Tuple[int, int]
        Spatial size :math:`(h, w)` of the token grid.
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.

    Forward
    -------
    q, k : torch.Tensor
        Query and key tensors of shape :math:`(\ldots, h \cdot w, head\_dim)`.
        Tokens must be in row-major order (height varies slowest), matching the
        order produced by ``tensor.flatten(-3, -2)`` from an :math:`(h, w)`
        spatial grid.
    latent_hw : Tuple[int, int], optional
        Override the spatial grid size at call time.  If given and different
        from the construction-time grid, the cos/sin tables are rebuilt in
        place before rotating (off the ``torch.compile`` fast path).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The rotated ``(q, k)``, same shape and dtype as the inputs.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import RotaryPositionEmbedding2D
    >>> rope = RotaryPositionEmbedding2D(head_dim=16, latent_hw=(4, 4))
    >>> q = torch.randn(2, 8, 16, 16)  # (B, heads, h*w, head_dim)
    >>> k = torch.randn(2, 8, 16, 16)
    >>> q_rot, k_rot = rope(q, k)
    >>> q_rot.shape
    torch.Size([2, 8, 16, 16])

    Wiring to :func:`torch.nn.functional.scaled_dot_product_attention` in a
    full multi-head self-attention pass over a flattened 2D token grid:

    .. code-block:: python

        import torch
        import torch.nn.functional as F
        from physicsnemo.nn.module.rope import RotaryPositionEmbedding2D

        B, num_heads, h, w, head_dim = 1, 4, 8, 8, 32
        D = num_heads * head_dim  # model dimension
        rope = RotaryPositionEmbedding2D(head_dim=head_dim, latent_hw=(h, w))
        N = h * w  # number of spatial tokens

        # Simulate linear Q/K/V projections from flat token sequence
        x = torch.randn(B, N, D)
        Wq = torch.nn.Linear(D, D, bias=False)
        Wk = torch.nn.Linear(D, D, bias=False)
        Wv = torch.nn.Linear(D, D, bias=False)
        q = Wq(x).view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        k = Wk(x).view(B, N, num_heads, head_dim).transpose(1, 2)
        v = Wv(x).view(B, N, num_heads, head_dim).transpose(1, 2)

        # Rotate queries and keys with axial 2D RoPE before attention
        q_rot, k_rot = rope(q, k)

        # Scaled dot-product attention; q_rot and k_rot carry position info
        out = F.scaled_dot_product_attention(q_rot, k_rot, v)  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)  # merge heads -> (B, N, D)
    """

    def __init__(
        self,
        head_dim: int,
        latent_hw: Tuple[int, int],
        theta: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 4 for axial 2D RoPE."
            )
        self.head_dim = int(head_dim)
        self.theta = float(theta)
        self._latent_hw: Tuple[int, int] = (int(latent_hw[0]), int(latent_hw[1]))
        cos, sin = build_axial_rope_cos_sin_2d(
            *self._latent_hw, self.head_dim, theta=self.theta
        )
        # Flatten the spatial axes to (h*w, head_dim) so the tables broadcast
        # against any (..., seq, head_dim) attention layout.
        self.register_buffer("cos", cos.reshape(-1, self.head_dim), persistent=False)
        self.register_buffer("sin", sin.reshape(-1, self.head_dim), persistent=False)

    def _rebuild_for_shape(self, h: int, w: int) -> None:
        """Rebuild the cos/sin tables for a new latent shape (off the hot path)."""
        target_dtype = self.cos.dtype
        target_device = self.cos.device
        cos, sin = build_axial_rope_cos_sin_2d(
            h, w, self.head_dim, theta=self.theta, device=target_device
        )
        self.register_buffer(
            "cos", cos.reshape(-1, self.head_dim).to(target_dtype), persistent=False
        )
        self.register_buffer(
            "sin", sin.reshape(-1, self.head_dim).to(target_dtype), persistent=False
        )
        self._latent_hw = (int(h), int(w))

    def forward(
        self,
        q: Float[torch.Tensor, "*batch seq head_dim"],
        k: Float[torch.Tensor, "*batch seq head_dim"],
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "*batch seq head_dim"],
        Float[torch.Tensor, "*batch seq head_dim"],
    ]:
        if latent_hw is not None and (
            (int(latent_hw[0]), int(latent_hw[1])) != self._latent_hw
        ):
            self._rebuild_for_shape(int(latent_hw[0]), int(latent_hw[1]))

        n = self.cos.shape[0]
        if not torch.compiler.is_compiling() and (q.shape[-2] != n or k.shape[-2] != n):
            raise ValueError(
                f"q/k sequence length must be h*w={n} (latent_hw={self._latent_hw}), "
                f"but got q={q.shape[-2]}, k={k.shape[-2]}"
            )
        return apply_rotary_pos_emb(q, self.cos, self.sin), apply_rotary_pos_emb(
            k, self.cos, self.sin
        )


class RotaryPositionEmbedding1D(Module):
    r"""Standard 1D rotary position embedding for sequence transformers.

    Encodes each token's absolute sequence position by
    rotating its query and key vectors before the attention dot-product.
    Because only the *relative* rotation between query and key enters the
    dot-product, attention scores are automatically sensitive to relative
    position — no learned positional vectors are added to the token features.
    This is the same RoPE variant used by most autoregressive and encoder
    transformer architectures (LLaMA, GPT-NeoX, etc.).

    Use it when your attention module operates on
    a *1D token sequence* in the standard
    :math:`(B, \text{heads}, \text{seq}, head\_dim)` layout.  Typical examples:

    * General encoder/decoder transformers over variable-length sequences.
    * Autoregressive language models with a causal attention mask.
    * Any custom attention block that needs sequence-position awareness.

    Inputs shorter than ``max_seq_len`` are rotated with the leading positions
    of the precomputed table, so a single module instance can serve any
    sequence length up to ``max_seq_len`` without rebuilding.  The cos/sin
    tables are stored as ``persistent=False`` buffers (they are
    deterministically reconstructed from ``(max_seq_len, head_dim, theta)``
    and do not need to be saved with the model weights).

    Parameters
    ----------
    head_dim : int
        Per-head channel dimension. Must be even (rotation acts on adjacent
        channel pairs).
    max_seq_len : int
        Maximum sequence length for which to precompute tables.
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.

    Forward
    -------
    q, k : torch.Tensor
        Query and key tensors of shape :math:`(\ldots, \text{seq}, head\_dim)`
        with ``seq <= max_seq_len``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The rotated ``(q, k)``, same shape and dtype as the inputs.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import RotaryPositionEmbedding1D
    >>> rope = RotaryPositionEmbedding1D(head_dim=16, max_seq_len=128)
    >>> q = torch.randn(2, 8, 100, 16)  # (B, heads, seq, head_dim)
    >>> k = torch.randn(2, 8, 100, 16)
    >>> q_rot, k_rot = rope(q, k)
    >>> q_rot.shape
    torch.Size([2, 8, 100, 16])

    Wiring to :func:`torch.nn.functional.scaled_dot_product_attention` with a
    causal mask, as used in autoregressive transformer decoders:

    .. code-block:: python

        import torch
        import torch.nn.functional as F
        from physicsnemo.nn.module.rope import RotaryPositionEmbedding1D

        B, num_heads, seq, head_dim = 2, 4, 64, 32
        D = num_heads * head_dim  # model dimension
        rope = RotaryPositionEmbedding1D(head_dim=head_dim, max_seq_len=128)

        # Simulate linear Q/K/V projections from a token sequence
        x = torch.randn(B, seq, D)
        Wq = torch.nn.Linear(D, D, bias=False)
        Wk = torch.nn.Linear(D, D, bias=False)
        Wv = torch.nn.Linear(D, D, bias=False)
        q = Wq(x).view(B, seq, num_heads, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        k = Wk(x).view(B, seq, num_heads, head_dim).transpose(1, 2)
        v = Wv(x).view(B, seq, num_heads, head_dim).transpose(1, 2)

        # Rotate queries and keys with 1D RoPE before attention
        q_rot, k_rot = rope(q, k)

        # Causal self-attention; RoPE makes dot-products sensitive to relative
        # position between query and key tokens, not absolute positions
        out = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, seq, D)  # merge heads -> (B, T, D)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for 1D RoPE.")
        self.head_dim = int(head_dim)
        self.theta = float(theta)
        self.max_seq_len = int(max_seq_len)
        cos, sin = build_rope_cos_sin_1d(
            self.max_seq_len, self.head_dim, theta=self.theta
        )
        self.register_buffer("cos", cos, persistent=False)  # (max_seq_len, head_dim)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        q: Float[torch.Tensor, "*batch seq head_dim"],
        k: Float[torch.Tensor, "*batch seq head_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "*batch seq head_dim"],
        Float[torch.Tensor, "*batch seq head_dim"],
    ]:
        seq_len = q.shape[-2]
        if not torch.compiler.is_compiling():
            if k.shape[-2] != seq_len:
                raise ValueError(
                    f"q and k must share a sequence length; got q={seq_len}, "
                    f"k={k.shape[-2]}"
                )
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
                )
        # Slice the leading positions so the module serves any length <= max.
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)
