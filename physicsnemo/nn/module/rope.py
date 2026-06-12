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
import torch.nn as nn
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


# --- Stereographic 2D RoPE (continuous spherical coordinates) ---
#
# A 2D RoPE for tokens that live on a sphere. Token latitude/longitude are mapped
# to a local tangent plane with a stereographic projection, and the resulting
# *continuous* (x, y) coordinates drive the same adjacent-pair rotation as the
# axial RoPE above (via :func:`apply_rotary_pos_emb`). The only new ingredient is
# a table builder for continuous positions
# (:func:`build_axial_rope_cos_sin_2d_continuous`); :func:`build_axial_rope_cos_sin_2d`
# is its integer-grid special case.


def stereographic_projection(
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat0: torch.Tensor,
    lon0: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Project latitude / longitude onto a local tangent plane.

    Stereographic projection of points :math:`(\text{lat}, \text{lon})` onto the
    plane tangent to the sphere at the center :math:`(\text{lat}_0, \text{lon}_0)`,
    with axes oriented so ``y`` points North and ``x`` points East. All inputs are
    in radians and may broadcast against each other (e.g. ``lat`` of shape
    :math:`(B, H, W)` with ``lat0`` of shape :math:`(B, 1, 1)`).

    Parameters
    ----------
    lat : torch.Tensor
        Latitude in radians, of shape :math:`(\ldots, H, W)`.
    lon : torch.Tensor
        Longitude in radians, of shape :math:`(\ldots, H, W)`.
    lat0 : torch.Tensor
        Center latitude in radians, broadcastable to ``lat``.
    lon0 : torch.Tensor
        Center longitude in radians, broadcastable to ``lon``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The projected ``(x, y)`` coordinates on the tangent plane (``x`` East,
        ``y`` North), each of the broadcasted shape of the inputs.

    Notes
    -----
    The projection diverges at the antipode of the center (:math:`\cos c = -1`).
    The denominator is clamped so outputs stay finite there (large but not
    infinite); this is intended for tiles local to the center, not whole-sphere use.
    """
    dlon = lon - lon0
    # cos_c: cosine of the great-circle angle to the center; k: stereographic scale.
    cos_c = torch.sin(lat0) * torch.sin(lat) + torch.cos(lat0) * torch.cos(
        lat
    ) * torch.cos(dlon)
    # Guard the antipodal singularity (cos_c = -1, the point opposite the center),
    # where the projection diverges: clamp the denominator so coordinates stay
    # finite for tiles that approach it.
    k = 2.0 / (1.0 + cos_c).clamp_min(1e-6)
    x = k * torch.cos(lat) * torch.sin(dlon)
    y = k * (
        torch.cos(lat0) * torch.sin(lat)
        - torch.sin(lat0) * torch.cos(lat) * torch.cos(dlon)
    )
    return x, y


def spherical_centroid(
    lat: torch.Tensor,
    lon: torch.Tensor,
    reduce_dims: Tuple[int, ...] = (-2, -1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Robust center :math:`(\text{lat}_0, \text{lon}_0)` of points on the sphere.

    Each ``(lat, lon)`` is lifted to a 3D unit vector
    :math:`(\cos\text{lat}\cos\text{lon},\ \cos\text{lat}\sin\text{lon},\ \sin\text{lat})`,
    the vectors are averaged over ``reduce_dims``, and the mean direction is read
    back as ``(lat0, lon0)``. Averaging in 3D — rather than per-axis on the
    angles — is correct at the poles (where the plain mean of latitude undershoots
    :math:`\pm\pi/2` and longitude is degenerate) and across the
    :math:`0 / 2\pi` longitude seam. The reduced dimensions are kept (size 1) for
    broadcasting.

    Parameters
    ----------
    lat : torch.Tensor
        Latitude in radians, of shape :math:`(\ldots, H, W)`.
    lon : torch.Tensor
        Longitude in radians, of shape :math:`(\ldots, H, W)`.
    reduce_dims : Tuple[int, ...], optional, default=(-2, -1)
        Dimensions to average over.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(lat0, lon0)`` in radians (``lat0`` in :math:`[-\pi/2, \pi/2]`, ``lon0``
        in :math:`(-\pi, \pi]`), with the reduced dimensions kept as size 1.

    Notes
    -----
    The mean direction is ill-defined only when the points nearly cancel (an
    antipodal / whole-sphere spread), which is outside this module's local-tile
    scope (see :class:`StereographicRotaryPositionEmbedding2D`).
    """
    cos_lat = lat.cos()
    x = (cos_lat * lon.cos()).mean(dim=reduce_dims, keepdim=True)
    y = (cos_lat * lon.sin()).mean(dim=reduce_dims, keepdim=True)
    z = lat.sin().mean(dim=reduce_dims, keepdim=True)
    lat0 = torch.atan2(z, torch.hypot(x, y))
    lon0 = torch.atan2(y, x)
    return lat0, lon0


def build_rope_cos_sin_1d_continuous(
    positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Build 1D RoPE cos/sin tables from arbitrary continuous positions.

    The continuous-position analog of :func:`build_rope_cos_sin_1d` (which takes an
    integer ``seq_len`` and generates positions :math:`0 \ldots seq\_len - 1`): here
    the positions are supplied explicitly and may be any real values. All ``dim``
    channels rotate by the position, with ``dim/2`` frequencies
    :math:`\theta_k = \text{theta}^{-2k/dim}`, each driving the adjacent channel pair
    ``(2k, 2k+1)`` so the result composes with :func:`apply_rotary_pos_emb`.

    This is the shared building block for continuous RoPE in higher dimensions:
    :func:`build_axial_rope_cos_sin_2d_continuous` calls it once per axis over
    ``head_dim/2`` channels.

    Parameters
    ----------
    positions : torch.Tensor
        Continuous positions of shape :math:`(\ldots, N)`.
    dim : int
        Number of channels rotated by ``positions``. Must be even. For standalone
        1D use this is ``head_dim``; per axis of a 2D embedding it is ``head_dim/2``.
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    device : torch.device, optional
        Device to place the positions and returned tables on. If ``None``, follows
        ``positions.device``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(\ldots, N, dim)`.
    """
    if dim % 2 != 0:
        raise ValueError(
            f"dim={dim} must be even (rotation acts on adjacent channel pairs)."
        )
    # Move positions to the requested device (if any) so positions, frequencies,
    # and the returned tables all share one device; otherwise follow positions.
    if device is not None:
        positions = positions.to(device)
    k = torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device)
    freqs = theta ** (-k / dim)  # (dim/2,)

    # Outer product of positions and frequencies -> per-token angles.
    ang = positions.to(torch.float32).unsqueeze(-1) * freqs  # (..., N, dim/2)
    # repeat_interleave(2) makes the adjacent channel pair (2k, 2k+1) share theta_k.
    cos = ang.cos().repeat_interleave(2, dim=-1)  # (..., N, dim)
    sin = ang.sin().repeat_interleave(2, dim=-1)
    return cos.contiguous(), sin.contiguous()


def build_axial_rope_cos_sin_2d_continuous(
    x_pos: torch.Tensor,
    y_pos: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Build axial 2D RoPE cos/sin tables from arbitrary continuous coordinates.

    The continuous-coordinate analog of :func:`build_axial_rope_cos_sin_2d` (which takes
    integer grid sizes ``h, w`` and generates row/column indices): here the per-token
    ``(x, y)`` coordinates are supplied explicitly and may be any real values.
    ``head_dim`` is split in half — the first half rotates by ``x_pos``, the second by
    ``y_pos`` — each half built with :func:`build_rope_cos_sin_1d_continuous` over
    ``head_dim/2`` channels, so the result composes with :func:`apply_rotary_pos_emb`.
    Passing integer row / column indices reproduces :func:`build_axial_rope_cos_sin_2d`
    (flattened over the grid).

    Parameters
    ----------
    x_pos : torch.Tensor
        First-axis coordinates of shape :math:`(\ldots, N)`.
    y_pos : torch.Tensor
        Second-axis coordinates of shape :math:`(\ldots, N)` (same shape as ``x_pos``).
    head_dim : int
        Per-head channel dimension. Must be divisible by 4 (half per axis, then
        adjacent pairs within each half).
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    device : torch.device, optional
        Device to place the coordinates and returned tables on. If ``None``, follows
        ``x_pos.device``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(\ldots, N, head\_dim)`.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by 4 for axial 2D RoPE "
            f"(half per axis, then adjacent pairs within each half)."
        )
    half = head_dim // 2  # channels per axis
    cos_x, sin_x = build_rope_cos_sin_1d_continuous(
        x_pos, half, theta=theta, device=device
    )
    cos_y, sin_y = build_rope_cos_sin_1d_continuous(
        y_pos, half, theta=theta, device=device
    )
    cos = torch.cat([cos_x, cos_y], dim=-1)  # (..., N, head_dim)
    sin = torch.cat([sin_x, sin_y], dim=-1)
    return cos, sin


class StereographicRotaryPositionEmbedding2D(nn.Module):
    r"""Stereographic 2D rotary position embedding for tokens on a sphere.

    The continuous-coordinate counterpart of :class:`RotaryPositionEmbedding2D`.
    Token positions, given as latitude / longitude, are mapped to a local tangent
    plane via :func:`stereographic_projection`, and the resulting continuous
    ``(x, y)`` coordinates drive a 2D RoPE on the query / key tensors (via
    :func:`build_axial_rope_cos_sin_2d_continuous` + :func:`apply_rotary_pos_emb`). Because the
    positions depend on the input geometry, the cos/sin tables are built per
    forward rather than cached as buffers, so the module is parameter-free and
    stateless beyond its ``head_dim`` / ``theta`` configuration.

    Parameters
    ----------
    head_dim : int
        Per-head channel dimension. Must be divisible by 4.
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.

    Forward
    -------
    q, k : torch.Tensor
        Query / key tensors of shape :math:`(\ldots, N, head\_dim)`.
    x_pos, y_pos : torch.Tensor
        Continuous tangent-plane coordinates of shape :math:`(N,)` (shared across
        the batch) or :math:`(B, N)` (per sample), e.g. from :meth:`project`. For
        the standard :math:`(B, \text{heads}, N, head\_dim)` ``q`` / ``k`` layout a
        heads axis is inserted automatically so per-sample tables broadcast over heads.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The rotated ``(q, k)``.

    Notes
    -----
    The stereographic projection is singular at the antipode of its center and its
    distortion grows with distance from that center. This embedding is therefore
    intended for tokens that are *local* to a single center -- i.e. a regional grid,
    where one :meth:`project` call (its center defaults to the grid's mean lat/lon)
    is enough. It is **not** meant for a single projection of a whole-sphere / very
    large global grid, where the far-field distortion breaks the relative-position
    interpretation. Covering a global field would require splitting it into local
    windows/tiles and projecting each around its own center -- which the caller must
    arrange; this module only provides the per-tile projection primitive.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import StereographicRotaryPositionEmbedding2D
    >>> rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    >>> q = torch.randn(2, 4, 6, 16)  # (B, heads, N, head_dim)
    >>> k = torch.randn(2, 4, 6, 16)
    >>> x = torch.randn(6)            # continuous per-token coordinates
    >>> y = torch.randn(6)
    >>> q_rot, k_rot = rope(q, k, x, y)
    >>> q_rot.shape
    torch.Size([2, 4, 6, 16])
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 4 for stereographic 2D RoPE."
            )
        self.head_dim = int(head_dim)
        self.theta = float(theta)

    def project(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        length_scale: float,
        lat0: Optional[torch.Tensor] = None,
        lon0: Optional[torch.Tensor] = None,
        reduce_dims: Tuple[int, ...] = (-2, -1),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Map latitude / longitude to stereographic tangent-plane coordinates.

        When ``lat0`` / ``lon0`` are not given, the projection center is the
        :func:`spherical_centroid` of the inputs (a 3D unit-vector mean), which is
        robust at the poles and across the longitude seam.

        Parameters
        ----------
        lat : torch.Tensor
            Latitude in radians, of shape :math:`(\ldots, H, W)`.
        lon : torch.Tensor
            Longitude in radians, of shape :math:`(\ldots, H, W)`.
        length_scale : float
            Positive divisor applied to the projected coordinates, bringing them to
            roughly token-spacing units so the RoPE frequencies behave like the
            integer-grid case. Required and with no default: a sensible value cannot
            be inferred from the data, so the caller must choose it explicitly.
        lat0 : torch.Tensor, optional
            Center latitude in radians; if ``None``, taken from the
            :func:`spherical_centroid` of ``(lat, lon)`` over ``reduce_dims``.
        lon0 : torch.Tensor, optional
            Center longitude in radians; if ``None``, taken from the
            :func:`spherical_centroid` of ``(lat, lon)`` over ``reduce_dims``.
        reduce_dims : Tuple[int, ...], optional, default=(-2, -1)
            Dimensions used to compute the center when ``lat0`` / ``lon0`` are not given.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The tangent-plane ``(x, y)`` coordinates (East, North), of the shape of ``lat``.
        """
        if length_scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {length_scale}")
        if lat0 is None or lon0 is None:
            c_lat, c_lon = spherical_centroid(lat, lon, reduce_dims=reduce_dims)
            if lat0 is None:
                lat0 = c_lat
            if lon0 is None:
                lon0 = c_lon
        x, y = stereographic_projection(lat, lon, lat0, lon0)
        return x / length_scale, y / length_scale

    def build_tables(
        self,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Build the ``(cos, sin)`` RoPE tables for continuous coordinates.

        Thin wrapper over :func:`build_axial_rope_cos_sin_2d_continuous` using this
        module's ``head_dim`` and ``theta``.

        Parameters
        ----------
        x_pos, y_pos : torch.Tensor
            Continuous coordinates of shape :math:`(\ldots, N)`.
        device : torch.device, optional
            Device for the frequency table.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(cos, sin)``, each of shape :math:`(\ldots, N, head\_dim)`.
        """
        return build_axial_rope_cos_sin_2d_continuous(
            x_pos, y_pos, self.head_dim, theta=self.theta, device=device
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.build_tables(x_pos, y_pos, device=q.device)
        # Per-sample (batched) coordinates give tables with one fewer dim than
        # q/k (no heads axis); insert it so they broadcast over the heads dim of
        # the standard (..., heads, N, head_dim) attention layout.
        if cos.ndim == q.ndim - 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)
