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

**Shared table-provider modules** (owners of the cos/sin tables):
  - :class:`RotaryEmbedding2DTables` — owns axial 2D RoPE cos/sin tables for an
    :math:`h \times w` token grid, in explicit ``(h, w, head_dim)`` layout.
  - :class:`RotaryEmbedding1DTables` — owns standard 1D sequence RoPE cos/sin
    tables of shape ``(max_seq_len, head_dim)``.

  A provider holds *no* projections and applies *no* rotation itself: its
  ``forward`` simply returns the ``(cos, sin)`` tables. The intended pattern is
  that a top-level, multi-block model constructs a *single* provider and passes
  the returned tables into every attention block's ``forward`` (which rotates
  Q/K with the functional :func:`apply_rotary_pos_emb`), so the tables are
  built, stored, and — under domain parallelism — sharded exactly once instead
  of once per block. See
  :class:`~physicsnemo.nn.module.dit_layers.RopeNatten2DSelfAttention` and
  :class:`~physicsnemo.models.dit.DiT` for a reference wiring.

**Low-level functional helpers** (:func:`build_axial_rope_cos_sin_2d`,
:func:`build_rope_cos_sin_1d`, :func:`apply_rotary_pos_emb`):
  Used internally by the providers above and by attention implementations that
  need direct control over the table layout (e.g. NATTEN windowed attention,
  which keeps explicit spatial ``(h, w)`` dimensions, or domain-parallel
  paths that shard the tables across GPUs).

Choosing the right API
----------------------
* Building a multi-block transformer (2D grid or 1D sequence)?  Construct one
  :class:`RotaryEmbedding2DTables` / :class:`RotaryEmbedding1DTables` at the top
  level of the model and share its tables across every block, applying them
  with :func:`apply_rotary_pos_emb`.
* Implementing a single attention block or need full control over the table
  layout?  Call the functional helpers directly and apply them with
  :func:`apply_rotary_pos_emb`.

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
    block where you obtain the tables from a
    :class:`RotaryEmbedding2DTables` / :class:`RotaryEmbedding1DTables` provider
    (or build them with :func:`build_axial_rope_cos_sin_2d` /
    :func:`build_rope_cos_sin_1d`) and need to apply them independently to
    queries and keys.

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


class RotaryEmbedding2DTables(Module):
    r"""Shared owner of axial 2D RoPE cos/sin tables for an :math:`h \times w` grid.

    This module *owns* the cos/sin tables and nothing else: it holds no
    projections and applies no rotation. Its :meth:`forward` returns the
    ``(cos, sin)`` tables in explicit ``(h, w, head_dim)`` spatial layout, which
    a consumer applies to its query/key with :func:`apply_rotary_pos_emb`.

    The intended pattern is that a top-level, multi-block model constructs a
    *single* instance and passes the returned tables into every attention
    block's ``forward`` (see
    :class:`~physicsnemo.nn.module.dit_layers.RopeNatten2DSelfAttention` and
    :class:`~physicsnemo.models.dit.DiT`). Building, storing, and — under domain
    parallelism — sharding the tables then happens exactly once for the whole
    model instead of once per block.

    The tables are stored as ``persistent=False`` buffers named ``rope_cos`` /
    ``rope_sin``: they are deterministically reconstructed from
    ``(latent_hw, head_dim, theta)`` and do not need to be saved with the model
    weights. The names and the height-first ``(h, w, head_dim)`` layout are
    chosen so that domain-parallel sharding along dimension 0 (height) gives
    each rank globally-correct rows with no explicit rank offset in model code.

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
    latent_hw : Tuple[int, int], optional
        Override the spatial grid size at call time. If given and different from
        the current grid, the cos/sin tables are rebuilt in place before being
        returned (off the ``torch.compile`` fast path). Under domain parallelism
        the in-place rebuild replaces the sharded buffers with plain tensors, so
        it is only appropriate for single-device variable-resolution inference.

    Outputs
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(rope_cos, rope_sin)``, each of shape :math:`(h, w, head\_dim)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import (
    ...     RotaryEmbedding2DTables,
    ...     apply_rotary_pos_emb,
    ... )
    >>> rope = RotaryEmbedding2DTables(head_dim=16, latent_hw=(4, 4))
    >>> cos, sin = rope()
    >>> cos.shape
    torch.Size([4, 4, 16])
    >>> # q reshaped to spatial layout (B, heads, h, w, head_dim)
    >>> q = torch.randn(2, 8, 4, 4, 16)
    >>> q_rot = apply_rotary_pos_emb(q, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
    >>> q_rot.shape
    torch.Size([2, 8, 4, 4, 16])
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
        # persistent=False: not in state_dict (rebuilt deterministically from
        # latent_hw + head_dim + theta), so checkpoints stay lean. Names
        # rope_cos/rope_sin and (h, w, head_dim) layout let domain parallelism
        # shard parameters along desired axes.
        self.register_buffer("rope_cos", cos, persistent=False)  # (h, w, head_dim)
        self.register_buffer("rope_sin", sin, persistent=False)  # (h, w, head_dim)

    def _maybe_rebuild(self, h: int, w: int) -> None:
        r"""Rebuild the cos/sin tables for a new latent shape if it changed.

        Reached only when :meth:`forward` is called with a ``latent_hw`` that
        differs from the current grid (e.g. variable-resolution inference); not
        part of the training-time hot path.
        """
        if (int(h), int(w)) == self._latent_hw:
            return
        target_dtype = self.rope_cos.dtype
        target_device = self.rope_cos.device
        cos, sin = build_axial_rope_cos_sin_2d(
            h, w, self.head_dim, theta=self.theta, device=target_device
        )
        self.register_buffer("rope_cos", cos.to(dtype=target_dtype), persistent=False)
        self.register_buffer("rope_sin", sin.to(dtype=target_dtype), persistent=False)
        self._latent_hw = (int(h), int(w))

    def forward(
        self,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "h w head_dim"],
        Float[torch.Tensor, "h w head_dim"],
    ]:
        if latent_hw is not None:
            self._maybe_rebuild(int(latent_hw[0]), int(latent_hw[1]))
        return self.rope_cos, self.rope_sin


class RotaryEmbedding1DTables(Module):
    r"""Shared owner of standard 1D RoPE cos/sin tables for a token sequence.

    This module *owns* the cos/sin tables and nothing else: it holds no
    projections and applies no rotation. Its :meth:`forward` returns the
    ``(cos, sin)`` tables of shape :math:`(seq\_len, head\_dim)`, which a
    consumer applies to its query/key with :func:`apply_rotary_pos_emb`. This is
    the same RoPE variant used by most autoregressive and encoder transformer
    architectures (LLaMA, GPT-NeoX, etc.).

    A top-level, multi-block transformer constructs a *single* instance and
    shares its tables across all blocks, so the tables are built and stored once
    instead of once per block. Sequences shorter than ``max_seq_len`` are served
    by returning the leading positions of the precomputed table, so one instance
    covers any length up to ``max_seq_len`` without rebuilding.

    The tables are stored as ``persistent=False`` buffers (they are
    deterministically reconstructed from ``(max_seq_len, head_dim, theta)`` and
    do not need to be saved with the model weights).

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
    seq_len : int, optional
        Number of leading positions to return. If ``None``, the full
        ``max_seq_len`` table is returned.

    Outputs
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(seq\_len, head\_dim)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import (
    ...     RotaryEmbedding1DTables,
    ...     apply_rotary_pos_emb,
    ... )
    >>> rope = RotaryEmbedding1DTables(head_dim=16, max_seq_len=128)
    >>> cos, sin = rope(seq_len=100)
    >>> cos.shape
    torch.Size([100, 16])
    >>> q = torch.randn(2, 8, 100, 16)  # (B, heads, seq, head_dim)
    >>> q_rot = apply_rotary_pos_emb(q, cos, sin)
    >>> q_rot.shape
    torch.Size([2, 8, 100, 16])
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
        seq_len: Optional[int] = None,
    ) -> Tuple[
        Float[torch.Tensor, "seq head_dim"],
        Float[torch.Tensor, "seq head_dim"],
    ]:
        if seq_len is None:
            return self.cos, self.sin
        if not torch.compiler.is_compiling() and seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        # Slice the leading positions so one instance serves any length <= max.
        return self.cos[:seq_len], self.sin[:seq_len]
