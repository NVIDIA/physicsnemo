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
r"""Spherical transformer (S2Transformer) for gridded geophysical fields.

This module provides :class:`S2Transformer`, a fully attention-based
encoder / processor / decoder backbone that operates directly on the sphere.
Every internal mixer is a spherical transformer block built around
torch-harmonics' :class:`~torch_harmonics.AttentionS2` /
:class:`~torch_harmonics.NeighborhoodAttentionS2` (Bonev et al., NeurIPS 2025,
https://arxiv.org/abs/2505.11157):

.. code-block:: text

    S2AttEncoder (in_channels -> embed_dim,  H_in,W_in -> h,w)
        + optional positional embedding on the (h, w) internal grid
        -> [SphericalTransformerBlock] x num_layers
    S2AttDecoder (embed_dim -> out_channels, h,w -> H_out,W_out)
        + optional big-skip residual from the input

Each :class:`SphericalTransformerBlock` is the canonical pre-norm transformer:
``x = x + drop_path(self_attn(norm0(x)))`` followed by
``x = x + drop_path(mlp(norm1(x)))``. The attention is quadrature-aware on the
sphere (softmax weights include the geodesic Jacobian), so the layer remains
SO(3)-equivariant in expectation and is well-behaved at the poles.

``torch_harmonics`` is an optional dependency; it is only imported when a model
from this module is instantiated (see :func:`physicsnemo.core.version_check`).
"""

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import (
    OptionalImport,
    check_version_spec,
    register_package_hint,
)

from .layers import MLP, DropPath

# torch_harmonics provides the spherical attention / resampling / SHT kernels.
# It is optional: the imports below are lazy (``th.<Attr>`` triggers the import
# on first access), so ``import physicsnemo`` never fails for users who do not
# need this model. A clear, actionable error is raised at model construction
# time when the package is missing (see ``S2Transformer.__init__``).
register_package_hint(
    "torch_harmonics",
    "torch_harmonics is required for the spherical transformer (S2Transformer) "
    "models.\nInstall with:\n  pip install torch_harmonics>=0.9.3\n"
    "  pip install nvidia-physicsnemo[model-extras]",
)
TORCH_HARMONICS_AVAILABLE = check_version_spec(
    "torch_harmonics", "0.9.3", hard_fail=False
)
th = OptionalImport("torch_harmonics")


def _neighborhood_radius_rad(nlat: int, factor: float) -> float:
    r"""Geodesic-cap radius (radians) for neighborhood attention.

    Expressed in units of latitude spacing on an equiangular-style grid:
    ``radius = factor * pi / (nlat - 1)``. ``factor = 1.0`` therefore covers
    one latitude ring, ``factor = 4.0`` four rings, etc. Attention has no kernel
    basis (unlike a DISCO convolution), so the radius is just a geodesic angle.

    Parameters
    ----------
    nlat : int
        Number of latitude points of the grid the attention runs on.
    factor : float
        Neighborhood radius in units of latitude rings.

    Returns
    -------
    float
        Cutoff radius in radians.
    """
    return float(factor) * math.pi / float(max(int(nlat) - 1, 1))


def _check_bchw(
    x: torch.Tensor, channels: int, shape: Tuple[int, int], name: str = "x"
) -> None:
    r"""Raise if ``x`` is not a :math:`(B, C, H, W)` tensor with the given ``C, H, W``.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to check.
    channels : int
        Expected channel count :math:`C`.
    shape : tuple of int
        Expected grid shape :math:`(H, W)`.
    name : str, optional, default="x"
        Argument name used in the error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``x`` does not have shape :math:`(B, C, H, W)`.
    """
    if x.ndim != 4 or x.shape[1] != channels or tuple(x.shape[-2:]) != tuple(shape):
        raise ValueError(
            f"Expected tensor {name} of shape (B, {channels}, {shape[0]}, {shape[1]}) "
            f"but got tensor of shape {tuple(x.shape)}"
        )


# -----------------------------------------------------------------------------
# Attention encoder / decoder
# -----------------------------------------------------------------------------


class S2AttEncoder(nn.Module):
    r"""Spherical-attention encoder: ``(in_chans, H_in, W_in) -> (out_chans, h, w)``.

    Pipeline:

    1. Pointwise 1x1 lift ``in_chans -> out_chans`` on the input grid.
    2. :class:`~torch_harmonics.NeighborhoodAttentionS2` **cross-resolution
       downsampling**: the attention itself aggregates from the full-resolution
       K/V tokens to the coarse Q positions, so no explicit resampling module is
       needed. Q must be at ``out_shape`` and K/V at ``inp_shape``. Q is seeded
       by a spherically-correct bilinear downsample (:class:`~torch_harmonics.ResampleS2`)
       which also serves as the residual base; an optional positional embedding
       is added to Q before the attention.

    When ``inp_shape == out_shape`` there is no downsampling to do, so the
    resampling and the cross-resolution attention are skipped entirely (not even
    constructed) and the encoder degenerates to the pointwise 1x1 lift.

    Requires ``nlon_in % nlon_out == 0`` (a hard constraint of the underlying
    spherical attention kernel).

    Parameters
    ----------
    inp_shape : tuple of int
        Input grid shape :math:`(H_{in}, W_{in})`.
    out_shape : tuple of int
        Output (latent) grid shape :math:`(h, w)`.
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    grid_in : str
        Quadrature grid type of the input grid.
    grid_out : str
        Quadrature grid type of the latent grid.
    num_heads : int
        Number of attention heads.
    theta_cutoff_factor : float
        Neighborhood radius (in latitude rings) for the cross-resolution attention.
    qk_norm : bool
        Whether to apply RMSNorm to the query/key projections.
    bias : bool
        Whether the linear projections use a bias.
    attn_optimized_kernel : bool
        Whether to use the optimized CUDA neighborhood-attention kernel.
    attn_dim : int, optional, default=None
        Channel width at which the (expensive, full-resolution) cross-attention
        runs. ``None`` means no bottleneck (``out_chans``).
    pos_embed_max_degree : int, optional, default=None
        Maximum spherical-harmonic degree of the owned positional embedding
        (only used when the encoder bottlenecks).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H_{in}, W_{in})`.

    Outputs
    -------
    torch.Tensor
        Encoded tensor of shape :math:`(B, C_{out}, h, w)`.
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        in_chans: int,
        out_chans: int,
        grid_in: str,
        grid_out: str,
        num_heads: int,
        theta_cutoff_factor: float,
        qk_norm: bool,
        bias: bool,
        attn_optimized_kernel: bool,
        attn_dim: Optional[int] = None,
        pos_embed_max_degree: Optional[int] = None,
    ):
        super().__init__()
        self.inp_shape = tuple(inp_shape)
        self.out_shape = tuple(out_shape)
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        # The cross-resolution attention (and its bilinear-downsample residual
        # base) only exist when the encoder actually changes resolution. When
        # inp_shape == out_shape there is nothing to downsample, so the encoder
        # degenerates to the pointwise lift.
        self._needs_resample = inp_shape != out_shape

        # Channel width at which the (expensive, full-resolution) cross-attention
        # runs. ``None`` -> out_chans (no bottleneck). When set smaller, the lift
        # maps straight to attn_dim, the attention runs narrow against the
        # full-resolution K/V, and a cheap 1x1 conv at the LATENT grid expands
        # attn_dim -> out_chans, slashing full-res attention memory by
        # ~ (out_chans / attn_dim)x.
        self.attn_dim = (
            int(attn_dim) if (attn_dim and self._needs_resample) else self.out_chans
        )
        if self._needs_resample and self.attn_dim % num_heads != 0:
            raise ValueError(
                f"S2AttEncoder: attn_dim ({self.attn_dim}) must be divisible "
                f"by num_heads ({num_heads})."
            )

        # When bottlenecking, lift straight to the narrow attn_dim; otherwise the
        # lift already produces the full out_chans.
        lift_out = self.attn_dim if self._needs_resample else self.out_chans
        self.lift = nn.Conv2d(in_chans, lift_out, kernel_size=1, bias=bias)
        nn.init.normal_(self.lift.weight, std=math.sqrt(2.0 / max(1, in_chans)))
        if bias:
            nn.init.zeros_(self.lift.bias)

        # Spatial PE applied to Q only. When bottlenecking, the shared embed_dim
        # PE no longer matches the narrow attention width, so the encoder owns a
        # PE at attn_dim; otherwise it is assigned externally by the parent model.
        self._owns_pe = self._needs_resample and (self.attn_dim != self.out_chans)
        if self._owns_pe:
            self.pos_embed: Optional[nn.Module] = _SpectralPositionEmbedding(
                out_shape,
                self.attn_dim,
                grid_out,
                max_degree=pos_embed_max_degree,
            )
        else:
            self.pos_embed = None

        if self._needs_resample:
            # Spherically-correct bilinear downsample to the latent grid.
            # Used both as the residual base and as Q seed for the cross-attention.
            self.resample = th.ResampleS2(
                inp_shape[0],
                inp_shape[1],
                out_shape[0],
                out_shape[1],
                grid_in=grid_in,
                grid_out=grid_out,
            )
            # Cross-resolution attention: K/V at inp_shape, Q at out_shape.
            # Output is added as a residual on top of the bilinear downsample.
            theta_cutoff = _neighborhood_radius_rad(inp_shape[0], theta_cutoff_factor)
            self.attn = th.NeighborhoodAttentionS2(
                in_channels=self.attn_dim,
                in_shape=inp_shape,
                out_shape=out_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                num_heads=num_heads,
                theta_cutoff=theta_cutoff,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=self.attn_dim,
                optimized_kernel=attn_optimized_kernel,
            )

        # Expand attn_dim -> out_chans at the latent grid (cheap, low-res). When
        # not bottlenecking this is an Identity.
        if self._needs_resample and self.attn_dim != self.out_chans:
            self.expand = nn.Conv2d(
                self.attn_dim, self.out_chans, kernel_size=1, bias=bias
            )
            nn.init.normal_(
                self.expand.weight, std=math.sqrt(2.0 / max(1, self.attn_dim))
            )
            if bias:
                nn.init.zeros_(self.expand.bias)
        else:
            self.expand = nn.Identity()

    def forward(
        self, x: Float[torch.Tensor, "batch in_chans in_lat in_lon"]
    ) -> Float[torch.Tensor, "batch out_chans lat lon"]:
        r"""Encode the input onto the latent grid."""
        if not torch.compiler.is_compiling():
            _check_bchw(x, self.in_chans, self.inp_shape)
        x = self.lift(x)  # (B, attn_dim | out_chans, H, W)
        if not self._needs_resample:
            # Same shape: pointwise lift only, no resampling/attention.
            return x
        dtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            # Residual base: spherically-correct bilinear downsample.
            q_base = self.resample(x_f)  # (B, attn_dim, h, w)
            # Q for attention: add spatial PE so attention is position-aware.
            q_attn = self.pos_embed(q_base) if self.pos_embed is not None else q_base
            # Cross-attention correction: what the full-resolution input adds
            # beyond the bilinear base.
            delta = self.attn(q_attn, key=x_f, value=x_f)  # (B, attn_dim, h, w)
            x = (q_base + delta).to(dtype)  # residual
        x = self.expand(x)  # (B, out_chans, h, w)
        return x


class S2AttDecoder(nn.Module):
    r"""Spherical-attention decoder: ``(in_chans, h, w) -> (out_chans, H_out, W_out)``.

    Mirror of :class:`S2AttEncoder`. When the latent and output grids differ,
    the decoder upsamples with a spherically-correct bilinear
    :class:`~torch_harmonics.ResampleS2` (or an SHT round-trip when
    ``upsample_sht=True``) and adds a learned cross-resolution "attnup"
    correction on top of that base:

    .. code-block:: text

        kv   = reduce(z)                          # optional attn_dim bottleneck (latent grid)
        base = upsample(kv)                       # -> output grid (residual base)
        q    = base + pos_embed                   # position-aware query on the output grid
        x    = base + attn(q, key=kv, value=kv)   # cross-resolution upsampling attention
        out  = project(x)                         # attn_dim -> out_chans

    Only the query is at the output resolution; K/V stay on the cheap latent grid.
    When the latent grid already equals the output grid (no resolution change) the
    upsample, positional embedding and attention are skipped entirely and the
    decoder degenerates to the pointwise channel projection (``reduce`` ->
    ``project``) -- exactly mirroring the encoder, whose attention only runs when
    resampling.

    The cross-resolution upsampling attention requires ``nlon_out % nlon_in == 0``.
    ``theta_cutoff_factor`` is a geodesic cap on the OUTPUT grid; the latent K/V
    spacing is ~``H_out / h`` output rings, so pick a factor at least as large as
    the up-sampling ratio (~``scale_factor``) so every fine query reaches >= 1
    coarse point.

    Parameters
    ----------
    inp_shape : tuple of int
        Input (latent) grid shape :math:`(h, w)`.
    out_shape : tuple of int
        Output grid shape :math:`(H_{out}, W_{out})`.
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    grid_in : str
        Quadrature grid type of the latent grid.
    grid_out : str
        Quadrature grid type of the output grid.
    num_heads : int
        Number of attention heads.
    theta_cutoff_factor : float
        Neighborhood radius (in output-grid latitude rings) for the upsampling
        attention. Use a value at least as large as the up-sampling ratio.
    qk_norm : bool
        Whether to apply RMSNorm to the query/key projections.
    bias : bool
        Whether the linear projections use a bias.
    attn_optimized_kernel : bool
        Whether to use the optimized CUDA neighborhood-attention kernel.
    upsample_sht : bool, optional, default=False
        Whether to upsample with an SHT round-trip instead of :class:`~torch_harmonics.ResampleS2`.
    attn_dim : int, optional, default=None
        Channel width at which the (full-resolution) attention query runs.
        ``None`` means no bottleneck (``in_chans``).
    pos_embed_max_degree : int, optional, default=None
        Maximum spherical-harmonic degree for the query positional embedding.

    Forward
    -------
    x : torch.Tensor
        Latent tensor of shape :math:`(B, C_{in}, h, w)`.

    Outputs
    -------
    torch.Tensor
        Decoded tensor of shape :math:`(B, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        in_chans: int,
        out_chans: int,
        grid_in: str,
        grid_out: str,
        num_heads: int,
        theta_cutoff_factor: float,
        qk_norm: bool,
        bias: bool,
        attn_optimized_kernel: bool,
        upsample_sht: bool = False,
        attn_dim: Optional[int] = None,
        pos_embed_max_degree: Optional[int] = None,
    ):
        super().__init__()
        self.inp_shape = tuple(inp_shape)
        self.out_shape = tuple(out_shape)
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self._needs_resample = inp_shape != out_shape
        # Channel width at which the (full-resolution) attention query runs.
        # ``None`` -> in_chans (no bottleneck). When set smaller, a cheap 1x1 conv
        # at the LATENT grid reduces in_chans -> attn_dim before the attnup, and
        # the final projection maps attn_dim -> out_chans.
        self.attn_dim = int(attn_dim) if attn_dim else self.in_chans
        if self._needs_resample and self.attn_dim % num_heads != 0:
            raise ValueError(
                f"S2AttDecoder: attn_dim ({self.attn_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        # Optional channel bottleneck at the cheap latent grid (Identity when off).
        if self.attn_dim != self.in_chans:
            self.reduce = nn.Conv2d(
                self.in_chans, self.attn_dim, kernel_size=1, bias=bias
            )
            nn.init.normal_(
                self.reduce.weight, std=math.sqrt(2.0 / max(1, self.in_chans))
            )
            if bias:
                nn.init.zeros_(self.reduce.bias)
        else:
            self.reduce = nn.Identity()

        # The upsample, positional-embedding query and cross-resolution attention
        # only exist when the decoder actually changes resolution. When
        # inp_shape == out_shape the decoder degenerates to the pointwise
        # projection (mirrors the encoder).
        if self._needs_resample:
            # Bilinear-spherical (or SHT) upsample latent -> output grid: both the
            # residual base and the source of the attention query.
            if upsample_sht:
                sht = th.RealSHT(*inp_shape, grid=grid_in).float()
                isht = th.InverseRealSHT(
                    *out_shape,
                    lmax=sht.lmax,
                    mmax=sht.mmax,
                    grid=grid_out,
                ).float()
                self.upsample = nn.Sequential(sht, isht)
            else:
                self.upsample = th.ResampleS2(
                    *inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out
                )

            # Position-aware query on the OUTPUT grid (non-persistent buffer, not
            # in the state dict; recomputed for whatever grid is requested).
            self.pos_embed: Optional[nn.Module] = _SpectralPositionEmbedding(
                out_shape,
                self.attn_dim,
                grid_out,
                max_degree=pos_embed_max_degree,
            )

            # Cross-resolution UPSAMPLING attention: K/V at the latent (inp_shape),
            # Q at the output (out_shape). Requires nlon_out % nlon_in == 0.
            theta_cutoff = _neighborhood_radius_rad(out_shape[0], theta_cutoff_factor)
            self.attn: Optional[nn.Module] = th.NeighborhoodAttentionS2(
                in_channels=self.attn_dim,
                in_shape=inp_shape,
                out_shape=out_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                num_heads=num_heads,
                theta_cutoff=theta_cutoff,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=self.attn_dim,
                optimized_kernel=attn_optimized_kernel,
            )
        else:
            self.upsample = nn.Identity()
            self.pos_embed = None
            self.attn = None

        self.project = nn.Conv2d(self.attn_dim, out_chans, kernel_size=1, bias=bias)
        nn.init.normal_(self.project.weight, std=math.sqrt(2.0 / max(1, self.attn_dim)))
        if bias:
            nn.init.zeros_(self.project.bias)

    def forward(
        self, x: Float[torch.Tensor, "batch in_chans lat lon"]
    ) -> Float[torch.Tensor, "batch out_chans out_lat out_lon"]:
        r"""Decode the latent tensor onto the output grid."""
        if not torch.compiler.is_compiling():
            _check_bchw(x, self.in_chans, self.inp_shape)
        x = self.reduce(x)  # (B, attn_dim, h, w)
        if not self._needs_resample:
            # No resolution change: pointwise channel projection only.
            return self.project(x)
        dtype = x.dtype
        with amp.autocast(device_type=x.device.type, enabled=False):
            kv = x.float()  # (B, attn_dim, h, w)
            base = self.upsample(kv)  # (B, attn_dim, H, W)
            q = self.pos_embed(base)  # base + PE (position-aware query)
            delta = self.attn(q, key=kv, value=kv)  # (B, attn_dim, H, W)
            x = (base + delta).to(dtype)  # residual on the bilinear base
        return self.project(x)  # (B, out_chans, H, W)


# -----------------------------------------------------------------------------
# Positional embeddings
# -----------------------------------------------------------------------------


class _SpectralPositionEmbedding(nn.Module):
    r"""Spherical-harmonic positional embedding (channel-wise basis function).

    Each channel is one real spherical harmonic on the ``(h, w)`` grid,
    normalised to unit max amplitude. Cheap, deterministic, and well-suited to
    spherical transformer pretraining.

    Degree coverage (``max_degree``):

    - ``None`` (default): channel ``i`` is the ``i``-th harmonic in the standard
      ``(l, m)`` enumeration, so for ``num_chans`` channels the degree only
      reaches ``l ~ floor(sqrt(num_chans))`` (low-frequency only).
    - integer: spread ``num_chans`` harmonics evenly (in enumeration index) over
      degrees ``0 .. max_degree`` (clamped to the grid bandlimit), so the
      embedding carries high-frequency components able to resolve fine position
      differences while still including the smooth low-degree modes.

    Parameters
    ----------
    grid_shape : tuple of int
        Grid shape :math:`(h, w)` the embedding is defined on.
    num_chans : int
        Number of channels (one harmonic per channel).
    grid : str
        Quadrature grid type.
    max_degree : int, optional, default=None
        Maximum spherical-harmonic degree to spread the channels over.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        num_chans: int,
        grid: str,
        max_degree: Optional[int] = None,
    ):
        super().__init__()
        H, W = int(grid_shape[0]), int(grid_shape[1])
        isht = th.InverseRealSHT(nlat=H, nlon=W, grid=grid)

        # Build the list of (l, m) harmonics assigned to each channel.
        if max_degree is None:
            # The first ``num_chans`` harmonics (all low-degree).
            pairs = []
            for i in range(num_chans):
                ell = math.floor(math.sqrt(i))
                m = i - ell * ell - ell
                pairs.append((ell, m))
        else:
            # Spread across degrees 0..L (clamped to the grid bandlimit), so
            # high-frequency harmonics are included for sharp localisation.
            L = max(0, min(int(max_degree), int(isht.lmax) - 1))
            all_pairs = [
                (ell, m)
                for ell in range(L + 1)
                for m in range(-ell, ell + 1)
                if abs(m) < int(isht.mmax)
            ]
            if len(all_pairs) >= num_chans:
                idx = (
                    torch.linspace(0, len(all_pairs) - 1, num_chans)
                    .round()
                    .long()
                    .tolist()
                )
                pairs = [all_pairs[j] for j in idx]
            else:
                # Fewer harmonics than channels (tiny grids): cycle through.
                pairs = [all_pairs[j % len(all_pairs)] for j in range(num_chans)]

        with torch.no_grad():
            pos_freq = torch.zeros(
                1, num_chans, isht.lmax, isht.mmax, dtype=torch.complex64
            )
            for i, (ell, m) in enumerate(pairs):
                # Guard against out-of-range indices on small grids.
                if ell >= isht.lmax or abs(m) >= isht.mmax:
                    continue
                if m < 0:
                    pos_freq[0, i, ell, -m] = 1.0j
                else:
                    pos_freq[0, i, ell, m] = 1.0
            pos_embed = isht(pos_freq)
            pos_embed = pos_embed / (
                pos_embed.abs().amax(dim=(-1, -2), keepdim=True) + 1.0e-8
            )
        self.register_buffer("position_embeddings", pos_embed.float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Add the (broadcast) positional embedding to ``x``."""
        return x + self.position_embeddings


def _build_pos_embedding(
    kind: str,
    grid_shape: Tuple[int, int],
    num_chans: int,
    grid: str,
    max_degree: Optional[int] = None,
) -> nn.Module:
    r"""Build the positional-embedding module selected by ``kind``.

    Parameters
    ----------
    kind : str
        One of ``"none"`` or ``"spectral"``.
    grid_shape : tuple of int
        Grid shape :math:`(h, w)`.
    num_chans : int
        Number of channels.
    grid : str
        Quadrature grid type (used by the spectral embedding).
    max_degree : int, optional, default=None
        Maximum spherical-harmonic degree for the spectral embedding.

    Returns
    -------
    nn.Module
        The positional-embedding module (or ``nn.Identity`` for ``"none"``).
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "spectral":
        return _SpectralPositionEmbedding(
            grid_shape, num_chans, grid, max_degree=max_degree
        )
    raise ValueError(f"Unknown pos_embed='{kind}'. Supported: 'none', 'spectral'.")


# -----------------------------------------------------------------------------
# Spherical transformer block
# -----------------------------------------------------------------------------


def _make_norm(kind: str, num_chans: int) -> nn.Module:
    r"""Build the normalization module selected by ``kind``.

    Parameters
    ----------
    kind : str
        One of ``"none"``, ``"layer_norm"``, ``"instance_norm"``.
    num_chans : int
        Number of channels to normalize over.

    Returns
    -------
    nn.Module
        The normalization module (or ``nn.Identity`` for ``"none"``).
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "layer_norm":
        # GroupNorm with one group, i.e. normalization over channels AND space
        # jointly, per sample. NOT the same as a LayerNorm applied over channels at
        # each location, which would rescale every pixel to the same magnitude and
        # erase the field's spatial contrast; here a quiet region stays quiet in the
        # hidden state.
        return nn.GroupNorm(num_groups=1, num_channels=num_chans, eps=1e-6, affine=True)
    if kind == "instance_norm":
        return nn.InstanceNorm2d(
            num_features=num_chans, eps=1e-6, affine=True, track_running_stats=False
        )
    raise ValueError(
        f"Unknown normalization_layer='{kind}'. Supported: 'none', 'layer_norm', 'instance_norm'."
    )


class SphericalTransformerBlock(nn.Module):
    r"""Pre-norm spherical transformer block.

    The spatial mixer (attention) keeps the channel count constant at
    ``in_chans``, and the MLP is the component that changes channels
    (``in_chans -> out_chans``). When ``in_chans != out_chans`` the MLP residual
    is disabled (there is no channel-matching identity path).

    Order of operations:

    .. code-block:: text

        residual = x
        x = norm0(x)
        x = self_attn(x)               # in_chans -> in_chans
        x = residual + drop_path(x)
        residual = x
        x = norm1(x)
        x = mlp(x)                     # in_chans -> out_chans
        x = residual + drop_path(x)    # only if in_chans == out_chans

    Parameters
    ----------
    in_shape : tuple of int
        Grid shape :math:`(h, w)` the block operates on.
    grid : str
        Quadrature grid type.
    in_chans : int
        Number of input channels (must be divisible by ``num_heads``).
    out_chans : int
        Number of output channels.
    num_heads : int
        Number of attention heads.
    attention_mode : str, optional, default="neighborhood"
        Either ``"neighborhood"`` (local geodesic-cap attention) or ``"global"``
        (full quadrature-weighted attention).
    attn_theta_cutoff : float, optional, default=None
        Neighborhood radius in radians (only used for ``"neighborhood"``).
    mlp_ratio : float, optional, default=2.0
        Ratio of the MLP hidden dimension to ``in_chans``.
    mlp_drop_rate : float, optional, default=0.0
        Dropout rate inside the MLP.
    attn_drop_rate : float, optional, default=0.0
        Dropout rate inside the global-attention softmax (no-op for neighborhood).
    path_drop_rate : float, optional, default=0.0
        Stochastic-depth (drop-path) rate.
    act_layer : nn.Module, optional, default=nn.GELU
        Activation layer constructor.
    normalization_layer : str, optional, default="layer_norm"
        One of ``"none"``, ``"layer_norm"``, ``"instance_norm"``.
    bias : bool, optional, default=True
        Whether the linear projections use a bias.
    use_mlp : bool, optional, default=True
        Whether to include the MLP sublayer. If ``False``, ``in_chans`` must
        equal ``out_chans``.
    qk_norm : bool, optional, default=False
        Whether to apply RMSNorm to the query/key projections.
    attn_optimized_kernel : bool, optional, default=True
        Whether to use the optimized CUDA neighborhood-attention kernel.
    checkpointing_level : int, optional, default=0
        Activation-checkpointing aggressiveness (``>= 2`` checkpoints the MLP).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, h, w)`.
    qk_pos_embed : torch.Tensor, optional, default=None
        Positional embedding of shape :math:`(1, C_{in}, h, w)` added to the
        attention queries and keys only. ``None`` attends on ``x`` directly.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, h, w)`.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        grid: str,
        in_chans: int,
        out_chans: int,
        num_heads: int,
        attention_mode: str = "neighborhood",
        attn_theta_cutoff: Optional[float] = None,
        mlp_ratio: float = 2.0,
        mlp_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        normalization_layer: str = "layer_norm",
        bias: bool = True,
        use_mlp: bool = True,
        qk_norm: bool = False,
        attn_optimized_kernel: bool = True,
        checkpointing_level: int = 0,
    ):
        super().__init__()

        if in_chans % num_heads != 0:
            raise ValueError(
                f"SphericalTransformerBlock: in_chans ({in_chans}) must be "
                f"divisible by num_heads ({num_heads})."
            )

        self.in_shape = tuple(in_shape)
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.attention_mode = attention_mode
        self.checkpointing_level = int(checkpointing_level)

        self.norm0 = _make_norm(normalization_layer, in_chans)
        if attention_mode == "neighborhood":
            self.self_attn = th.NeighborhoodAttentionS2(
                in_channels=in_chans,
                in_shape=in_shape,
                out_shape=in_shape,
                grid_in=grid,
                grid_out=grid,
                num_heads=num_heads,
                theta_cutoff=attn_theta_cutoff,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=in_chans,
                optimized_kernel=attn_optimized_kernel,
            )
        elif attention_mode == "global":
            self.self_attn = th.AttentionS2(
                in_channels=in_chans,
                num_heads=num_heads,
                in_shape=in_shape,
                out_shape=in_shape,
                grid_in=grid,
                grid_out=grid,
                use_qknorm=qk_norm,
                bias=bias,
                out_channels=in_chans,
                drop_rate=attn_drop_rate,
            )
        else:
            raise ValueError(
                f"Unknown attention_mode='{attention_mode}'. Supported: 'neighborhood', 'global'."
            )
        self.drop_path0 = (
            DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        )

        self.norm1 = _make_norm(normalization_layer, in_chans)
        if use_mlp:
            self.mlp = MLP(
                in_features=in_chans,
                out_features=out_chans,
                hidden_features=int(in_chans * mlp_ratio),
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                checkpointing=(self.checkpointing_level >= 2),
            )
        else:
            self.mlp = None
            if in_chans != out_chans:
                raise ValueError(
                    f"SphericalTransformerBlock: use_mlp=False requires "
                    f"in_chans ({in_chans}) == out_chans ({out_chans}); "
                    f"with no MLP there is no channel-changing path."
                )
        self.drop_path1 = (
            DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        )

        # A channel-changing MLP cannot share the residual stream, so the MLP
        # residual is only applied when in_chans == out_chans.
        self._mlp_residual = self.in_chans == self.out_chans

    def _attn_forward(
        self,
        x: torch.Tensor,
        qk_pos_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Both AttentionS2 and NeighborhoodAttentionS2 manage mixed precision
        # internally: the optimized neighborhood kernel registers an AutocastCUDA
        # dispatch that follows the active autocast dtype (and the torch fallback
        # upcasts q/k/v to fp32), so no manual autocast guard is needed here.
        if qk_pos_embed is None:
            return self.self_attn(x)

        # Position enters each layer through Q/K only. Values and the residual
        # stream remain pure content. The attention module applies its learned
        # Q/K/V projections and optional QK normalization afterward.
        qk = x + qk_pos_embed.to(device=x.device, dtype=x.dtype)
        return self.self_attn(qk, qk, x)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_chans lat lon"],
        qk_pos_embed: Optional[Float[torch.Tensor, "1 in_chans lat lon"]] = None,
    ) -> Float[torch.Tensor, "batch out_chans lat lon"]:
        r"""Apply the pre-norm attention and MLP sublayers."""
        if not torch.compiler.is_compiling():
            _check_bchw(x, self.in_chans, self.in_shape)
            if qk_pos_embed is not None:
                _check_bchw(qk_pos_embed, self.in_chans, self.in_shape, "qk_pos_embed")
        residual = x
        h = self.norm0(x)
        h = self._attn_forward(h, qk_pos_embed)
        x = residual + self.drop_path0(h)

        if self.mlp is not None:
            residual = x
            h = self.norm1(x)
            h = self.mlp(h)
            h = self.drop_path1(h)
            x = residual + h if self._mlp_residual else h
        return x


# -----------------------------------------------------------------------------
# S2Transformer
# -----------------------------------------------------------------------------


@dataclass
class MetaData(ModelMetaData):
    r"""Metadata for :class:`S2Transformer`."""

    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


# -----------------------------------------------------------------------------
# Configuration groups
# -----------------------------------------------------------------------------
# The many S2Transformer hyper-parameters are grouped into small dataclasses so
# the constructor stays readable. Each group can be passed as the dataclass or as
# a plain dict (which is normalised to the dataclass); dicts are also what gets
# stored in the checkpoint (``asdict`` on save), so ``.mdlus`` round-trips remain
# JSON-serialisable.


@dataclass
class S2ProcessorConfig:
    r"""Processor (latent) transformer stack: block layout + self-attention."""

    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    attention_mode: Union[str, List[str]] = "neighborhood"  # or a per-layer list
    attn_theta_cutoff_factor: float = 2.0  # neighborhood radius (lat rings)
    attn_drop_rate: float = 0.0  # global-attention softmax dropout
    attn_optimized_kernel: bool = True
    qk_norm: bool = False
    mlp_ratio: float = 2.0
    activation: str = "gelu"
    normalization: str = "layer_norm"
    use_mlp: bool = True
    path_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0


@dataclass
class S2ResamplingConfig:
    r"""Encoder/decoder cross-resolution attention settings.

    The encoder/decoder are activated automatically whenever the grid shape or the
    channel count changes between the input/output grids and the processor (latent)
    grid; these settings only tune them when active.

    ``decoder_theta_cutoff_factor=None`` uses the decoder's upsampling ratio
    :math:`(H_{out} - 1) / (h - 1)` (at least 1), the smallest neighborhood that
    reaches a latent point from every output point.
    """

    encoder_theta_cutoff_factor: float = 1.0
    decoder_theta_cutoff_factor: Optional[float] = None
    upsample_sht: bool = False


@dataclass
class S2PosEmbedConfig:
    r"""Processor-grid positional embedding settings."""

    kind: str = "spectral"  # "none" or "spectral"
    application: str = "input"  # "input" (legacy) or per-block "qk"
    max_degree: Optional[int] = None
    drop_rate: float = 0.0


def _as_cfg(value: Union[Any, dict, None], cls: type):
    r"""Normalise a config group to its dataclass (accepts dataclass / dict / None)."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if is_dataclass(value):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__}, dict, or None; got {type(value)!r}.")


class S2Transformer(Module):
    r"""Fully attention-based spherical transformer (S2Transformer) backbone.

    An encoder / processor / decoder network for gridded geophysical fields
    (developed for precipitation diagnostics). The internal (processor) grid is
    either derived from ``scale_factor`` (``h, w = inp_shape // scale_factor``)
    or set explicitly via ``latent_shape=(h, w)``, which takes precedence. The
    latent longitude ``w`` MUST divide the input longitude (a hard constraint of
    the spherical-attention p-shift); latitude ``h`` is free.

    The spherical-attention encoder / decoder are activated automatically only
    when the grid shape or the channel count changes between the input/output
    grids and the processor grid. When the input grid already equals the processor
    grid (``scale_factor=1`` or ``latent_shape=inp_shape``) and
    ``in_channels == embed_dim`` (and likewise ``embed_dim == out_channels`` for
    the output side), the encoder/decoder collapse to ``nn.Identity()`` and the
    processor runs directly on the input at full resolution. Note that whenever
    the encoder is active, spherical attention requires ``num_heads | in_channels``.

    Hyper-parameters are grouped into small dataclasses
    (:class:`S2ProcessorConfig`, :class:`S2ResamplingConfig`,
    :class:`S2PosEmbedConfig`). Each group may be passed as its dataclass or as a
    plain dict (normalised internally); ``None`` uses the group's defaults.

    Parameters
    ----------
    model_grid_type : str, optional, default="equiangular"
        Backward-compatible quadrature grid type of the input/output grids.
        Used for either side when ``inp_grid_type`` or ``out_grid_type`` is
        omitted.
    sht_grid_type : str, optional, default="legendre-gauss"
        Quadrature grid type of the internal (processor) grid.
    inp_shape : tuple of int, optional, default=(721, 1440)
        Input grid shape :math:`(H_{in}, W_{in})`.
    out_shape : tuple of int, optional, default=(721, 1440)
        Output grid shape :math:`(H_{out}, W_{out})`.
    in_channels : int, optional, default=27
        Number of input channels.
    out_channels : int, optional, default=1
        Number of output channels.
    scale_factor : int, optional, default=8
        Spatial downsampling factor for the processor grid (ignored when
        ``latent_shape`` is given).
    latent_shape : tuple of int, optional, default=None
        Explicit processor grid shape :math:`(h, w)` (takes precedence over
        ``scale_factor``).
    channel_names : list of str, optional, default=None
        Optional channel names (metadata only).
    big_skip : bool, optional, default=False
        Whether to add a learned 1x1 residual from the input to the output.
    bias : bool, optional, default=True
        Whether the linear projections use a bias.
    checkpointing_level : int, optional, default=0
        Activation-checkpointing aggressiveness.
    processor : S2ProcessorConfig or dict, optional, default=None
        Processor stack + self-attention settings (``embed_dim``, ``num_layers``,
        ``num_heads``, ``attention_mode``, ``attn_theta_cutoff_factor``,
        ``attn_drop_rate``, ``attn_optimized_kernel``, ``qk_norm``, ``mlp_ratio``,
        ``activation``, ``normalization``, ``use_mlp``, ``path_drop_rate``,
        ``mlp_drop_rate``). ``embed_dim`` must be divisible by ``num_heads``, and
        ``attention_mode`` may be ``"neighborhood"``, ``"global"``, or a per-layer
        list.
    resampling : S2ResamplingConfig or dict, optional, default=None
        Encoder/decoder tuning (``encoder_theta_cutoff_factor``,
        ``decoder_theta_cutoff_factor``, ``upsample_sht``). The encoder/decoder
        are activated automatically when the grid shape or channel count differs
        between the input/output grids and the processor grid.
    pos_embed : S2PosEmbedConfig or dict, optional, default=None
        Positional-embedding settings (``kind`` in ``{"none", "spectral"}``,
        ``application`` in ``{"input", "qk"}``, ``max_degree``, ``drop_rate``).
        ``"qk"`` adds the embedding to the attention queries and keys of every
        processor block instead of to the tokens.
    inp_grid_type : str, optional, default=None
        Quadrature grid type of the input grid. ``None`` preserves legacy
        behavior by using ``model_grid_type``.
    out_grid_type : str, optional, default=None
        Quadrature grid type of the output grid. ``None`` preserves legacy
        behavior by using ``model_grid_type``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H_{in}, W_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H_{out}, W_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.s2transformer import (
    ...     S2ProcessorConfig,
    ...     S2Transformer,
    ... )
    >>> model = S2Transformer(
    ...     inp_shape=(16, 32),
    ...     out_shape=(16, 32),
    ...     in_channels=4,
    ...     out_channels=1,
    ...     scale_factor=1,
    ...     processor=S2ProcessorConfig(
    ...         embed_dim=16,
    ...         num_layers=2,
    ...         num_heads=4,
    ...         attention_mode="global",
    ...         attn_optimized_kernel=False,
    ...     ),
    ... )
    >>> model(torch.randn(2, 4, 16, 32)).shape
    torch.Size([2, 1, 16, 32])
    """

    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": (
            "Converting an S2Transformer 0.1.0 checkpoint: dropping the unused "
            "'noise' argument."
        ),
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""Map constructor arguments of older checkpoints to the current format.

        Checkpoints with version ``0.1.0`` carry a ``noise`` argument for a
        conditioning mode that this model no longer provides. Deterministic
        checkpoints (``noise["mode"] is None``) are converted by dropping it.

        Parameters
        ----------
        version : str
            Version of the checkpoint being loaded.
        args : Dict[str, Any]
            Constructor arguments stored in the checkpoint.

        Returns
        -------
        Dict[str, Any]
            Constructor arguments accepted by the current version.

        Raises
        ------
        ValueError
            If the checkpoint was trained with noise conditioning enabled.
        """
        args = super()._backward_compat_arg_mapper(version, args)
        if version == "0.1.0" and "noise" in args:
            noise = args.pop("noise") or {}
            if noise.get("mode") is not None:
                raise ValueError(
                    "S2Transformer checkpoint uses noise.mode="
                    f"{noise['mode']!r}, which is not supported."
                )
        return args

    def __init__(
        self,
        model_grid_type: str = "equiangular",
        sht_grid_type: str = "legendre-gauss",
        inp_shape: Tuple[int, int] = (721, 1440),
        out_shape: Tuple[int, int] = (721, 1440),
        in_channels: int = 27,
        out_channels: int = 1,
        scale_factor: int = 8,
        latent_shape: Optional[Tuple[int, int]] = None,
        channel_names: Optional[List[str]] = None,
        big_skip: bool = False,
        bias: bool = True,
        checkpointing_level: int = 0,
        processor: Union[S2ProcessorConfig, dict, None] = None,
        resampling: Union[S2ResamplingConfig, dict, None] = None,
        pos_embed: Union[S2PosEmbedConfig, dict, None] = None,
        inp_grid_type: Optional[str] = None,
        out_grid_type: Optional[str] = None,
    ):
        super().__init__(meta=MetaData())

        if not TORCH_HARMONICS_AVAILABLE:
            raise ImportError(
                "S2Transformer requires the optional dependency "
                "'torch_harmonics' (>=0.9.3), which is not installed or too old.\n"
                "Install with:\n  pip install torch_harmonics>=0.9.3\n"
                "  pip install nvidia-physicsnemo[model-extras]"
            )

        # Grouped hyper-parameters (accept dataclass / dict / None).
        self.processor = _as_cfg(processor, S2ProcessorConfig)
        self.resampling = _as_cfg(resampling, S2ResamplingConfig)
        self.pos_embed_cfg = _as_cfg(pos_embed, S2PosEmbedConfig)

        self.pos_embed_application = self.pos_embed_cfg.application.lower()
        if self.pos_embed_application not in {"input", "qk"}:
            raise ValueError(
                "pos_embed.application must be 'input' or 'qk'; "
                f"got {self.pos_embed_cfg.application!r}."
            )
        if self.pos_embed_application == "qk" and self.pos_embed_cfg.drop_rate != 0.0:
            raise ValueError("Q/K positional embedding does not support dropout.")

        # Convenience scalar attributes used throughout the model / forward.
        self.inp_shape = tuple(inp_shape)
        self.out_shape = tuple(out_shape)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.embed_dim = int(self.processor.embed_dim)
        self.num_layers = int(self.processor.num_layers)
        self.big_skip = bool(big_skip)
        self.checkpointing_level = int(checkpointing_level)
        # Keep model_grid_type as the single legacy default while allowing the
        # encoder and decoder boundary grids to differ. These arguments are
        # explicit in the signature so PhysicsNeMo's Module constructor capture
        # serializes them into checkpoints.
        self.model_grid_type = model_grid_type
        self.sht_grid_type = sht_grid_type
        self.inp_grid_type = model_grid_type if inp_grid_type is None else inp_grid_type
        self.out_grid_type = model_grid_type if out_grid_type is None else out_grid_type

        _acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if self.processor.activation not in _acts:
            raise ValueError(f"Unknown activation function {self.processor.activation}")
        act_layer = _acts[self.processor.activation]

        # Internal (processor) spatial shape. Two ways to specify it:
        #   1. latent_shape=(h, w) - explicit, takes precedence.
        #   2. scale_factor - h,w = inp_shape // scale_factor.
        # The latent longitude (w) MUST divide the input longitude (a hard
        # constraint of the spherical-attention p-shift kernel); latitude (h)
        # is free.
        if latent_shape is not None:
            self.h = int(latent_shape[0])
            self.w = int(latent_shape[1])
        else:
            self.h = int(self.inp_shape[0] // scale_factor)
            self.w = int(self.inp_shape[1] // scale_factor)

        # The encoder/decoder are needed only when the grid shape OR the channel
        # count changes between the input/output grids and the processor (latent)
        # grid. When both match, they collapse to nn.Identity() and the processor
        # runs directly on the input.
        self.use_encoder = (tuple(self.inp_shape) != (self.h, self.w)) or (
            self.in_channels != self.embed_dim
        )
        self.use_decoder = (tuple(self.out_shape) != (self.h, self.w)) or (
            self.embed_dim != self.out_channels
        )
        if self.use_encoder and self.inp_shape[1] % self.w != 0:
            raise ValueError(
                f"S2Transformer: latent longitude w={self.w} must divide "
                f"input longitude W={self.inp_shape[1]} (spherical-attention "
                f"p-shift). Pick a w that divides {self.inp_shape[1]}."
            )
        proc = self.processor  # local alias for readability

        # Encoder/decoder: when off they are pure nn.Identity() and the channel
        # routing in_channels -> embed_dim -> out_channels is pushed into the
        # first/last processor block (the block's MLP changes channels). The
        # spatial shape cannot change without the attention encoder/decoder, so
        # use_encoder=False / use_decoder=False require the processor grid to
        # equal the input grid.
        if self.use_encoder:
            self.encoder = S2AttEncoder(
                inp_shape=self.inp_shape,
                out_shape=(self.h, self.w),
                in_chans=self.in_channels,
                out_chans=self.embed_dim,
                grid_in=self.inp_grid_type,
                grid_out=self.sht_grid_type,
                num_heads=int(proc.num_heads),
                theta_cutoff_factor=self.resampling.encoder_theta_cutoff_factor,
                qk_norm=proc.qk_norm,
                bias=bias,
                attn_optimized_kernel=proc.attn_optimized_kernel,
            )
        else:
            self.encoder = nn.Identity()

        if self.use_decoder:
            decoder_theta_cutoff_factor = self.resampling.decoder_theta_cutoff_factor
            if decoder_theta_cutoff_factor is None:
                decoder_theta_cutoff_factor = max(
                    1.0, (self.out_shape[0] - 1) / max(self.h - 1, 1)
                )
            self.decoder = S2AttDecoder(
                inp_shape=(self.h, self.w),
                out_shape=self.out_shape,
                in_chans=self.embed_dim,
                out_chans=self.out_channels,
                grid_in=self.sht_grid_type,
                grid_out=self.out_grid_type,
                num_heads=int(proc.num_heads),
                theta_cutoff_factor=decoder_theta_cutoff_factor,
                qk_norm=proc.qk_norm,
                bias=bias,
                attn_optimized_kernel=proc.attn_optimized_kernel,
                upsample_sht=self.resampling.upsample_sht,
            )
        else:
            self.decoder = nn.Identity()

        # Resolve per-block attention modes (allow list-of-strings for mix-and-match).
        if isinstance(proc.attention_mode, str):
            modes = [proc.attention_mode] * self.num_layers
        else:
            modes = list(proc.attention_mode)
            if len(modes) != self.num_layers:
                raise ValueError(
                    f"processor.attention_mode list length ({len(modes)}) must "
                    f"equal num_layers ({self.num_layers})."
                )

        attn_theta_cutoff = _neighborhood_radius_rad(
            self.h, proc.attn_theta_cutoff_factor
        )

        # Per-block channel routing:
        #   first block:  in_ch = in_channels   if use_encoder=False else embed_dim
        #   last block:   out_ch = out_channels if use_decoder=False else embed_dim
        #   middle blocks always operate at embed_dim. The channel change happens
        #   inside SphericalTransformerBlock's MLP.
        def _block_in_ch(i: int) -> int:
            if i == 0 and not self.use_encoder:
                return self.in_channels
            return self.embed_dim

        def _block_out_ch(i: int) -> int:
            if i == self.num_layers - 1 and not self.use_decoder:
                return self.out_channels
            return self.embed_dim

        # Legacy "input" PE is added once before the processor stack. "qk" PE
        # is instead reused by every block immediately before attention.
        first_in_ch = _block_in_ch(0)
        if self.pos_embed_application == "qk" and any(
            _block_in_ch(i) != first_in_ch for i in range(self.num_layers)
        ):
            raise ValueError("Q/K positional embedding requires one processor width.")
        pdrop = self.pos_embed_cfg.drop_rate
        self.pos_drop = nn.Dropout(p=pdrop) if pdrop > 0.0 else nn.Identity()
        self.pos_embed = _build_pos_embedding(
            self.pos_embed_cfg.kind,
            (self.h, self.w),
            first_in_ch,
            sht_grid_type,
            max_degree=self.pos_embed_cfg.max_degree,
        )

        dpr = [
            float(x)
            for x in torch.linspace(0, proc.path_drop_rate, max(1, self.num_layers))
        ]
        self.blocks = nn.ModuleList(
            [
                SphericalTransformerBlock(
                    in_shape=(self.h, self.w),
                    grid=sht_grid_type,
                    in_chans=_block_in_ch(i),
                    out_chans=_block_out_ch(i),
                    num_heads=int(proc.num_heads),
                    attention_mode=modes[i],
                    attn_theta_cutoff=attn_theta_cutoff,
                    mlp_ratio=proc.mlp_ratio,
                    mlp_drop_rate=proc.mlp_drop_rate,
                    attn_drop_rate=proc.attn_drop_rate,
                    path_drop_rate=dpr[i],
                    act_layer=act_layer,
                    normalization_layer=proc.normalization,
                    bias=bias,
                    use_mlp=proc.use_mlp,
                    qk_norm=proc.qk_norm,
                    attn_optimized_kernel=proc.attn_optimized_kernel,
                    checkpointing_level=self.checkpointing_level,
                )
                for i in range(self.num_layers)
            ]
        )

        if self.big_skip:
            self.residual_transform = nn.Conv2d(
                self.in_channels, self.out_channels, 1, bias=False
            )
            scale = math.sqrt(0.5 / max(1, self.in_channels))
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # Store JSON-serialisable (dict) forms of the config groups so that
        # Module.save() (which json-dumps __args__) round-trips; from_checkpoint
        # passes them back as dicts, which __init__ normalises to dataclasses.
        for _name, _cfg in (
            ("processor", self.processor),
            ("resampling", self.resampling),
            ("pos_embed", self.pos_embed_cfg),
        ):
            self._args["__args__"][_name] = asdict(_cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self, x: Float[torch.Tensor, "batch in_channels in_lat in_lon"]
    ) -> Float[torch.Tensor, "batch embed_dim lat lon"]:
        r"""Encode the input onto the processor grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C_{in}, H_{in}, W_{in})`.

        Returns
        -------
        torch.Tensor
            Latent tensor of shape :math:`(B, D, h, w)`, where :math:`D` is
            ``processor.embed_dim`` when the encoder is active.
        """
        if not torch.compiler.is_compiling():
            _check_bchw(x, self.in_channels, self.inp_shape)
        return self.encoder(x.contiguous())

    def decode(
        self, x: Float[torch.Tensor, "batch embed_dim lat lon"]
    ) -> Float[torch.Tensor, "batch out_channels out_lat out_lon"]:
        r"""Decode the processor output onto the output grid.

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor of shape :math:`(B, D, h, w)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, C_{out}, H_{out}, W_{out})`.
        """
        if not torch.compiler.is_compiling() and len(self.blocks) > 0:
            _check_bchw(x, self.blocks[-1].out_chans, (self.h, self.w))
        return self.decoder(x)

    def processor_blocks(
        self, x: Float[torch.Tensor, "batch embed_dim lat lon"]
    ) -> Float[torch.Tensor, "batch embed_dim lat lon"]:
        r"""Apply the positional embedding and the stack of processor blocks.

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor of shape :math:`(B, D, h, w)`.

        Returns
        -------
        torch.Tensor
            Processed latent tensor of shape :math:`(B, D', h, w)`.
        """
        if not torch.compiler.is_compiling() and len(self.blocks) > 0:
            _check_bchw(x, self.blocks[0].in_chans, (self.h, self.w))
        if self.pos_embed_application == "input":
            x = self.pos_embed(self.pos_drop(x))
        qk_pos_embed = (
            getattr(self.pos_embed, "position_embeddings", None)
            if self.pos_embed_application == "qk"
            else None
        )
        for blk in self.blocks:
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, qk_pos_embed, use_reentrant=False)
            else:
                x = blk(x, qk_pos_embed=qk_pos_embed)
        return x

    def forward(
        self, x: Float[torch.Tensor, "batch in_channels in_lat in_lon"]
    ) -> Float[torch.Tensor, "batch out_channels out_lat out_lon"]:
        r"""Run the encoder, processor and decoder."""
        if not torch.compiler.is_compiling():
            _check_bchw(x, self.in_channels, self.inp_shape)
        residual = x.contiguous() if self.big_skip else None

        if self.checkpointing_level >= 1:
            x = checkpoint(self.encode, x, use_reentrant=False)
        else:
            x = self.encode(x)

        x = self.processor_blocks(x)

        if self.checkpointing_level >= 1:
            x = checkpoint(self.decode, x, use_reentrant=False)
        else:
            x = self.decode(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)
        return x
