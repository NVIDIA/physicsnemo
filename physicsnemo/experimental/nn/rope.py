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

r"""Continuous-coordinate and stereographic RoPE helpers (experimental).

Continuous-position generalizations of the integer-grid RoPE table builders in
:mod:`physicsnemo.nn.module.rope`, plus the spherical-geometry helpers that turn
latitude / longitude into tangent-plane coordinates for a stereographic 2D RoPE.
The tables compose with :func:`physicsnemo.nn.apply_rotary_pos_emb`.

These live under ``experimental`` while the continuous / stereographic RoPE API
matures; the production integer-grid builders remain in
:mod:`physicsnemo.nn.module.rope`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "stereographic_projection",
    "spherical_centroid",
    "build_rope_cos_sin_1d_continuous",
    "build_axial_rope_cos_sin_2d_continuous",
]


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
    antipodal / whole-sphere spread), which is outside the local-tile scope these
    helpers target.
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

    The continuous-position analog of
    :func:`~physicsnemo.nn.build_rope_cos_sin_1d` (which takes an integer
    ``seq_len`` and generates positions :math:`0 \ldots seq\_len - 1`): here the
    positions are supplied explicitly and may be any real values. All ``dim``
    channels rotate by the position, with ``dim/2`` frequencies
    :math:`\theta_k = \text{theta}^{-2k/dim}`, each driving the adjacent channel pair
    ``(2k, 2k+1)`` so the result composes with
    :func:`~physicsnemo.nn.apply_rotary_pos_emb`.

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

    The continuous-coordinate analog of
    :func:`~physicsnemo.nn.build_axial_rope_cos_sin_2d` (which takes integer grid
    sizes ``h, w`` and generates row/column indices): here the per-token
    ``(x, y)`` coordinates are supplied explicitly and may be any real values.
    ``head_dim`` is split in half — the first half rotates by ``x_pos``, the second by
    ``y_pos`` — each half built with :func:`build_rope_cos_sin_1d_continuous` over
    ``head_dim/2`` channels, so the result composes with
    :func:`~physicsnemo.nn.apply_rotary_pos_emb`. Passing integer row / column
    indices reproduces :func:`~physicsnemo.nn.build_axial_rope_cos_sin_2d`
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
