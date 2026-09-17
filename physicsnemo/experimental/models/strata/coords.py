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

r"""Token-coordinate builders for Strata RoPE.

Pure, instance-free helpers that build the per-token ``(x, y)`` coordinates fed to
the 2D RoPE table builder (``build_axial_rope_cos_sin_2d_continuous``)
in :class:`~physicsnemo.experimental.models.strata.StrataTransformer3D` and
:class:`~physicsnemo.experimental.models.strata.Strata`. Keeping them as free
functions (rather than methods on a model) decouples the two stages — the pixel
stage no longer reaches into the backbone stage's instance state — and makes the
geometry independently unit-testable.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from jaxtyping import Float

from physicsnemo.experimental.nn import spherical_centroid, stereographic_projection


def build_axial_token_coords(
    d: int, h: int, w: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    r"""Build integer ``(row, col)`` token coordinates for axial RoPE.

    The same ``(row, col)`` pair is tiled across every depth level, so depth tokens
    that share a horizontal location get identical horizontal positions.

    Parameters
    ----------
    d : int
        Number of depth tokens.
    h : int
        Number of row (height) tokens.
    w : int
        Number of column (width) tokens.
    device : torch.device, optional
        Device for the returned tensor. If ``None``, uses the default device.

    Returns
    -------
    torch.Tensor
        Coordinates of shape :math:`(d \cdot h \cdot w, 2)`.
    """
    ii = torch.arange(h, dtype=torch.float32, device=device)
    jj = torch.arange(w, dtype=torch.float32, device=device)
    imesh, jmesh = torch.meshgrid(ii, jj, indexing="ij")
    ij_hw = torch.stack([imesh, jmesh], dim=-1).reshape(h * w, 2)
    return ij_hw.repeat(d, 1)


def build_stereographic_token_coords(
    pos: Float[torch.Tensor, "batch 2 height width"],
    patch_hw: Union[int, Tuple[int, int]],
    d_patch: int,
    length_scale: float,
) -> Float[torch.Tensor, "batch tokens 2"]:
    r"""Build stereographic token coordinates from latitude / longitude.

    Pools the per-pixel lat/lon over each ``(p_h, p_w)`` patch (pole- and
    seam-robust :func:`~physicsnemo.experimental.nn.spherical_centroid`), projects
    the patch centers onto the tile-tangent plane via
    :func:`~physicsnemo.experimental.nn.stereographic_projection`, normalizes by
    ``length_scale``, and tiles the result across the depth axis. Used at patch
    resolution by ``StrataTransformer3D`` and at pixel resolution (``patch_hw=1``) by ``Strata``.

    Parameters
    ----------
    pos : torch.Tensor
        Latitude / longitude in radians of shape :math:`(B, 2, H, W)`.
    patch_hw : int | Tuple[int, int]
        Horizontal patch size ``(p_h, p_w)`` (or a single int for both).
    d_patch : int
        Number of depth tokens to tile the horizontal coordinates over.
    length_scale : float
        Positive coordinate normalization divisor.

    Returns
    -------
    torch.Tensor
        Token coordinates of shape :math:`(B, d\_patch \cdot h \cdot w, 2)`.
    """
    if length_scale <= 0:
        raise ValueError(f"length_scale must be > 0, got {length_scale}")
    if isinstance(patch_hw, int):
        patch_hw = (patch_hw, patch_hw)
    ph, pw = patch_hw

    lat = pos[:, 0]  # (B, H, W)
    lon = pos[:, 1]  # (B, H, W)

    # Tile center over the full grid: pole- and seam-robust spherical centroid.
    lat0, lon0 = spherical_centroid(lat, lon, reduce_dims=(1, 2))  # (B, 1, 1)

    # Pool lat/lon to the patch grid via the same spherical centroid.
    lat_resh = rearrange(lat, "b (h ph) (w pw) -> b h w ph pw", ph=ph, pw=pw)
    lon_resh = rearrange(lon, "b (h ph) (w pw) -> b h w ph pw", ph=ph, pw=pw)
    lat_p, lon_p = spherical_centroid(lat_resh, lon_resh, reduce_dims=(3, 4))
    lat_p = lat_p.squeeze((-2, -1))  # (B, h, w)
    lon_p = lon_p.squeeze((-2, -1))  # (B, h, w)

    # Project to the tangent plane, normalize, flatten, then tile over depth.
    x_hw, y_hw = stereographic_projection(lat_p, lon_p, lat0, lon0)  # each (B, h, w)
    x_hw, y_hw = x_hw / length_scale, y_hw / length_scale
    coords_hw = torch.stack([x_hw, y_hw], dim=-1)  # (B, h, w, 2)
    coords_hw = rearrange(coords_hw, "b h w c -> b (h w) c")
    coords = coords_hw.unsqueeze(1).expand(-1, d_patch, -1, -1)  # (B, d, h*w, 2)
    return rearrange(coords, "b d hw c -> b (d hw) c")
