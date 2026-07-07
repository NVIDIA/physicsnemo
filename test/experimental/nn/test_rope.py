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

"""Tests for the experimental continuous / stereographic RoPE helpers."""

import math

import pytest
import torch

from physicsnemo.experimental.nn import (
    build_axial_rope_cos_sin_2d_continuous,
    build_rope_cos_sin_1d_continuous,
    spherical_centroid,
    stereographic_projection,
)
from physicsnemo.nn import (
    apply_rotary_pos_emb,
    build_axial_rope_cos_sin_2d,
    build_rope_cos_sin_1d,
)


@torch.no_grad()
def test_build_rope_cos_sin_1d_continuous_matches_1d():
    """Integer positions reproduce build_rope_cos_sin_1d (its continuous twin)."""
    seq_len, head_dim = 10, 16
    pos = torch.arange(seq_len).float()
    cos_c, sin_c = build_rope_cos_sin_1d_continuous(pos, head_dim)
    cos_i, sin_i = build_rope_cos_sin_1d(seq_len, head_dim)
    assert cos_c.shape == (seq_len, head_dim)
    assert torch.allclose(cos_c, cos_i, atol=1e-6)
    assert torch.allclose(sin_c, sin_i, atol=1e-6)
    # dim must be even.
    with pytest.raises(ValueError):
        build_rope_cos_sin_1d_continuous(pos, dim=15)


@torch.no_grad()
def test_build_axial_rope_cos_sin_2d_continuous_matches_axial():
    """Integer row/col coordinates reproduce build_axial_rope_cos_sin_2d (flattened)."""
    h, w, head_dim = 3, 5, 16
    rows = torch.arange(h).reshape(h, 1).expand(h, w).reshape(-1).float()
    cols = torch.arange(w).reshape(1, w).expand(h, w).reshape(-1).float()
    cos2, sin2 = build_axial_rope_cos_sin_2d_continuous(rows, cols, head_dim)
    cos_ax, sin_ax = build_axial_rope_cos_sin_2d(h, w, head_dim)
    assert cos2.shape == (h * w, head_dim)
    assert torch.allclose(cos2, cos_ax.reshape(h * w, head_dim), atol=1e-6)
    assert torch.allclose(sin2, sin_ax.reshape(h * w, head_dim), atol=1e-6)
    # head_dim must be divisible by 4.
    with pytest.raises(ValueError):
        build_axial_rope_cos_sin_2d_continuous(rows, cols, head_dim=6)


@torch.no_grad()
def test_stereographic_projection_geometry():
    """Center maps to the origin; East gives x > 0, North gives y > 0."""
    zero = torch.zeros(1, 1, 1)
    x, y = stereographic_projection(
        torch.zeros(1, 3, 3), torch.zeros(1, 3, 3), zero, zero
    )
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
    x_east, _ = stereographic_projection(zero, torch.full((1, 1, 1), 0.2), zero, zero)
    assert float(x_east) > 0.0
    _, y_north = stereographic_projection(torch.full((1, 1, 1), 0.2), zero, zero, zero)
    assert float(y_north) > 0.0


@torch.no_grad()
def test_stereographic_projection_matches_closed_form():
    """Projection equals the analytic stereographic formula, pinning exact
    numerics (the sign-only geometry test above would miss a wrong scale)."""
    zero = torch.zeros(1, 1, 1)
    d = 0.3
    # The closed form along a single meridian / parallel from center (0, 0) is
    # 2 * tan(delta / 2): East displacement -> x, North displacement -> y.
    expected = 2.0 * math.tan(d / 2)
    x, y = stereographic_projection(zero, torch.full((1, 1, 1), d), zero, zero)
    assert torch.allclose(x, torch.full_like(x, expected), atol=1e-6)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
    x2, y2 = stereographic_projection(torch.full((1, 1, 1), d), zero, zero, zero)
    assert torch.allclose(x2, torch.zeros_like(x2), atol=1e-6)
    assert torch.allclose(y2, torch.full_like(y2, expected), atol=1e-6)


@torch.no_grad()
def test_spherical_centroid_handles_pole_and_seam():
    """The 3D vector centroid centers a ring around the North Pole at +pi/2 (where
    a plain latitude mean undershoots), and a longitude ring straddling the
    0 / 2*pi seam near 0 (not pi)."""
    # Ring of points at 89 deg latitude spanning all longitudes -> center = pole.
    lon_ring = torch.linspace(0.0, 2 * torch.pi, 8)[:-1]  # 7 evenly spaced, exclusive
    lat_ring = torch.full_like(lon_ring, math.radians(89.0))
    lat0, _ = spherical_centroid(lat_ring.reshape(1, 1, -1), lon_ring.reshape(1, 1, -1))
    assert torch.allclose(lat0.reshape(()), torch.tensor(math.pi / 2), atol=1e-3)
    # The plain mean of latitude would undershoot the pole.
    assert float(lat_ring.mean()) < math.pi / 2 - 1e-3
    # Longitude seam: points at +/-0.1 around 0 center near 0, not pi.
    lat_eq = torch.zeros(1, 1, 2)
    lon_seam = torch.tensor([[[0.1, 2 * torch.pi - 0.1]]])
    _, lon0 = spherical_centroid(lat_eq, lon_seam)
    wrapped = (lon0.reshape(()) + torch.pi) % (2 * torch.pi) - torch.pi  # to [-pi, pi)
    assert abs(float(wrapped)) < 1e-5


@torch.no_grad()
def test_stereographic_projection_finite_near_antipode():
    """The antipodal singularity is guarded: outputs stay finite, not inf/nan."""
    zero = torch.zeros(1, 1, 1)
    # A point at the antipode of the center (dlon = pi, same latitude) has
    # cos_c = -1, the projection's singular point.
    lat = torch.zeros(1, 1, 1)
    lon = torch.full((1, 1, 1), float(torch.pi))
    x, y = stereographic_projection(lat, lon, zero, zero)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()


@torch.no_grad()
def test_continuous_rope_relative_position_invariance():
    """RoPE encodes relative position: shifting all coordinates by a constant
    leaves the query-key score matrix unchanged (tables + apply_rotary_pos_emb)."""
    torch.manual_seed(0)
    head_dim = 16
    q = torch.randn(1, 2, 6, head_dim)
    k = torch.randn(1, 2, 6, head_dim)
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)

    def rotate(xp, yp):
        cos, sin = build_axial_rope_cos_sin_2d_continuous(xp, yp, head_dim)
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)

    q1, k1 = rotate(x_pos, y_pos)
    q2, k2 = rotate(x_pos + 0.7, y_pos - 1.3)
    assert torch.allclose(
        q1 @ k1.transpose(-1, -2), q2 @ k2.transpose(-1, -2), atol=1e-4
    )


@torch.no_grad()
def test_continuous_rope_theta_changes_tables():
    """The theta base is wired through: a different theta gives different tables."""
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)
    cos_a, sin_a = build_axial_rope_cos_sin_2d_continuous(
        x_pos, y_pos, 16, theta=10000.0
    )
    cos_b, sin_b = build_axial_rope_cos_sin_2d_continuous(x_pos, y_pos, 16, theta=100.0)
    assert not torch.allclose(cos_a, cos_b)
    assert not torch.allclose(sin_a, sin_b)


@torch.no_grad()
def test_continuous_rope_preserves_low_precision_dtype():
    """bf16 q/k come back as bf16 (apply_rotary_pos_emb rotates in fp32 internally)."""
    torch.manual_seed(0)
    head_dim = 16
    q = torch.randn(2, 4, 6, head_dim, dtype=torch.bfloat16)
    k = torch.randn(2, 4, 6, head_dim, dtype=torch.bfloat16)
    cos, sin = build_axial_rope_cos_sin_2d_continuous(
        torch.randn(6), torch.randn(6), head_dim
    )
    q_rot = apply_rotary_pos_emb(q, cos, sin)
    k_rot = apply_rotary_pos_emb(k, cos, sin)
    assert q_rot.dtype == torch.bfloat16 and k_rot.dtype == torch.bfloat16
    assert torch.isfinite(q_rot.float()).all() and torch.isfinite(k_rot.float()).all()
