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

import math

import pytest
import torch

from physicsnemo.nn.module.rope import (
    RotaryPositionEmbedding1D,
    RotaryPositionEmbedding2D,
    StereographicRotaryPositionEmbedding2D,
    apply_rotary_pos_emb,
    build_axial_rope_cos_sin,
    build_axial_rope_cos_sin_continuous,
    build_rope_cos_sin_1d,
    build_rope_cos_sin_1d_continuous,
    rotate_half_pairs,
    spherical_centroid,
    stereographic_projection,
)


@torch.no_grad()
def test_rotary_module_shapes_and_validation():
    head_dim, h, w = 16, 4, 5
    rope = RotaryPositionEmbedding2D(head_dim=head_dim, latent_hw=(h, w))
    # Tables are flattened to (h*w, head_dim) so they broadcast over (..., N, D).
    assert rope.cos.shape == (h * w, head_dim)
    assert rope.sin.shape == (h * w, head_dim)
    assert "cos" not in rope.state_dict()  # persistent=False

    q = torch.randn(2, 8, h * w, head_dim)
    k = torch.randn(2, 8, h * w, head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape

    # head_dim must be divisible by 4.
    with pytest.raises(ValueError):
        RotaryPositionEmbedding2D(head_dim=6, latent_hw=(h, w))

    # Wrong sequence length is rejected.
    with pytest.raises(ValueError):
        rope(torch.randn(2, 8, h * w + 1, head_dim), k)


@torch.no_grad()
def test_rotary_module_matches_flattened_tables():
    """The module result must equal applying the flattened cos/sin directly."""
    torch.manual_seed(0)
    head_dim, h, w = 32, 6, 4
    rope = RotaryPositionEmbedding2D(head_dim=head_dim, latent_hw=(h, w))
    q = torch.randn(3, 4, h * w, head_dim)

    cos, sin = build_axial_rope_cos_sin(h, w, head_dim)
    cos_flat = cos.reshape(-1, head_dim)
    sin_flat = sin.reshape(-1, head_dim)
    expected = apply_rotary_pos_emb(q, cos_flat, sin_flat)

    q_rot, _ = rope(q, q)
    assert torch.equal(q_rot, expected)


@torch.no_grad()
def test_rotary_module_layout_matches_spatial_rotation():
    """Rotating a flattened (B, H, N, D) tensor with the module must match
    rotating the spatial (B, H, h, w, D) tensor with the raw tables and then
    flattening — i.e. the module's row-major (h, then w) assumption holds."""
    torch.manual_seed(0)
    head_dim, h, w = 16, 3, 5
    B, heads = 2, 4

    cos, sin = build_axial_rope_cos_sin(h, w, head_dim)  # (h, w, head_dim)
    q_spatial = torch.randn(B, heads, h, w, head_dim)
    spatial_rot = apply_rotary_pos_emb(
        q_spatial, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    )
    spatial_rot_flat = spatial_rot.reshape(B, heads, h * w, head_dim)

    rope = RotaryPositionEmbedding2D(head_dim=head_dim, latent_hw=(h, w))
    q_flat = q_spatial.reshape(B, heads, h * w, head_dim)
    module_rot, _ = rope(q_flat, q_flat)

    assert torch.allclose(module_rot, spatial_rot_flat, atol=1e-6)


@torch.no_grad()
def test_rotary_module_rebuild_for_new_shape():
    head_dim = 16
    rope = RotaryPositionEmbedding2D(head_dim=head_dim, latent_hw=(4, 4))
    q = torch.randn(1, 2, 5 * 6, head_dim)
    # Passing a new latent_hw rebuilds the tables in place.
    q_rot, _ = rope(q, q, latent_hw=(5, 6))
    assert rope.cos.shape == (5 * 6, head_dim)
    assert q_rot.shape == q.shape


@torch.no_grad()
def test_apply_rotary_pos_emb_preserves_dtype_and_norm():
    torch.manual_seed(0)
    head_dim, n = 16, 12
    cos, sin = build_axial_rope_cos_sin(3, 4, head_dim)
    cos_flat, sin_flat = cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)

    x = torch.randn(2, n, head_dim, dtype=torch.float32)
    x_rot = apply_rotary_pos_emb(x, cos_flat, sin_flat)
    assert x_rot.dtype == x.dtype

    # Rotation preserves each channel pair's norm.
    pair_in = x[..., 0::2].square() + x[..., 1::2].square()
    pair_out = x_rot[..., 0::2].square() + x_rot[..., 1::2].square()
    assert torch.allclose(pair_in, pair_out, atol=1e-5)

    # Sanity: rotate_half_pairs applied twice negates (rotation by 90 deg twice).
    assert torch.allclose(rotate_half_pairs(rotate_half_pairs(x)), -x, atol=1e-6)


# --- 1D RoPE ---


@torch.no_grad()
def test_build_rope_cos_sin_1d_shape_and_validation():
    seq_len, head_dim = 10, 16
    cos, sin = build_rope_cos_sin_1d(seq_len, head_dim, theta=10000.0)
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    # Adjacent channels (2k, 2k+1) share a frequency.
    assert torch.allclose(cos[..., 0::2], cos[..., 1::2])
    assert torch.allclose(sin[..., 0::2], sin[..., 1::2])
    # Position 0 has zero angle: cos == 1, sin == 0.
    assert torch.allclose(cos[0], torch.ones(head_dim))
    assert torch.allclose(sin[0], torch.zeros(head_dim))
    # head_dim must be even.
    with pytest.raises(ValueError):
        build_rope_cos_sin_1d(seq_len, head_dim=15)


@torch.no_grad()
def test_rotary_1d_module_shapes_and_validation():
    head_dim, max_seq_len = 16, 32
    rope = RotaryPositionEmbedding1D(head_dim=head_dim, max_seq_len=max_seq_len)
    assert rope.cos.shape == (max_seq_len, head_dim)
    assert "cos" not in rope.state_dict()  # persistent=False

    q = torch.randn(2, 4, 20, head_dim)
    k = torch.randn(2, 4, 20, head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape

    with pytest.raises(ValueError):
        RotaryPositionEmbedding1D(head_dim=15, max_seq_len=max_seq_len)
    # Exceeding max_seq_len is rejected.
    with pytest.raises(ValueError):
        rope(torch.randn(2, 4, max_seq_len + 1, head_dim), k)
    # Mismatched q/k lengths are rejected.
    with pytest.raises(ValueError):
        rope(torch.randn(2, 4, 20, head_dim), torch.randn(2, 4, 19, head_dim))


@torch.no_grad()
def test_rotary_1d_module_matches_sliced_tables():
    """Shorter inputs use the leading positions of the precomputed tables."""
    torch.manual_seed(0)
    head_dim, max_seq_len = 32, 64
    rope = RotaryPositionEmbedding1D(head_dim=head_dim, max_seq_len=max_seq_len)

    seq_len = 40
    q = torch.randn(3, 4, seq_len, head_dim)
    cos, sin = build_rope_cos_sin_1d(max_seq_len, head_dim)
    expected = apply_rotary_pos_emb(q, cos[:seq_len], sin[:seq_len])

    q_rot, _ = rope(q, q)
    assert torch.equal(q_rot, expected)


@torch.no_grad()
def test_rotary_1d_relative_phase_is_translation_invariant():
    """RoPE encodes position as a relative rotation: the q.k inner product
    between positions i and j depends only on (i - j)."""
    torch.manual_seed(0)
    head_dim, max_seq_len = 16, 64
    rope = RotaryPositionEmbedding1D(head_dim=head_dim, max_seq_len=max_seq_len)

    # Same content at every position; rotate, then compare inner products of
    # pairs sharing the same offset.
    base = torch.randn(1, 1, 1, head_dim)
    seq = base.expand(1, 1, max_seq_len, head_dim).contiguous()
    q_rot, k_rot = rope(seq, seq)

    def dot(i, j):
        return (q_rot[0, 0, i] * k_rot[0, 0, j]).sum()

    # Offset of 3 gives the same score regardless of absolute position.
    assert torch.allclose(dot(5, 2), dot(20, 17), atol=1e-4)
    assert torch.allclose(dot(10, 4), dot(30, 24), atol=1e-4)


# --- Stereographic 2D RoPE ---


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
def test_build_axial_rope_cos_sin_continuous_matches_axial():
    """Integer row/col coordinates reproduce build_axial_rope_cos_sin (flattened)."""
    h, w, head_dim = 3, 5, 16
    rows = torch.arange(h).reshape(h, 1).expand(h, w).reshape(-1).float()
    cols = torch.arange(w).reshape(1, w).expand(h, w).reshape(-1).float()
    cos2, sin2 = build_axial_rope_cos_sin_continuous(rows, cols, head_dim)
    cos_ax, sin_ax = build_axial_rope_cos_sin(h, w, head_dim)
    assert cos2.shape == (h * w, head_dim)
    assert torch.allclose(cos2, cos_ax.reshape(h * w, head_dim), atol=1e-6)
    assert torch.allclose(sin2, sin_ax.reshape(h * w, head_dim), atol=1e-6)
    # head_dim must be divisible by 4.
    with pytest.raises(ValueError):
        build_axial_rope_cos_sin_continuous(rows, cols, head_dim=6)


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
def test_stereo_rope_module_shapes_and_validation():
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    q = torch.randn(2, 4, 6, 16)
    k = torch.randn(2, 4, 6, 16)
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)
    q_rot, k_rot = rope(q, k, x_pos, y_pos)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    # Rotation preserves the per-token norm.
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    # head_dim must be divisible by 4.
    with pytest.raises(ValueError):
        StereographicRotaryPositionEmbedding2D(head_dim=6)


@torch.no_grad()
def test_stereo_rope_relative_position_invariance():
    """RoPE encodes relative position: shifting all coordinates by a constant
    leaves the query-key score matrix unchanged."""
    torch.manual_seed(0)
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    q = torch.randn(1, 2, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)

    q1, k1 = rope(q, k, x_pos, y_pos)
    scores1 = q1 @ k1.transpose(-1, -2)
    q2, k2 = rope(q, k, x_pos + 0.7, y_pos - 1.3)
    scores2 = q2 @ k2.transpose(-1, -2)
    assert torch.allclose(scores1, scores2, atol=1e-4)


@torch.no_grad()
def test_stereo_rope_forward_batched_coords():
    """Per-sample (B, N) coordinates broadcast over heads automatically."""
    torch.manual_seed(0)
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    B, heads, n = 2, 4, 6
    q = torch.randn(B, heads, n, 16)
    k = torch.randn(B, heads, n, 16)
    x_pos = torch.randn(B, n)  # distinct coords per batch sample
    y_pos = torch.randn(B, n)
    q_rot, k_rot = rope(q, k, x_pos, y_pos)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    # Equivalent to manually inserting the heads axis into the tables.
    cos, sin = rope.build_tables(x_pos, y_pos)
    expected = apply_rotary_pos_emb(q, cos.unsqueeze(-3), sin.unsqueeze(-3))
    assert torch.allclose(q_rot, expected, atol=1e-6)


@torch.no_grad()
def test_stereo_rope_end_to_end_latlon_pipeline():
    """The intended usage path: project a lat/lon tile to coordinates, flatten to
    the token axis, then rotate (B, heads, N, head_dim) q/k."""
    torch.manual_seed(0)
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    b, heads, h, w = 2, 4, 3, 3
    n = h * w
    lat = torch.linspace(-0.2, 0.2, h).reshape(1, h, 1).expand(b, h, w)
    lon = torch.linspace(1.0, 1.4, w).reshape(1, 1, w).expand(b, h, w)
    x_pos, y_pos = rope.project(lat, lon, length_scale=0.1)
    x_pos = x_pos.reshape(b, n)  # flatten spatial dims to the token axis
    y_pos = y_pos.reshape(b, n)
    q = torch.randn(b, heads, n, 16)
    k = torch.randn(b, heads, n, 16)
    q_rot, k_rot = rope(q, k, x_pos, y_pos)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    # Per-token norm is preserved through the full project -> forward pipeline.
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


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
def test_stereo_rope_project_length_scale():
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    lat = torch.linspace(-0.3, 0.3, 4).reshape(1, 4, 1).expand(1, 4, 4)
    lon = torch.linspace(0.0, 0.5, 4).reshape(1, 1, 4).expand(1, 4, 4)
    # Happy path: positive length_scale gives finite coords; doubling it halves them.
    x1, y1 = rope.project(lat, lon, length_scale=1.0)
    x2, y2 = rope.project(lat, lon, length_scale=2.0)
    assert x1.shape == lat.shape and torch.isfinite(x1).all()
    assert torch.allclose(x2, x1 / 2, atol=1e-6) and torch.allclose(
        y2, y1 / 2, atol=1e-6
    )
    # length_scale is required and must be positive.
    with pytest.raises(ValueError):
        rope.project(lat, lon, length_scale=0.0)
    with pytest.raises(ValueError):
        rope.project(lat, lon, length_scale=-1.0)


@torch.no_grad()
def test_stereo_rope_project_centering_and_override():
    """Default centering uses the spherical centroid (3D vector mean); an explicit
    center overrides it and matches the bare projection."""
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    lat = torch.linspace(0.3, 0.7, 4).reshape(1, 4, 1).expand(1, 4, 4)
    lon = torch.linspace(1.0, 1.4, 4).reshape(1, 1, 4).expand(1, 4, 4)

    # Default center == spherical_centroid(lat, lon), not per-axis means.
    xd, yd = rope.project(lat, lon, length_scale=1.0)
    lat0_ref, lon0_ref = spherical_centroid(lat, lon, reduce_dims=(-2, -1))
    bx, by = stereographic_projection(lat, lon, lat0_ref, lon0_ref)
    assert torch.allclose(xd, bx, atol=1e-6) and torch.allclose(yd, by, atol=1e-6)

    # Longitude centering is circular: a field straddling the 0/2*pi seam stays
    # near the origin. Plain-mean centering would place the center ~pi away and
    # blow the coordinates up toward the antipode.
    lon_seam = (
        torch.tensor([0.05, 2 * torch.pi - 0.05]).reshape(1, 1, 2).expand(1, 2, 2)
    )
    lat_seam = torch.zeros(1, 2, 2)
    xs, ys = rope.project(lat_seam, lon_seam, length_scale=1.0)
    assert xs.abs().max() < 0.2 and ys.abs().max() < 0.2

    # Explicit center overrides the default and equals the bare projection / scale.
    lat0 = torch.full((1, 1, 1), 0.4)
    lon0 = torch.full((1, 1, 1), 1.0)
    xe, ye = rope.project(lat, lon, length_scale=2.0, lat0=lat0, lon0=lon0)
    ox, oy = stereographic_projection(lat, lon, lat0, lon0)
    assert torch.allclose(xe, ox / 2, atol=1e-6) and torch.allclose(
        ye, oy / 2, atol=1e-6
    )


@torch.no_grad()
def test_stereo_rope_forward_matches_manual_tables():
    """Shared (N,) coords: forward exactly equals applying build_tables directly
    (the non-batched broadcast branch, mirroring the axial module's table test)."""
    torch.manual_seed(0)
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    q = torch.randn(2, 4, 6, 16)
    k = torch.randn(2, 4, 6, 16)
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)
    cos, sin = rope.build_tables(x_pos, y_pos)
    expected_q = apply_rotary_pos_emb(q, cos, sin)
    expected_k = apply_rotary_pos_emb(k, cos, sin)
    q_rot, k_rot = rope(q, k, x_pos, y_pos)
    assert torch.equal(q_rot, expected_q) and torch.equal(k_rot, expected_k)


@torch.no_grad()
def test_stereo_rope_forward_preserves_low_precision_dtype():
    """bf16 q/k come back as bf16 (the rotation runs in fp32 internally)."""
    torch.manual_seed(0)
    rope = StereographicRotaryPositionEmbedding2D(head_dim=16)
    q = torch.randn(2, 4, 6, 16, dtype=torch.bfloat16)
    k = torch.randn(2, 4, 6, 16, dtype=torch.bfloat16)
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)
    q_rot, k_rot = rope(q, k, x_pos, y_pos)
    assert q_rot.dtype == torch.bfloat16 and k_rot.dtype == torch.bfloat16
    assert torch.isfinite(q_rot.float()).all() and torch.isfinite(k_rot.float()).all()


@torch.no_grad()
def test_stereo_rope_theta_changes_tables():
    """The theta base is wired through: a different theta gives different tables."""
    x_pos = torch.randn(6)
    y_pos = torch.randn(6)
    cos_a, sin_a = StereographicRotaryPositionEmbedding2D(
        head_dim=16, theta=10000.0
    ).build_tables(x_pos, y_pos)
    cos_b, sin_b = StereographicRotaryPositionEmbedding2D(
        head_dim=16, theta=100.0
    ).build_tables(x_pos, y_pos)
    assert not torch.allclose(cos_a, cos_b)
    assert not torch.allclose(sin_a, sin_b)
