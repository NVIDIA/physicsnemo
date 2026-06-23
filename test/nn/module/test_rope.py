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

import pytest
import torch

from physicsnemo.core import Module
from physicsnemo.nn.module.rope import (
    RotaryPositionEmbedding1D,
    RotaryPositionEmbedding2D,
    apply_rotary_pos_emb,
    build_axial_rope_cos_sin_2d,
    build_rope_cos_sin_1d,
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

    cos, sin = build_axial_rope_cos_sin_2d(h, w, head_dim)
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

    cos, sin = build_axial_rope_cos_sin_2d(h, w, head_dim)  # (h, w, head_dim)
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
    cos, sin = build_axial_rope_cos_sin_2d(3, 4, head_dim)
    cos_flat, sin_flat = cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)

    x = torch.randn(2, n, head_dim, dtype=torch.float32)
    x_rot = apply_rotary_pos_emb(x, cos_flat, sin_flat)
    assert x_rot.dtype == x.dtype

    # Rotation preserves each channel pair's norm.
    pair_in = x[..., 0::2].square() + x[..., 1::2].square()
    pair_out = x_rot[..., 0::2].square() + x_rot[..., 1::2].square()
    assert torch.allclose(pair_in, pair_out, atol=1e-5)

    # Sanity: a uniform 90-degree rotation (cos=0, sin=1) is the rotate-half
    # operation; applying it twice negates x (rotation by 90 deg twice).
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    once = apply_rotary_pos_emb(x, zeros, ones)
    twice = apply_rotary_pos_emb(once, zeros, ones)
    assert torch.allclose(twice, -x, atol=1e-6)


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


# --- physicsnemo.Module checkpoint round-trips ---
#
# Both RoPE modules subclass physicsnemo.core.Module, so they must support the
# .save() / Module.from_checkpoint() recipe. Their cos/sin tables are
# persistent=False buffers, deterministically rebuilt from the __init__ args, so
# a round-trip must reproduce the forward exactly without the tables appearing in
# the checkpoint.


@torch.no_grad()
def test_rotary_2d_module_checkpoint_round_trip(tmp_path):
    head_dim, h, w = 16, 4, 5
    rope = RotaryPositionEmbedding2D(head_dim=head_dim, latent_hw=(h, w), theta=5000.0)
    assert isinstance(rope, Module)
    # persistent=False: tables are not serialized.
    assert "cos" not in rope.state_dict() and "sin" not in rope.state_dict()

    q = torch.randn(2, 3, h * w, head_dim)
    k = torch.randn(2, 3, h * w, head_dim)
    q_ref, k_ref = rope(q, k)

    path = str(tmp_path / "rope2d.mdlus")
    rope.save(path)
    loaded = Module.from_checkpoint(path)
    # Tables were rebuilt at the right shape from the saved __init__ args.
    assert loaded.cos.shape == (h * w, head_dim)
    assert loaded.theta == 5000.0
    q_out, k_out = loaded(q, k)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)


@torch.no_grad()
def test_rotary_1d_module_checkpoint_round_trip(tmp_path):
    head_dim, max_seq_len = 16, 32
    rope = RotaryPositionEmbedding1D(
        head_dim=head_dim, max_seq_len=max_seq_len, theta=5000.0
    )
    assert isinstance(rope, Module)
    assert "cos" not in rope.state_dict() and "sin" not in rope.state_dict()

    q = torch.randn(2, 4, 20, head_dim)
    k = torch.randn(2, 4, 20, head_dim)
    q_ref, k_ref = rope(q, k)

    path = str(tmp_path / "rope1d.mdlus")
    rope.save(path)
    loaded = Module.from_checkpoint(path)
    assert loaded.cos.shape == (max_seq_len, head_dim)
    q_out, k_out = loaded(q, k)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)
