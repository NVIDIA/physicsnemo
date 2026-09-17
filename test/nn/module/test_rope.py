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

from physicsnemo.core import Module
from physicsnemo.nn.module.rope import (
    RotaryEmbedding1DTables,
    RotaryEmbedding2DTables,
    apply_rotary_pos_emb,
    build_axial_rope_cos_sin_2d,
    build_rope_cos_sin_1d,
)


@torch.no_grad()
def test_rotary_2d_tables_shapes_and_validation():
    head_dim, h, w = 16, 4, 5
    rope = RotaryEmbedding2DTables(head_dim=head_dim, latent_hw=(h, w))
    # The provider owns the tables in explicit (h, w, head_dim) spatial layout,
    # named rope_cos/rope_sin so domain parallelism can shard them along height.
    assert rope.rope_cos.shape == (h, w, head_dim)
    assert rope.rope_sin.shape == (h, w, head_dim)
    assert "rope_cos" not in rope.state_dict()  # persistent=False

    # forward() returns the owned tables.
    cos, sin = rope()
    assert cos.shape == (h, w, head_dim)
    assert torch.equal(cos, rope.rope_cos) and torch.equal(sin, rope.rope_sin)

    # head_dim must be divisible by 4.
    with pytest.raises(ValueError):
        RotaryEmbedding2DTables(head_dim=6, latent_hw=(h, w))


@torch.no_grad()
def test_rotary_2d_tables_match_builder():
    """The provider's tables must equal the functional builder's output."""
    head_dim, h, w = 32, 6, 4
    rope = RotaryEmbedding2DTables(head_dim=head_dim, latent_hw=(h, w), theta=5000.0)
    cos, sin = rope()

    exp_cos, exp_sin = build_axial_rope_cos_sin_2d(h, w, head_dim, theta=5000.0)
    assert torch.equal(cos, exp_cos)
    assert torch.equal(sin, exp_sin)

    # Applying the tables in spatial (B, heads, h, w, head_dim) layout works.
    torch.manual_seed(0)
    q = torch.randn(2, 4, h, w, head_dim)
    q_rot = apply_rotary_pos_emb(
        q, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    )
    assert q_rot.shape == q.shape


@torch.no_grad()
def test_rotary_2d_tables_rebuild_for_new_shape():
    head_dim = 16
    rope = RotaryEmbedding2DTables(head_dim=head_dim, latent_hw=(4, 4))
    # Passing a new latent_hw rebuilds the tables in place and returns them.
    cos, sin = rope(latent_hw=(5, 6))
    assert cos.shape == (5, 6, head_dim)
    assert rope.rope_cos.shape == (5, 6, head_dim)
    assert rope._latent_hw == (5, 6)

    # Re-requesting the same shape is a no-op (no rebuild).
    same_cos, _ = rope(latent_hw=(5, 6))
    assert torch.equal(same_cos, cos)


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


@torch.no_grad()
def test_apply_rotary_pos_emb_rotation_direction():
    """Pin the handedness of the pair rotation, not just its magnitude.

    The norm- and relative-position tests are blind to the rotation's sign (a
    flipped ``rotate_half`` passes them all). Rotating a basis pair by a generic
    angle must follow a proper counter-clockwise rotation
    ``(x0, x1) -> (x0 cos - x1 sin, x0 sin + x1 cos)``: so ``(1, 0) -> (cos, sin)``
    and ``(0, 1) -> (-sin, cos)``. A flipped sign would send ``(1, 0) -> (cos, -sin)``.
    """
    theta = math.pi / 6
    c, s = math.cos(theta), math.sin(theta)
    cos = torch.full((1, 2), c)  # head_dim=2: a single rotated pair
    sin = torch.full((1, 2), s)
    e_x = torch.tensor([[1.0, 0.0]])
    e_y = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(
        apply_rotary_pos_emb(e_x, cos, sin), torch.tensor([[c, s]]), atol=1e-6
    )
    assert torch.allclose(
        apply_rotary_pos_emb(e_y, cos, sin), torch.tensor([[-s, c]]), atol=1e-6
    )


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
def test_rotary_1d_tables_shapes_and_validation():
    head_dim, max_seq_len = 16, 32
    rope = RotaryEmbedding1DTables(head_dim=head_dim, max_seq_len=max_seq_len)
    assert rope.cos.shape == (max_seq_len, head_dim)
    assert "cos" not in rope.state_dict()  # persistent=False

    # Full table when no seq_len is given; sliced leading positions otherwise.
    cos_full, sin_full = rope()
    assert cos_full.shape == (max_seq_len, head_dim)
    cos, sin = rope(seq_len=20)
    assert cos.shape == (20, head_dim) and sin.shape == (20, head_dim)
    assert torch.equal(cos, cos_full[:20])

    with pytest.raises(ValueError):
        RotaryEmbedding1DTables(head_dim=15, max_seq_len=max_seq_len)
    # Exceeding max_seq_len is rejected.
    with pytest.raises(ValueError):
        rope(seq_len=max_seq_len + 1)


@torch.no_grad()
def test_rotary_1d_tables_match_sliced_builder():
    """Shorter requests use the leading positions of the precomputed tables."""
    head_dim, max_seq_len = 32, 64
    rope = RotaryEmbedding1DTables(head_dim=head_dim, max_seq_len=max_seq_len)

    seq_len = 40
    cos, sin = rope(seq_len=seq_len)
    exp_cos, exp_sin = build_rope_cos_sin_1d(max_seq_len, head_dim)
    assert torch.equal(cos, exp_cos[:seq_len])
    assert torch.equal(sin, exp_sin[:seq_len])


@torch.no_grad()
def test_rotary_1d_relative_phase_is_translation_invariant():
    """RoPE encodes position as a relative rotation: the q.k inner product
    between positions i and j depends only on (i - j)."""
    torch.manual_seed(0)
    head_dim, max_seq_len = 16, 64
    rope = RotaryEmbedding1DTables(head_dim=head_dim, max_seq_len=max_seq_len)
    cos, sin = rope()

    # Same content at every position; rotate, then compare inner products of
    # pairs sharing the same offset.
    base = torch.randn(1, 1, 1, head_dim)
    seq = base.expand(1, 1, max_seq_len, head_dim).contiguous()
    q_rot = apply_rotary_pos_emb(seq, cos, sin)
    k_rot = apply_rotary_pos_emb(seq, cos, sin)

    def dot(i, j):
        return (q_rot[0, 0, i] * k_rot[0, 0, j]).sum()

    # Offset of 3 gives the same score regardless of absolute position.
    assert torch.allclose(dot(5, 2), dot(20, 17), atol=1e-4)
    assert torch.allclose(dot(10, 4), dot(30, 24), atol=1e-4)


# --- physicsnemo.Module checkpoint round-trips ---
#
# Both providers subclass physicsnemo.core.Module, so they must support the
# .save() / Module.from_checkpoint() recipe. They cache cos/sin as
# persistent=False buffers (deterministically rebuilt from the __init__ args), so
# a round-trip reproduces the tables exactly without them appearing in the
# checkpoint.


@torch.no_grad()
def test_rotary_2d_tables_checkpoint_round_trip(tmp_path):
    head_dim, h, w = 16, 4, 5
    rope = RotaryEmbedding2DTables(head_dim=head_dim, latent_hw=(h, w), theta=5000.0)
    assert isinstance(rope, Module)
    # persistent=False: tables are not serialized.
    assert "rope_cos" not in rope.state_dict() and "rope_sin" not in rope.state_dict()

    cos_ref, sin_ref = rope()

    path = str(tmp_path / "rope2d.mdlus")
    rope.save(path)
    loaded = Module.from_checkpoint(path)
    # Tables were rebuilt at the right shape from the saved __init__ args.
    assert loaded.rope_cos.shape == (h, w, head_dim)
    assert loaded.theta == 5000.0
    cos_out, sin_out = loaded()
    assert torch.equal(cos_out, cos_ref) and torch.equal(sin_out, sin_ref)


@torch.no_grad()
def test_rotary_1d_tables_checkpoint_round_trip(tmp_path):
    head_dim, max_seq_len = 16, 32
    rope = RotaryEmbedding1DTables(
        head_dim=head_dim, max_seq_len=max_seq_len, theta=5000.0
    )
    assert isinstance(rope, Module)
    assert "cos" not in rope.state_dict() and "sin" not in rope.state_dict()

    cos_ref, sin_ref = rope()

    path = str(tmp_path / "rope1d.mdlus")
    rope.save(path)
    loaded = Module.from_checkpoint(path)
    assert loaded.cos.shape == (max_seq_len, head_dim)
    cos_out, sin_out = loaded()
    assert torch.equal(cos_out, cos_ref) and torch.equal(sin_out, sin_ref)
