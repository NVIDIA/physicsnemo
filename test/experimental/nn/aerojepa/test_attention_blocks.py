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

"""Tests for AeroJEPA attention building blocks."""

import pytest
import torch

from physicsnemo.experimental.nn.aerojepa import (
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
    ResidualMLP,
)

# ---------------------------------------------------------------------------
# ResidualMLP
# ---------------------------------------------------------------------------


def test_residual_mlp_construction_and_shape(device):
    """Default ResidualMLP preserves shape and has learnable parameters."""
    mlp = ResidualMLP(dim=32, mlp_ratio=2, dropout=0.0).to(device)
    assert mlp.conditioning is None
    assert mlp.adaln_zero is False
    x = torch.randn(8, 32, device=device)
    out = mlp(x)
    assert out.shape == x.shape
    assert sum(p.numel() for p in mlp.parameters()) > 0


def test_residual_mlp_conditioning_requires_cond(device):
    """A conditioned ResidualMLP must be given a ``cond`` at forward time."""
    mlp = ResidualMLP(dim=16, mlp_ratio=2, dropout=0.0, conditioning_dim=8).to(device)
    with pytest.raises(ValueError, match="conditioning input must be provided"):
        mlp(torch.randn(4, 16, device=device))


def test_residual_mlp_adaln_zero_identity_at_init(device):
    """Zero-init conditioning + AdaLN-Zero yields output equal to input at init."""
    mlp = ResidualMLP(
        dim=16, mlp_ratio=2, dropout=0.0, conditioning_dim=8, adaln_zero=True
    ).to(device)
    x = torch.randn(4, 16, device=device)
    out = mlp(x, cond=torch.randn(8, device=device))
    assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# LocalPointTransformerBlock
# ---------------------------------------------------------------------------


def _make_lpt(dim=32, num_heads=4, **kwargs):
    """Construct a ``LocalPointTransformerBlock`` with sensible defaults."""
    return LocalPointTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        neighbor_k=kwargs.pop("neighbor_k", 8),
        dilation=kwargs.pop("dilation", 1),
        mlp_ratio=kwargs.pop("mlp_ratio", 2),
        dropout=kwargs.pop("dropout", 0.0),
        knn_chunk_size=kwargs.pop("knn_chunk_size", 64),
        **kwargs,
    )


def test_lpt_constructor_attributes():
    """Constructor stores the dim/num_heads/neighbor_k and derives head_dim."""
    blk = _make_lpt(dim=32, num_heads=4, neighbor_k=6, dilation=2)
    assert blk.dim == 32
    assert blk.num_heads == 4
    assert blk.head_dim == 8
    assert blk.neighbor_k == 6
    assert blk.dilation == 2


def test_lpt_dim_not_divisible_by_heads_raises():
    """``dim`` must be divisible by ``num_heads``."""
    with pytest.raises(ValueError, match="dim must be divisible by num_heads"):
        _make_lpt(dim=30, num_heads=4)


def test_lpt_dilation_clamped_to_one():
    """A non-positive dilation is clamped up to 1."""
    blk = _make_lpt(dilation=0)
    assert blk.dilation == 1


def test_lpt_forward_shape(device):
    """Self-attention preserves the (N, dim) shape."""
    blk = _make_lpt().to(device)
    feats = torch.randn(20, 32, device=device)
    coords = torch.randn(20, 3, device=device)
    out = blk(feats, coords)
    assert out.shape == (20, 32)


def test_lpt_single_point_fallback(device):
    """N <= 1 skips attention and applies only the FFN."""
    blk = _make_lpt().to(device)
    feats = torch.randn(1, 32, device=device)
    coords = torch.randn(1, 3, device=device)
    out = blk(feats, coords)
    assert out.shape == (1, 32)


def test_lpt_with_batch_ids(device):
    """Per-point ``batch_ids`` are accepted without changing the output shape."""
    blk = _make_lpt(neighbor_k=4).to(device)
    feats = torch.randn(12, 32, device=device)
    coords = torch.randn(12, 3, device=device)
    batch_ids = torch.tensor([0] * 6 + [1] * 6, device=device, dtype=torch.long)
    out = blk(feats, coords, batch_ids=batch_ids)
    assert out.shape == (12, 32)


def test_lpt_conditioning_requires_cond(device):
    """A conditioned block needs ``cond`` at forward time."""
    blk = _make_lpt(conditioning_dim=8).to(device)
    feats = torch.randn(10, 32, device=device)
    coords = torch.randn(10, 3, device=device)
    with pytest.raises(ValueError, match="conditioning input must be provided"):
        blk(feats, coords)


# ---------------------------------------------------------------------------
# LocalTokenCrossAttentionBlock
# ---------------------------------------------------------------------------


def _make_ltca(dim=32, num_heads=4, **kwargs):
    """Construct a ``LocalTokenCrossAttentionBlock`` with sensible defaults."""
    return LocalTokenCrossAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        neighbor_k=kwargs.pop("neighbor_k", 6),
        mlp_ratio=kwargs.pop("mlp_ratio", 2),
        dropout=kwargs.pop("dropout", 0.0),
        knn_chunk_size=kwargs.pop("knn_chunk_size", 64),
        **kwargs,
    )


def test_ltca_constructor_attributes():
    """Constructor stores key configuration and derives head_dim."""
    blk = _make_ltca(dim=32, num_heads=4, neighbor_k=6)
    assert blk.dim == 32
    assert blk.num_heads == 4
    assert blk.head_dim == 8
    assert blk.neighbor_k == 6


def test_ltca_dim_not_divisible_by_heads_raises():
    """``dim`` must be divisible by ``num_heads``."""
    with pytest.raises(ValueError, match="dim must be divisible by num_heads"):
        _make_ltca(dim=30, num_heads=4)


def test_ltca_forward_shape(device):
    """Cross-attention returns one feature row per query."""
    blk = _make_ltca().to(device)
    qf = torch.randn(15, 32, device=device)
    qc = torch.randn(15, 3, device=device)
    cf = torch.randn(25, 32, device=device)
    cc = torch.randn(25, 3, device=device)
    out = blk(qf, qc, cf, cc)
    assert out.shape == (15, 32)


def test_ltca_empty_query_fallback(device):
    """Empty query or context short-circuits to the (unchanged) query features."""
    blk = _make_ltca().to(device)
    qf_empty = torch.zeros(0, 32, device=device)
    qc_empty = torch.zeros(0, 3, device=device)
    cf = torch.randn(10, 32, device=device)
    cc = torch.randn(10, 3, device=device)
    out = blk(qf_empty, qc_empty, cf, cc)
    assert out.shape == (0, 32)


def test_ltca_conditioning_identity_at_init(device):
    """Zero-init 5-way conditioning yields identity output for the query side."""
    blk = _make_ltca(conditioning_dim=8, adaln_zero=True).to(device)
    qf = torch.randn(8, 32, device=device)
    qc = torch.randn(8, 3, device=device)
    cf = torch.randn(12, 32, device=device)
    cc = torch.randn(12, 3, device=device)
    out = blk(qf, qc, cf, cc, cond=torch.randn(8, device=device))
    assert torch.allclose(out, qf)


def test_ltca_with_batch_ids(device):
    """Cross-batch masking is allowed when both batch_ids are present."""
    blk = _make_ltca(neighbor_k=4).to(device)
    qf = torch.randn(10, 32, device=device)
    qc = torch.randn(10, 3, device=device)
    cf = torch.randn(20, 32, device=device)
    cc = torch.randn(20, 3, device=device)
    q_ids = torch.tensor([0] * 5 + [1] * 5, device=device, dtype=torch.long)
    c_ids = torch.tensor([0] * 10 + [1] * 10, device=device, dtype=torch.long)
    out = blk(qf, qc, cf, cc, query_batch_ids=q_ids, context_batch_ids=c_ids)
    assert out.shape == (10, 32)


def test_ltca_conditioning_requires_cond(device):
    """A conditioned cross-attention block needs ``cond``."""
    blk = _make_ltca(conditioning_dim=8).to(device)
    qf = torch.randn(8, 32, device=device)
    qc = torch.randn(8, 3, device=device)
    cf = torch.randn(12, 32, device=device)
    cc = torch.randn(12, 3, device=device)
    with pytest.raises(ValueError, match="conditioning input must be provided"):
        blk(qf, qc, cf, cc)
