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

"""Tests for :class:`PrototypeTokenJEPAHead`."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.predictor import PrototypeTokenJEPAHead
from physicsnemo.experimental.nn.aerojepa import TokenSet


def _build(*, cond_dim: int = 4) -> PrototypeTokenJEPAHead:
    return PrototypeTokenJEPAHead(
        token_dim=32,
        cond_dim=cond_dim,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        neighbor_k=4,
        knn_chunk_size=32,
        query_pe_bands=4,
        mlp_ratio=2,
        dropout=0.0,
    )


def test_forward_unbatched(device):
    """Unbatched context + rank-2 ``target_positions`` + rank-1 ``cond`` works."""
    head = _build().to(device).eval()
    ctx = TokenSet(
        features=torch.randn(16, 32, device=device),
        coords=torch.randn(16, 3, device=device),
    )
    out = head.forward(
        context_tokens=ctx,
        target_positions=torch.randn(12, 3, device=device),
        cond=torch.randn(4, device=device),
    )
    assert out.shape == (12, 32)


def test_forward_batched(device):
    """Batched context (B=2) + rank-3 ``target_positions`` + rank-2 ``cond`` works."""
    head = _build().to(device).eval()
    ctx = TokenSet(
        features=torch.randn(2, 16, 32, device=device),
        coords=torch.randn(2, 16, 3, device=device),
        mask=torch.ones(2, 16, dtype=torch.bool, device=device),
    )
    out = head.forward(
        context_tokens=ctx,
        target_positions=torch.randn(2, 12, 3, device=device),
        cond=torch.randn(2, 4, device=device),
    )
    assert out.shape == (2, 12, 32)


def test_target_positions_broadcast(device):
    """Rank-2 ``target_positions`` is broadcast across the context batch."""
    head = _build().to(device).eval()
    ctx = TokenSet(
        features=torch.randn(2, 16, 32, device=device),
        coords=torch.randn(2, 16, 3, device=device),
        mask=torch.ones(2, 16, dtype=torch.bool, device=device),
    )
    out = head.forward(
        context_tokens=ctx,
        target_positions=torch.randn(12, 3, device=device),
        cond=torch.randn(2, 4, device=device),
    )
    assert out.shape == (2, 12, 32)


def test_cond_b1_broadcast(device):
    """``cond`` with leading dim 1 is expanded to match the context batch."""
    head = _build().to(device).eval()
    ctx = TokenSet(
        features=torch.randn(2, 16, 32, device=device),
        coords=torch.randn(2, 16, 3, device=device),
        mask=torch.ones(2, 16, dtype=torch.bool, device=device),
    )
    out = head.forward(
        context_tokens=ctx,
        target_positions=torch.randn(2, 12, 3, device=device),
        cond=torch.randn(1, 4, device=device),
    )
    assert out.shape == (2, 12, 32)


def test_no_conditioning(device):
    """``cond_dim=0`` works without a ``cond`` argument."""
    head = _build(cond_dim=0).to(device).eval()
    ctx = TokenSet(
        features=torch.randn(16, 32, device=device),
        coords=torch.randn(16, 3, device=device),
    )
    out = head.forward(
        context_tokens=ctx,
        target_positions=torch.randn(12, 3, device=device),
    )
    assert out.shape == (12, 32)


def test_missing_cond_raises(device):
    """``cond_dim>0`` without ``cond`` is rejected."""
    head = _build().to(device).eval()
    ctx = TokenSet(
        features=torch.randn(16, 32, device=device),
        coords=torch.randn(16, 3, device=device),
    )
    with pytest.raises(ValueError, match="cond must be provided"):
        head.forward(
            context_tokens=ctx,
            target_positions=torch.randn(12, 3, device=device),
            cond=None,
        )


def test_target_positions_batch_mismatch_raises(device):
    """``target_positions`` with a batch dim that doesn't match context is rejected."""
    head = _build().to(device).eval()
    ctx = TokenSet(
        features=torch.randn(16, 32, device=device),
        coords=torch.randn(16, 3, device=device),
    )
    with pytest.raises(ValueError, match="does not match context batch"):
        head.forward(
            context_tokens=ctx,
            target_positions=torch.randn(3, 12, 3, device=device),
            cond=torch.randn(4, device=device),
        )
