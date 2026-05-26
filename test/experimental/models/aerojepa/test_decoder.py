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

"""Tests for :class:`QueryTokenDecoder` and its SIREN helpers."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.decoder import (
    QueryTokenDecoder,
    SineLayer,
    SirenHead,
)
from physicsnemo.experimental.nn.aerojepa import TokenSet


def _build(**overrides) -> QueryTokenDecoder:
    cfg = dict(
        token_dim=32,
        hidden_dim=64,
        num_layers=2,
        out_dim=4,
        use_sdf=True,
        cond_dim=4,
        pe_num_bands=4,
        cross_attention_heads=4,
        cross_attention_layers=1,
        cross_attention_k=4,
        attention_mlp_ratio=2,
        dropout=0.0,
        query_chunk_size=128,
    )
    cfg.update(overrides)
    return QueryTokenDecoder(**cfg)


# ---------------------------------------------------------------------------
# SIREN helpers
# ---------------------------------------------------------------------------


def test_sine_layer_shape(device):
    """``SineLayer`` forward returns ``(N, out_dim)``."""
    layer = SineLayer(8, 16, omega_0=30.0).to(device)
    assert layer(torch.randn(5, 8, device=device)).shape == (5, 16)


def test_siren_head_shape(device):
    """``SirenHead`` returns ``(N, out_dim)`` after its hidden + final linear."""
    head = SirenHead(in_dim=8, hidden_dim=32, out_dim=4, num_layers=2).to(device)
    assert head(torch.randn(5, 8, device=device)).shape == (5, 4)


# ---------------------------------------------------------------------------
# QueryTokenDecoder
# ---------------------------------------------------------------------------


def test_basic_forward(device):
    """Plain decoder returns ``(pred, emb)`` of expected shapes."""
    dec = _build().to(device).eval()
    target = TokenSet(
        features=torch.randn(16, 32, device=device),
        coords=torch.randn(16, 3, device=device),
    )
    qp = torch.randn(50, 3, device=device)
    qsdf = torch.randn(50, 1, device=device)
    cond = torch.randn(50, 4, device=device)
    pred, emb = dec.forward(
        query_pos=qp, query_sdf=qsdf, target_tokens=target, cond=cond
    )
    assert pred.shape == (50, 4)
    assert emb.shape == (50, 32)


def test_forward_batched(device):
    """Batched decoder returns padded ``(B, Nq, out_dim)``."""
    dec = _build().to(device).eval()
    B = 2
    target = TokenSet(
        features=torch.randn(B, 16, 32, device=device),
        coords=torch.randn(B, 16, 3, device=device),
        mask=torch.ones(B, 16, dtype=torch.bool, device=device),
    )
    qp = torch.randn(B, 30, 3, device=device)
    qsdf = torch.randn(B, 30, 1, device=device)
    counts = torch.tensor([25, 30], device=device, dtype=torch.long)
    cond = torch.randn(B, 4, device=device)
    pred, _ = dec.forward_batched(
        query_pos=qp,
        query_sdf=qsdf,
        query_counts=counts,
        target_tokens=target,
        cond=cond,
    )
    assert pred.shape == (B, 30, 4)


def test_pressure_split_head_mlp(device):
    """``pressure_split_head_enabled`` with ``'mlp'`` produces 4-channel output."""
    dec = _build(pressure_split_head_enabled=True, pressure_head_style="mlp")
    dec = dec.to(device).eval()
    target = TokenSet(
        features=torch.randn(8, 32, device=device),
        coords=torch.randn(8, 3, device=device),
    )
    pred, _ = dec.forward(
        query_pos=torch.randn(10, 3, device=device),
        query_sdf=torch.randn(10, 1, device=device),
        target_tokens=target,
        cond=torch.randn(10, 4, device=device),
    )
    assert pred.shape == (10, 4)


def test_pressure_split_head_siren_and_final_refinement(device):
    """SIREN pressure head + SIREN final refinement compose."""
    dec = (
        _build(
            pressure_split_head_enabled=True,
            pressure_head_style="siren",
            pressure_head_siren_layers=2,
            final_refinement_style="siren",
            final_refinement_siren_layers=2,
        )
        .to(device)
        .eval()
    )
    target = TokenSet(
        features=torch.randn(8, 32, device=device),
        coords=torch.randn(8, 3, device=device),
    )
    pred, _ = dec.forward(
        query_pos=torch.randn(10, 3, device=device),
        query_sdf=torch.randn(10, 1, device=device),
        target_tokens=target,
        cond=torch.randn(10, 4, device=device),
    )
    assert pred.shape == (10, 4)


def test_wall_velocity_gate(device):
    """Wall gate composes with the default head."""
    dec = _build(wall_velocity_gate_enabled=True).to(device).eval()
    target = TokenSet(
        features=torch.randn(8, 32, device=device),
        coords=torch.randn(8, 3, device=device),
    )
    pred, _ = dec.forward(
        query_pos=torch.randn(10, 3, device=device),
        query_sdf=torch.randn(10, 1, device=device),
        target_tokens=target,
        cond=torch.randn(10, 4, device=device),
    )
    assert pred.shape == (10, 4)


def test_extra_sdf_features(device):
    """``extra_sdf_features_enabled`` appends 3 channels to the per-query feature."""
    dec = _build(extra_sdf_features_enabled=True).to(device).eval()
    target = TokenSet(
        features=torch.randn(8, 32, device=device),
        coords=torch.randn(8, 3, device=device),
    )
    pred, _ = dec.forward(
        query_pos=torch.randn(10, 3, device=device),
        query_sdf=torch.randn(10, 1, device=device),
        target_tokens=target,
        cond=torch.randn(10, 4, device=device),
    )
    assert pred.shape == (10, 4)


def test_bad_pressure_head_style_raises():
    """Unrecognised ``pressure_head_style`` is rejected."""
    with pytest.raises(ValueError, match="pressure_head_style"):
        QueryTokenDecoder(token_dim=32, pressure_head_style="bogus")


def test_bad_final_refinement_style_raises():
    """Unrecognised ``final_refinement_style`` is rejected."""
    with pytest.raises(ValueError, match="final_refinement_style"):
        QueryTokenDecoder(token_dim=32, final_refinement_style="bogus")


def test_missing_sdf_raises(device):
    """``use_sdf=True`` without ``query_sdf`` is rejected."""
    dec = _build().to(device).eval()
    target = TokenSet(
        features=torch.randn(8, 32, device=device),
        coords=torch.randn(8, 3, device=device),
    )
    with pytest.raises(ValueError, match="query_sdf"):
        dec.forward(
            query_pos=torch.randn(5, 3, device=device),
            query_sdf=None,
            target_tokens=target,
            cond=torch.randn(5, 4, device=device),
        )


def test_empty_target_tokens_raises(device):
    """All-False target mask leaves zero valid tokens — rejected."""
    dec = _build().to(device).eval()
    target = TokenSet(
        features=torch.randn(8, 32, device=device),
        coords=torch.randn(8, 3, device=device),
        mask=torch.zeros(8, dtype=torch.bool, device=device),
    )
    with pytest.raises(ValueError, match="at least one valid token"):
        dec.forward(
            query_pos=torch.randn(5, 3, device=device),
            query_sdf=torch.randn(5, 1, device=device),
            target_tokens=target,
            cond=torch.randn(5, 4, device=device),
        )
