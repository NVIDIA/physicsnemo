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

"""Tests for :class:`TargetTransformer`."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.encoders.base import BaseTargetEncoder
from physicsnemo.experimental.models.aerojepa.encoders.target import TargetTransformer


def _build(token_dim: int = 32, max_tokens: int = 12) -> TargetTransformer:
    return TargetTransformer(
        point_input_dim=4,
        token_dim=token_dim,
        max_point_tokens=max_tokens,
        tokenizer_strategy="fps",
        tokenizer_knn_chunk_size=32,
        point_pos_pe_bands=4,
        num_heads=4,
        num_layers=2,
        neighbor_k=4,
        mlp_ratio=2,
        dropout=0.0,
    )


def test_subclass_and_batched_flag():
    """TargetTransformer is a BaseTargetEncoder and supports batched forward."""
    tt = _build()
    assert isinstance(tt, BaseTargetEncoder)
    assert tt.supports_batched_forward is True


def test_constructor_drops_cross_and_conditioning_args():
    """``num_cross_layers`` / ``use_gen_conditioning`` are not constructor args."""
    co_vars = set(TargetTransformer.__init__.__code__.co_varnames)
    assert "num_cross_layers" not in co_vars
    assert "num_cross_heads" not in co_vars
    assert "cross_neighbor_k" not in co_vars
    assert "use_gen_conditioning" not in co_vars
    assert "gen_conditioning_dim" not in co_vars


def test_forward_signature_drops_gen_params_and_context_tokens():
    """``forward`` / ``forward_batched`` do not take ``gen_params`` or ``context_tokens``."""
    for fn in (TargetTransformer.forward, TargetTransformer.forward_batched):
        co_vars = set(fn.__code__.co_varnames)
        assert "gen_params" not in co_vars
        assert "context_tokens" not in co_vars


def test_forward_concatenates_surface_and_volume(device):
    """Single forward produces ``(max_point_tokens, token_dim)`` from concatenated input."""
    tt = _build(token_dim=32, max_tokens=12).to(device).eval()
    surf_pos = torch.randn(40, 3, device=device)
    surf_feat = torch.randn(40, 4, device=device)
    vol_pos = torch.randn(60, 3, device=device)
    vol_feat = torch.randn(60, 4, device=device)
    out = tt.forward(
        surface_pos=surf_pos,
        surface_main_feat=surf_feat,
        volume_pos=vol_pos,
        volume_feat=vol_feat,
    )
    assert out.tokens.features.shape == (12, 32)


def test_forward_batched_weaves_per_batch(device):
    """Batched forward weaves variable-length surface and volume halves."""
    tt = _build(token_dim=32, max_tokens=10).to(device).eval()
    surf_pos = torch.randn(2, 40, 3, device=device)
    surf_feat = torch.randn(2, 40, 4, device=device)
    vol_pos = torch.randn(2, 60, 3, device=device)
    vol_feat = torch.randn(2, 60, 4, device=device)
    surf_n = torch.tensor([35, 40], device=device, dtype=torch.long)
    vol_n = torch.tensor([50, 60], device=device, dtype=torch.long)
    out = tt.forward_batched(
        surface_pos=surf_pos,
        surface_main_feat=surf_feat,
        surface_pos_n=surf_n,
        volume_pos=vol_pos,
        volume_feat=vol_feat,
        volume_pos_n=vol_n,
    )
    assert out.tokens.features.shape == (2, 10, 32)


def test_forward_rejects_mismatched_feature_dims(device):
    """Mismatched surface vs volume feature dims is rejected."""
    tt = _build().to(device).eval()
    with pytest.raises(ValueError, match="matching feature dims"):
        tt.forward(
            surface_pos=torch.randn(10, 3, device=device),
            surface_main_feat=torch.randn(10, 4, device=device),
            volume_pos=torch.randn(15, 3, device=device),
            volume_feat=torch.randn(15, 5, device=device),
        )
