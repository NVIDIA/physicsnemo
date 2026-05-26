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

"""Tests for :class:`ContextTransformer`."""

import torch

from physicsnemo.experimental.models.aerojepa.encoders.base import BaseContextEncoder
from physicsnemo.experimental.models.aerojepa.encoders.context import ContextTransformer


def _build(token_dim: int = 32, max_tokens: int = 12) -> ContextTransformer:
    return ContextTransformer(
        point_input_dim=3,
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
    """ContextTransformer is a BaseContextEncoder and supports batched forward."""
    ct = _build()
    assert isinstance(ct, BaseContextEncoder)
    assert ct.supports_batched_forward is True


def test_forward_signature_drops_gen_params():
    """``forward`` and ``forward_batched`` do not take ``gen_params``."""
    assert "gen_params" not in ContextTransformer.forward.__code__.co_varnames
    assert "gen_params" not in ContextTransformer.forward_batched.__code__.co_varnames


def test_constructor_drops_conditioning_kwargs():
    """``use_gen_conditioning`` / ``gen_conditioning_dim`` are not constructor args."""
    co_vars = set(ContextTransformer.__init__.__code__.co_varnames)
    assert "use_gen_conditioning" not in co_vars
    assert "gen_conditioning_dim" not in co_vars


def test_forward_shape(device):
    """Single-cloud forward returns ``(max_point_tokens, token_dim)``."""
    ct = _build(token_dim=32, max_tokens=12).to(device).eval()
    pos = torch.randn(60, 3, device=device)
    feat = torch.zeros(60, 0, device=device)
    out = ct.forward(context_pos=pos, context_feat=feat)
    assert out.tokens.features.shape == (12, 32)


def test_forward_batched_shape(device):
    """Batched forward returns ``(B, max_point_tokens, token_dim)``."""
    ct = _build(token_dim=32, max_tokens=10).to(device).eval()
    pos = torch.randn(2, 50, 3, device=device)
    feat = torch.zeros(2, 50, 0, device=device)
    counts = torch.tensor([45, 50], device=device, dtype=torch.long)
    out = ct.forward_batched(context_pos=pos, context_feat=feat, context_pos_n=counts)
    assert out.tokens.features.shape == (2, 10, 32)
    assert out.tokens.mask.shape == (2, 10)


def test_has_learnable_parameters():
    """The encoder is a real ``nn.Module`` with parameters."""
    ct = _build()
    assert sum(p.numel() for p in ct.parameters()) > 0
