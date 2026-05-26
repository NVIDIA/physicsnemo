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

"""Tests for :class:`AeroJEPATrunk`."""

import torch

from physicsnemo.experimental.models.aerojepa.decoder import QueryTokenDecoder
from physicsnemo.experimental.models.aerojepa.encoders.context import (
    ContextTransformer,
)
from physicsnemo.experimental.models.aerojepa.encoders.target import (
    TargetTransformer,
)
from physicsnemo.experimental.models.aerojepa.trunk import AeroJEPATrunk


def _enc_kwargs() -> dict:
    return dict(
        point_input_dim=3,
        token_dim=32,
        max_point_tokens=12,
        tokenizer_strategy="fps",
        tokenizer_knn_chunk_size=32,
        point_pos_pe_bands=4,
        num_heads=4,
        num_layers=2,
        neighbor_k=4,
        mlp_ratio=2,
        dropout=0.0,
    )


def _build_decoder() -> QueryTokenDecoder:
    return QueryTokenDecoder(
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
        query_chunk_size=128,
    )


def _build_trunk(*, mask: bool = False) -> AeroJEPATrunk:
    return AeroJEPATrunk(
        context_encoder=ContextTransformer(**_enc_kwargs()),
        target_encoder=TargetTransformer(**_enc_kwargs()),
        decoder=_build_decoder(),
        include_geometry_global_in_decoder_cond=False,
        mask_prediction_enabled=mask,
        mask_head_hidden_dim=32,
    )


def test_supports_batched_forward():
    """Trunk advertises batched-forward support."""
    assert AeroJEPATrunk.supports_batched_forward() is True


def test_default_mask_head_disabled():
    """``mask_head`` is ``None`` when ``mask_prediction_enabled=False``."""
    trunk = _build_trunk()
    assert trunk.mask_head is None


def test_forward_single_shape(device):
    """End-to-end single-sample forward returns ``(Nq, out_dim)``."""
    trunk = _build_trunk().to(device).eval()
    out = trunk.forward_single(
        context_pos=torch.randn(40, 3, device=device),
        context_feat=torch.zeros(40, 0, device=device),
        surface_main_feat=torch.randn(40, 3, device=device),
        volume_pos=torch.randn(60, 3, device=device),
        volume_feat=torch.randn(60, 3, device=device),
        query_pos=torch.randn(30, 3, device=device),
        query_sdf=torch.randn(30, 1, device=device),
        gen_params=torch.randn(4, device=device),
    )
    assert out.shape == (30, 4)


def test_forward_batch_shape(device):
    """End-to-end batched forward returns ``(B, Nq, out_dim)``."""
    trunk = _build_trunk().to(device).eval()
    B = 2
    out = trunk.forward_batch(
        context_pos=torch.randn(B, 40, 3, device=device),
        context_feat=torch.zeros(B, 40, 0, device=device),
        surface_main_feat=torch.randn(B, 40, 3, device=device),
        context_pos_n=torch.tensor([40, 40], device=device, dtype=torch.long),
        volume_pos=torch.randn(B, 60, 3, device=device),
        volume_feat=torch.randn(B, 60, 3, device=device),
        volume_pos_n=torch.tensor([55, 60], device=device, dtype=torch.long),
        query_pos=torch.randn(B, 30, 3, device=device),
        query_sdf=torch.randn(B, 30, 1, device=device),
        query_pos_n=torch.tensor([25, 30], device=device, dtype=torch.long),
        gen_params=torch.randn(B, 4, device=device),
    )
    assert out.shape == (B, 30, 4)


def test_forward_with_mask_returns_tuple(device):
    """``return_mask_logits=True`` returns ``(pred, mask_logits)`` when mask head enabled."""
    trunk = _build_trunk(mask=True).to(device).eval()
    out = trunk.forward_single(
        context_pos=torch.randn(40, 3, device=device),
        context_feat=torch.zeros(40, 0, device=device),
        surface_main_feat=torch.randn(40, 3, device=device),
        volume_pos=torch.randn(60, 3, device=device),
        volume_feat=torch.randn(60, 3, device=device),
        query_pos=torch.randn(30, 3, device=device),
        query_sdf=torch.randn(30, 1, device=device),
        gen_params=torch.randn(4, device=device),
        return_mask_logits=True,
    )
    pred, mask_logits = out
    assert pred.shape == (30, 4)
    assert mask_logits.shape == (30, 1)


def test_encode_context_returns_expected_keys(device):
    """``encode_context`` returns a dict with the keys the decoder consumes."""
    trunk = _build_trunk().to(device).eval()
    ctx = trunk.encode_context(
        context_pos=torch.randn(40, 3, device=device),
        context_feat=torch.zeros(40, 0, device=device),
        surface_main_feat=torch.randn(40, 3, device=device),
        volume_pos=torch.randn(60, 3, device=device),
        volume_feat=torch.randn(60, 3, device=device),
        gen_params=torch.randn(4, device=device),
    )
    assert set(ctx.keys()) == {"context_tokens", "target_tokens", "cond_global"}
