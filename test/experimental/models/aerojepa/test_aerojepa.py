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

"""Tests for the top-level :class:`AeroJEPA` model."""

import inspect

import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.aerojepa import (
    AeroJEPA,
    AeroJEPAMetaData,
    AeroJEPATrunk,
    ContextTransformer,
    PrototypeTokenJEPAHead,
    QueryTokenDecoder,
    TargetTransformer,
)


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


def _build_model() -> AeroJEPA:
    trunk = AeroJEPATrunk(
        context_encoder=ContextTransformer(**_enc_kwargs()),
        target_encoder=TargetTransformer(**_enc_kwargs()),
        decoder=QueryTokenDecoder(
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
        ),
        include_geometry_global_in_decoder_cond=False,
    )
    predictor = PrototypeTokenJEPAHead(
        token_dim=32,
        cond_dim=4,
        depth=2,
        num_heads=4,
        neighbor_k=4,
        knn_chunk_size=32,
        query_pe_bands=4,
        mlp_ratio=2,
        dropout=0.0,
    )
    return AeroJEPA(trunk=trunk, predictor=predictor)


def test_inherits_physicsnemo_module():
    """``AeroJEPA`` inherits ``physicsnemo.core.module.Module``."""
    model = _build_model()
    assert isinstance(model, Module)


def test_meta_data():
    """``AeroJEPAMetaData`` has the expected default flags."""
    meta = AeroJEPAMetaData()
    assert meta.jit is False
    assert meta.cuda_graphs is False
    assert meta.amp is True
    assert meta.onnx_cpu is False
    assert meta.onnx_gpu is False
    assert meta.onnx_runtime is False


def test_forward_returns_plain_tensor(device):
    """``forward`` returns a plain ``torch.Tensor`` (not a ``TokenSet``)."""
    model = _build_model().to(device).eval()
    field = model.forward(
        context_pos=torch.randn(40, 3, device=device),
        context_feat=torch.zeros(40, 0, device=device),
        gen_params=torch.randn(4, device=device),
        query_pos=torch.randn(30, 3, device=device),
        query_sdf=torch.randn(30, 1, device=device),
    )
    assert isinstance(field, torch.Tensor)
    assert field.shape == (30, 4)


def test_forward_signature_drops_target_coords():
    """The forward API no longer accepts ``target_coords``."""
    sig = inspect.signature(AeroJEPA.forward)
    assert "target_coords" not in sig.parameters
    assert "context_pos" in sig.parameters
    assert "context_feat" in sig.parameters


def test_predict_is_no_grad(device):
    """``predict`` runs in no-grad mode regardless of caller context."""
    model = _build_model().to(device).eval()
    with torch.enable_grad():
        field = model.predict(
            context_pos=torch.randn(40, 3, device=device),
            context_feat=torch.zeros(40, 0, device=device),
            gen_params=torch.randn(4, device=device),
            query_pos=torch.randn(30, 3, device=device),
            query_sdf=torch.randn(30, 1, device=device),
        )
    assert field.requires_grad is False


def test_build_target_token_coords_single_arg(device):
    """``build_target_token_coords`` takes a single ``point_positions`` arg."""
    model = _build_model().to(device).eval()
    sig = inspect.signature(AeroJEPA.build_target_token_coords)
    assert set(sig.parameters) == {"self", "point_positions"}
    coords = model.build_target_token_coords(
        point_positions=torch.randn(40, 3, device=device)
    )
    assert coords.ndim == 2
    assert coords.shape[-1] == 3


def test_decode_field_chunked_fp32_returns_cpu(device):
    """``decode_field_chunked`` with fp32 returns a CPU tensor of the right shape."""
    model = _build_model().to(device).eval()
    ctx_pos = torch.randn(40, 3, device=device)
    ctx_feat = torch.zeros(40, 0, device=device)
    gen = torch.randn(4, device=device)
    ctx_tokens, cg = model.encode_geometry(
        context_pos=ctx_pos, context_feat=ctx_feat, gen_params=gen
    )
    tc = model.build_target_token_coords(point_positions=ctx_pos)
    pf = model.predict_field_tokens(
        context_tokens=ctx_tokens,
        target_positions=tc,
        conditions=gen.unsqueeze(0),
    )
    if pf.ndim == 3 and pf.shape[0] == 1:
        pf = pf[0]
    from physicsnemo.experimental.models.aerojepa.layers import TokenSet

    tt = TokenSet(
        features=pf,
        coords=tc,
        mask=torch.ones(tc.shape[0], dtype=torch.bool, device=device),
    )
    out = model.decode_field_chunked(
        target_tokens=tt,
        cond_global=cg,
        query_pos=torch.randn(200, 3),
        query_sdf=torch.randn(200, 1),
        chunk_size=64,
        precision="fp32",
    )
    assert out.device.type == "cpu"
    assert out.shape == (200, 4)


def test_encode_geometry_and_flow_returns_dict(device):
    """``encode_geometry_and_flow`` returns the dict the decoder consumes."""
    model = _build_model().to(device).eval()
    ctx = model.encode_geometry_and_flow(
        context_pos=torch.randn(40, 3, device=device),
        context_feat=torch.zeros(40, 0, device=device),
        target_surface_pos=torch.randn(50, 3, device=device),
        target_surface_main_feat=torch.randn(50, 3, device=device),
        target_volume_pos=torch.randn(60, 3, device=device),
        target_volume_feat=torch.randn(60, 3, device=device),
        gen_params=torch.randn(4, device=device),
    )
    assert set(ctx.keys()) == {"context_tokens", "target_tokens", "cond_global"}


def test_accessor_properties(device):
    """``context_encoder`` / ``target_encoder`` / ``decoder`` delegate to the trunk."""
    model = _build_model().to(device)
    assert model.context_encoder is model.trunk.context_encoder
    assert model.target_encoder is model.trunk.target_encoder
    assert model.decoder is model.trunk.decoder
    assert model.mask_head is None
    assert model.include_geometry_global_in_decoder_cond is False
