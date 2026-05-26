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

"""Tests for :class:`PointTransformer`, :class:`PointClusterGraphPool`, and the geometry-feature helper."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.encoders.point import (
    PointClusterGraphPool,
    PointTransformer,
    build_geometry_features,
)

# ---------------------------------------------------------------------------
# build_geometry_features
# ---------------------------------------------------------------------------


def test_build_geometry_features_identity(device):
    """No flags set: returns ``context_pos`` unchanged."""
    pos = torch.randn(8, 3, device=device)
    out = build_geometry_features(context_pos=pos, context_feat=torch.zeros(8, 5))
    assert out is pos


def test_build_geometry_features_with_sdf(device):
    """``use_sdf=True`` appends a single channel from the trailing slot."""
    pos = torch.randn(8, 3, device=device)
    feat = torch.randn(8, 4, device=device)
    out = build_geometry_features(context_pos=pos, context_feat=feat, use_sdf=True)
    assert out.shape == (8, 4)


def test_build_geometry_features_rejects_short_feat():
    """Too few trailing channels is rejected."""
    pos = torch.zeros(4, 3)
    feat = torch.zeros(4, 2)  # not enough for sdf+normals
    with pytest.raises(ValueError, match=r"not contain enough channels"):
        build_geometry_features(
            context_pos=pos, context_feat=feat, use_sdf=True, use_solid_normals=True
        )


# ---------------------------------------------------------------------------
# PointClusterGraphPool
# ---------------------------------------------------------------------------


def test_graph_pool_basic_shape(device):
    """Forward returns ``(N_tok, F)`` shape."""
    pool = PointClusterGraphPool(
        point_feature_dim=8, hidden_dim=16, num_layers=2, dropout=0.0
    ).to(device)
    src_pos = torch.randn(20, 3, device=device)
    src_feat = torch.randn(20, 8, device=device)
    tok_pos = torch.randn(5, 3, device=device)
    tok_feat = torch.randn(5, 8, device=device)
    cluster_idx = torch.randint(0, 20, (5, 4), device=device)
    out = pool(
        point_positions=src_pos,
        point_features=src_feat,
        token_positions=tok_pos,
        token_features=tok_feat,
        cluster_idx=cluster_idx,
    )
    assert out.shape == (5, 8)


def test_graph_pool_empty_tokens_is_noop(device):
    """Empty token positions short-circuits to the input token features."""
    pool = PointClusterGraphPool(
        point_feature_dim=8, hidden_dim=16, num_layers=1, dropout=0.0
    ).to(device)
    src_pos = torch.randn(20, 3, device=device)
    src_feat = torch.randn(20, 8, device=device)
    tok_pos = torch.zeros(0, 3, device=device)
    tok_feat = torch.zeros(0, 8, device=device)
    out = pool(
        point_positions=src_pos,
        point_features=src_feat,
        token_positions=tok_pos,
        token_features=tok_feat,
        cluster_idx=torch.zeros(0, 4, dtype=torch.long, device=device),
    )
    assert out.shape == (0, 8)


# ---------------------------------------------------------------------------
# PointTransformer
# ---------------------------------------------------------------------------


def _build_pt(*, use_conditioning: bool = False, cluster_pooling: str = "mean"):
    """Helper to build a small PointTransformer for tests."""
    return PointTransformer(
        point_input_dim=3,
        token_dim=32,
        max_point_tokens=12,
        tokenizer_strategy=("fps_cluster" if cluster_pooling == "graph" else "fps"),
        tokenizer_deterministic_eval=True,
        tokenizer_cluster_size=4 if cluster_pooling == "graph" else None,
        tokenizer_voxel_size=None,
        tokenizer_prototype_coords=None,
        tokenizer_prototype_knn_k=None,
        tokenizer_knn_chunk_size=32,
        point_pos_pe_bands=4,
        num_heads=4,
        num_layers=2,
        neighbor_k=4,
        dilation_schedule=None,
        mlp_ratio=2,
        dropout=0.0,
        tokenizer_cluster_pooling=cluster_pooling,
        tokenizer_graph_pool_hidden_dim=16,
        tokenizer_graph_pool_layers=1,
        use_gen_conditioning=use_conditioning,
        gen_conditioning_dim=8 if use_conditioning else None,
    )


def test_point_transformer_encode_single(device):
    """Unbatched ``encode_single`` produces ``(max_point_tokens, token_dim)``."""
    pt = _build_pt().to(device).eval()
    pos = torch.randn(60, 3, device=device)
    out = pt.encode_single(point_positions=pos, point_features=pos, gen_params=None)
    assert out.tokens.features.shape == (12, 32)


def test_point_transformer_forward_batched(device):
    """Batched ``forward_batched`` produces ``(B, max_point_tokens, token_dim)``."""
    pt = _build_pt().to(device).eval()
    pos = torch.randn(2, 60, 3, device=device)
    counts = torch.tensor([55, 60], device=device, dtype=torch.long)
    out = pt.forward_batched(
        point_positions=pos, point_features=pos, point_counts=counts, gen_params=None
    )
    assert out.tokens.features.shape == (2, 12, 32)
    assert out.tokens.mask.shape == (2, 12)


def test_point_transformer_conditioning_requires_gen_params(device):
    """``use_gen_conditioning=True`` without ``gen_params`` is rejected."""
    pt = _build_pt(use_conditioning=True).to(device).eval()
    pos = torch.randn(40, 3, device=device)
    with pytest.raises(ValueError, match=r"gen_params must be provided"):
        pt.encode_single(point_positions=pos, point_features=pos, gen_params=None)


def test_point_transformer_graph_pool_path(device):
    """``tokenizer_cluster_pooling='graph'`` runs end-to-end."""
    pt = _build_pt(cluster_pooling="graph").to(device).eval()
    assert pt.graph_pool is not None
    pos = torch.randn(60, 3, device=device)
    out = pt.encode_single(point_positions=pos, point_features=pos, gen_params=None)
    assert out.tokens.features.shape == (12, 32)


def test_point_transformer_graph_requires_cluster_strategy():
    """``'graph'`` pool with a non-cluster strategy is rejected."""
    with pytest.raises(ValueError, match=r"cluster tokenizer strategy"):
        PointTransformer(
            point_input_dim=3,
            token_dim=32,
            max_point_tokens=12,
            tokenizer_strategy="fps",  # non-cluster
            tokenizer_deterministic_eval=True,
            tokenizer_cluster_size=None,
            tokenizer_voxel_size=None,
            tokenizer_prototype_coords=None,
            tokenizer_prototype_knn_k=None,
            tokenizer_knn_chunk_size=32,
            point_pos_pe_bands=4,
            num_heads=4,
            num_layers=2,
            neighbor_k=4,
            dilation_schedule=None,
            mlp_ratio=2,
            dropout=0.0,
            tokenizer_cluster_pooling="graph",
            tokenizer_graph_pool_hidden_dim=16,
            tokenizer_graph_pool_layers=1,
            use_gen_conditioning=False,
            gen_conditioning_dim=None,
        )
