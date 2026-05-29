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

"""Tests for the AeroJEPA point-cloud tokenizer."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.layers import PointCloudTokenizer


def _make_pc(n=64, d=3, f=8, device="cpu"):
    """Make a synthetic point cloud of size N with D-dim coords and F-dim features."""
    return (
        torch.randn(n, d, device=device),
        torch.randn(n, f, device=device),
    )


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_identity_default_construction():
    """``strategy='identity'`` doesn't require any other arguments."""
    tok = PointCloudTokenizer()
    assert tok.strategy == "identity"
    assert tok.uses_cluster_pooling is False
    assert tok.requires_per_sample_loop is False


def test_unknown_strategy_raises():
    """Strategy strings outside the allowed set are rejected."""
    with pytest.raises(ValueError, match="strategy must be one of"):
        PointCloudTokenizer(strategy="bogus")


def test_max_point_tokens_nonpositive_raises():
    """``max_point_tokens <= 0`` is rejected."""
    with pytest.raises(ValueError, match="max_point_tokens must be > 0"):
        PointCloudTokenizer(max_point_tokens=0, strategy="random")


def test_cluster_size_nonpositive_raises():
    """``cluster_size <= 0`` is rejected."""
    with pytest.raises(ValueError, match="cluster_size must be > 0"):
        PointCloudTokenizer(max_point_tokens=8, strategy="fps_cluster", cluster_size=0)


def test_knn_chunk_size_nonpositive_raises():
    """``knn_chunk_size <= 0`` is rejected."""
    with pytest.raises(ValueError, match="knn_chunk_size must be > 0"):
        PointCloudTokenizer(strategy="fps", knn_chunk_size=0)


def test_data_prototype_cluster_requires_proto_coords():
    """``data_prototype_cluster`` without ``prototype_coords`` is rejected."""
    with pytest.raises(ValueError, match="prototype_coords must be provided"):
        PointCloudTokenizer(strategy="data_prototype_cluster")


def test_prototype_coords_wrong_shape_raises():
    """``prototype_coords`` must have shape ``(P, 3)``."""
    with pytest.raises(ValueError, match=r"prototype_coords must have shape"):
        PointCloudTokenizer(
            strategy="data_prototype_cluster",
            prototype_coords=torch.zeros(4, 2),
        )


def test_prototype_coords_buffer_non_persistent():
    """``prototype_coords`` registers as a non-persistent buffer."""
    tok = PointCloudTokenizer(
        strategy="data_prototype_cluster",
        prototype_coords=torch.zeros(4, 3),
    )
    # Buffer is exposed for use but excluded from state_dict.
    assert "prototype_coords" not in tok.state_dict()
    assert tok.prototype_coords.shape == (4, 3)


# ---------------------------------------------------------------------------
# Forward (single-cloud)
# ---------------------------------------------------------------------------


def test_forward_identity_passes_through(device):
    """``identity`` returns positions/features unchanged."""
    tok = PointCloudTokenizer(strategy="identity").to(device).eval()
    pos, feat = _make_pc(device=device)
    out_pos, out_feat = tok(point_positions=pos, point_features=feat)
    assert torch.equal(out_pos, pos)
    assert torch.equal(out_feat, feat)


@pytest.mark.parametrize(
    "strategy,extra_kw,max_tokens",
    [
        ("random", {}, 16),
        ("fps", {}, 16),
        ("random_cluster", {"cluster_size": 8}, 16),
        ("fps_cluster", {"cluster_size": 8}, 16),
        ("voxel_fps_cluster", {"cluster_size": 8, "voxel_size": 0.3}, 16),
    ],
)
def test_forward_shapes_per_strategy(device, strategy, extra_kw, max_tokens):
    """Each strategy reduces to ``max_point_tokens`` token rows."""
    tok = (
        PointCloudTokenizer(
            max_point_tokens=max_tokens,
            strategy=strategy,
            knn_chunk_size=32,
            **extra_kw,
        )
        .to(device)
        .eval()
    )
    pos, feat = _make_pc(device=device)
    out_pos, out_feat = tok(point_positions=pos, point_features=feat)
    assert out_pos.shape == (max_tokens, 3)
    assert out_feat.shape == (max_tokens, feat.shape[-1])


def test_data_prototype_cluster_forward_shapes(device):
    """Prototype-anchored clustering returns one token per prototype center."""
    proto = torch.randn(12, 3)
    tok = (
        PointCloudTokenizer(
            strategy="data_prototype_cluster",
            prototype_coords=proto,
            prototype_knn_k=6,
            knn_chunk_size=32,
        )
        .to(device)
        .eval()
    )
    assert tok.requires_per_sample_loop is True
    pos, feat = _make_pc(device=device)
    out_pos, out_feat, cluster_idx = tok.tokenize_with_clusters(
        point_positions=pos, point_features=feat
    )
    assert out_pos.shape == (12, 3)
    assert out_feat.shape == (12, feat.shape[-1])
    assert cluster_idx.shape == (12, 6)


def test_voxel_fps_cluster_missing_voxel_size_raises(device):
    """``voxel_fps_cluster`` requires ``voxel_size``."""
    tok = PointCloudTokenizer(
        max_point_tokens=8, strategy="voxel_fps_cluster", cluster_size=4
    ).to(device)
    with pytest.raises(ValueError, match="voxel_size must be provided"):
        tok(
            point_positions=torch.randn(64, 3, device=device),
            point_features=torch.randn(64, 4, device=device),
        )


def test_forward_rank_mismatch_raises(device):
    """``forward`` rejects rank-3 inputs (use ``forward_batched``)."""
    tok = PointCloudTokenizer(max_point_tokens=8, strategy="random").to(device)
    with pytest.raises(ValueError, match="expects rank-2"):
        tok(
            point_positions=torch.randn(2, 16, 3, device=device),
            point_features=torch.randn(2, 16, 4, device=device),
        )


# ---------------------------------------------------------------------------
# Forward (batched)
# ---------------------------------------------------------------------------


def test_forward_batched_shapes(device):
    """Padded batched forward returns padded outputs + mask + counts."""
    tok = (
        PointCloudTokenizer(
            max_point_tokens=12,
            strategy="fps_cluster",
            cluster_size=4,
            knn_chunk_size=32,
        )
        .to(device)
        .eval()
    )
    pos = torch.randn(2, 60, 3, device=device)
    feat = torch.randn(2, 60, 8, device=device)
    counts = torch.tensor([50, 60], device=device, dtype=torch.long)
    pp, pf, mask, tc = tok.forward_batched(
        point_positions=pos, point_features=feat, point_counts=counts
    )
    assert pp.shape == (2, 12, 3)
    assert pf.shape == (2, 12, 8)
    assert mask.shape == (2, 12)
    assert tc.shape == (2,)
    assert mask.sum().item() == int(tc.sum().item())


def test_forward_batched_rank_mismatch_raises(device):
    """``forward_batched`` rejects rank-2 inputs."""
    tok = PointCloudTokenizer(strategy="identity").to(device)
    with pytest.raises(ValueError, match="expects rank-3"):
        tok.forward_batched(
            point_positions=torch.randn(64, 3, device=device),
            point_features=torch.randn(64, 4, device=device),
        )


def test_forward_batched_wrong_counts_shape_raises(device):
    """Wrong shape on ``point_counts`` is rejected."""
    tok = PointCloudTokenizer(strategy="identity").to(device)
    with pytest.raises(ValueError, match=r"point_counts must have shape"):
        tok.forward_batched(
            point_positions=torch.randn(2, 32, 3, device=device),
            point_features=torch.randn(2, 32, 4, device=device),
            point_counts=torch.zeros(3, dtype=torch.long, device=device),
        )
