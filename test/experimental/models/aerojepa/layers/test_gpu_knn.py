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

"""Tests for the package-private GPU k-NN helpers.

These functions are not re-exported at the ``aerojepa`` package level (the
filename is prefixed with an underscore), so the tests reach in via the
submodule path directly.
"""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.layers._gpu_knn import (
    _gpu_knn_chunked,
    gpu_knn_bipartite,
    gpu_knn_interpolate,
    gpu_knn_self,
)


def test_chunked_basic_shapes(device):
    """``_gpu_knn_chunked`` returns ``(Nq, k)`` distances and ``int64`` indices."""
    query = torch.randn(20, 3, device=device)
    source = torch.randn(30, 3, device=device)
    dist, idx = _gpu_knn_chunked(query, source, k=5, chunk_size=8)
    assert dist.shape == (20, 5)
    assert idx.shape == (20, 5)
    assert idx.dtype == torch.long


def test_chunked_clamps_k_to_n_source(device):
    """``k`` is clamped down to ``n_source``."""
    query = torch.randn(5, 3, device=device)
    source = torch.randn(3, 3, device=device)
    dist, idx = _gpu_knn_chunked(query, source, k=10, chunk_size=8)
    assert dist.shape == (5, 3)
    assert idx.shape == (5, 3)


def test_self_no_self_edges(device):
    """Without ``include_self_edges`` no row contains its own index."""
    positions = torch.randn(15, 3, device=device)
    s, r, d = gpu_knn_self(positions, k=4)
    assert s.shape == r.shape == d.shape
    # Number of edges should be ``n * k`` exactly.
    assert s.shape[0] == 15 * 4
    # No edge connects a node to itself.
    assert not (s == r).any().item()


def test_self_with_self_edges(device):
    """``include_self_edges=True`` does not mask out self-loops."""
    positions = torch.randn(8, 3, device=device)
    s, r, d = gpu_knn_self(positions, k=3, include_self_edges=True)
    assert s.shape[0] == 8 * 3


def test_self_empty(device):
    """Empty input returns empty edge tensors with matching device/dtype."""
    positions = torch.empty(0, 3, device=device)
    s, r, d = gpu_knn_self(positions, k=5)
    assert s.numel() == 0
    assert d.dtype == positions.dtype


def test_self_single_point_without_self_returns_empty(device):
    """One point + no self-edges → no edges."""
    s, r, d = gpu_knn_self(torch.zeros(1, 3, device=device), k=4)
    assert s.numel() == 0


def test_bipartite_basic(device):
    """Each receiver gets ``k`` senders."""
    senders = torch.randn(15, 3, device=device)
    receivers = torch.randn(10, 3, device=device)
    s, r, d = gpu_knn_bipartite(senders, receivers, k=4)
    assert s.shape[0] == 10 * 4
    assert r.shape == s.shape
    assert int(r.max().item()) < 10
    assert int(s.max().item()) < 15


def test_bipartite_empty_receivers(device):
    """Empty receivers produce empty edge tensors."""
    senders = torch.randn(5, 3, device=device)
    s, r, d = gpu_knn_bipartite(senders, torch.empty(0, 3, device=device), k=3)
    assert s.numel() == 0


def test_bipartite_empty_senders_raises(device):
    """Empty senders are rejected."""
    with pytest.raises(ValueError, match="at least one point"):
        gpu_knn_bipartite(
            torch.empty(0, 3, device=device),
            torch.randn(4, 3, device=device),
            k=2,
        )


def test_interpolate_shape(device):
    """IDW interpolation returns ``(Nq, F)``."""
    src_pos = torch.randn(20, 3, device=device)
    src_feat = torch.randn(20, 8, device=device)
    query_pos = torch.randn(7, 3, device=device)
    out = gpu_knn_interpolate(src_pos, src_feat, query_pos, k=4)
    assert out.shape == (7, 8)


def test_interpolate_empty_query(device):
    """Empty queries return an empty result with the source's feature dim."""
    src_pos = torch.randn(5, 3, device=device)
    src_feat = torch.randn(5, 8, device=device)
    out = gpu_knn_interpolate(src_pos, src_feat, torch.empty(0, 3, device=device), k=2)
    assert out.shape == (0, 8)


def test_interpolate_empty_source_raises(device):
    """Empty source positions are rejected."""
    with pytest.raises(ValueError, match="at least one point"):
        gpu_knn_interpolate(
            torch.empty(0, 3, device=device),
            torch.empty(0, 4, device=device),
            torch.randn(3, 3, device=device),
            k=2,
        )
