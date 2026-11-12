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

"""Euclidean reference neighbourhoods on the library kNN: ranking, ties, padding."""

import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")  # local_neighborhoods re-exports the Data class
pytest.importorskip("scipy")  # the pinned KD-tree backend of the library kNN

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from datapipes.local_neighborhoods import euclidean_knn, with_self_slot  # noqa: E402

from physicsnemo.experimental.nn.geodesic_knn import geodesic_knn  # noqa: E402


def grid(nx, ny, spacing=(1.0, 1.0), z=0.0, offset=0):
    """Quad grid: positions and bidirectional perimeter edges (no diagonals)."""
    xs, ys = np.meshgrid(np.arange(nx) * spacing[0], np.arange(ny) * spacing[1])
    pos = np.stack((xs.ravel(), ys.ravel(), np.full(nx * ny, z)), axis=1)
    node = lambda i, j: offset + j * nx + i  # noqa: E731
    edges = []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                edges.append((node(i, j), node(i + 1, j)))
            if j + 1 < ny:
                edges.append((node(i, j), node(i, j + 1)))
    edges = np.asarray(edges, dtype=np.int64)
    edge_index = np.concatenate((edges, edges[:, ::-1]), axis=0).T
    return torch.tensor(pos, dtype=torch.float32), torch.from_numpy(edge_index.copy())


def test_euclidean_matches_a_brute_force_ranking_without_ties():
    """Random points: the library kNN result equals a float64 all-pairs ranking."""
    generator = torch.Generator().manual_seed(0)
    pos = torch.rand(300, 3, generator=generator, dtype=torch.float64).float()
    k = 7
    xyz = pos.double().numpy()
    distances = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    ranked = np.sort(distances, axis=1)
    assert (ranked[:, k] - ranked[:, k - 1] > 1e-9).all(), "the test needs no ties"
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    result = euclidean_knn(pos, k)
    assert result.mask.all()
    assert result.index.dtype == torch.int64 and result.distance.dtype == torch.float32
    assert result.index.numpy().tolist() == order.tolist()
    np.testing.assert_allclose(
        result.distance.numpy(),
        np.take_along_axis(distances, order, axis=1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_euclidean_crosses_the_gap_the_geodesic_query_respects():
    """Two sheets 1 mm apart: Euclidean neighbors cross, geodesic ones do not."""
    lower_pos, lower_edges = grid(4, 4, spacing=(10.0, 10.0), z=0.0)
    upper_pos, upper_edges = grid(4, 4, spacing=(10.0, 10.0), z=1.0, offset=16)
    pos = torch.cat((lower_pos, upper_pos))
    edge_index = torch.cat((lower_edges, upper_edges), dim=1)
    k = 5
    euclidean = euclidean_knn(pos, k)
    geodesic = geodesic_knn(pos, edge_index, k)
    assert euclidean.mask.all() and geodesic.mask.all()
    # Every lower node's nearest Euclidean neighbor is the node 1 mm above it.
    torch.testing.assert_close(euclidean.index[:16, 0], torch.arange(16, 32))
    torch.testing.assert_close(euclidean.distance[:16, 0], torch.ones(16))
    assert bool((geodesic.index[:16] < 16).all())


def test_euclidean_ties_at_the_k_boundary_follow_the_kd_tree():
    """Boundary ties are the KD-tree's choice, returned candidates at equal
    distance are ordered by node ID, and runs are deterministic."""
    pos, _ = grid(5, 5)
    first, second = euclidean_knn(pos, 1), euclidean_knn(pos, 1)
    torch.testing.assert_close(first.index, second.index)
    # Corner node 0 has nodes 1 and 5 at distance 1: one of them is returned.
    assert first.index[0].item() in (1, 5)
    torch.testing.assert_close(first.distance[0], torch.tensor([1.0]))
    # Centre node 12 has four nodes at distance 1, listed by node ID.
    assert euclidean_knn(pos, 4).index[12].tolist() == [7, 11, 13, 17]


def test_coincident_nodes_are_ordinary_neighbors():
    """Duplicate positions are neighbors at zero distance; only the node itself goes."""
    pos = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    result = euclidean_knn(pos, 2)
    assert result.index[0].tolist() == [1, 2]
    torch.testing.assert_close(result.distance[0], torch.tensor([0.0, 1.0]))
    assert result.index[1].tolist() == [0, 2]


PADDED_POSITIONS = torch.tensor(
    [[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [50, 0, 0], [51, 0, 0]]
)


def test_small_point_sets_are_padded():
    """Fewer than ``k`` other nodes pad with the query's own ID and ``mask=False``."""
    few = euclidean_knn(PADDED_POSITIONS[:3], 5)
    assert few.mask.sum(1).tolist() == [2, 2, 2]
    assert few.index[0].tolist() == [1, 2, 0, 0, 0]
    torch.testing.assert_close(few.distance[0], torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))
    alone = euclidean_knn(PADDED_POSITIONS[:1], 3)
    assert alone.index.tolist() == [[0, 0, 0]] and not alone.mask.any()
    empty = euclidean_knn(PADDED_POSITIONS[:0], 3)
    assert empty.index.shape == (0, 3) and empty.mask.shape == (0, 3)
    index, mask = with_self_slot(few)
    assert index[:, 0].tolist() == [0, 1, 2] and bool(mask[:, 0].all())
    assert index.shape == (3, 6)


def test_validation():
    """Bad positions and ``k`` are rejected before the query runs."""
    pos, _ = grid(3, 2)
    with pytest.raises(ValueError, match="positive integer"):
        euclidean_knn(pos, 0)
    with pytest.raises(ValueError, match="CPU float"):
        euclidean_knn(pos.half(), 2)
    with pytest.raises(ValueError, match="CPU float"):
        euclidean_knn(pos[:, :2], 2)
    with pytest.raises(ValueError, match="CPU float"):
        euclidean_knn(torch.tensor([[0.0, 0, float("nan")]]), 1)
