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
"""Static reference kNN: exactness against dense Dijkstra, topology, padding.

The geodesic path is pure Python and runs everywhere; the Euclidean path and
the dense Dijkstra reference need SciPy, so only those tests skip without it.
"""

import numpy as np
import pytest
import torch

from physicsnemo.experimental.nn.static_knn import (
    euclidean_knn,
    geodesic_knn,
    static_knn,
)


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


requires_scipy = pytest.mark.skipif(
    not _has_scipy(),
    reason="SciPy is needed for Euclidean neighbourhoods and the Dijkstra reference",
)


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


def dense_geodesic_reference(pos, edge_index, k):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(pos)
    src, dst = edge_index.numpy()
    lengths = np.linalg.norm(pos.numpy()[src] - pos.numpy()[dst], axis=1)
    graph = coo_matrix((lengths, (src, dst)), shape=(n, n)).tocsr()
    distances = dijkstra(graph, directed=False)
    index = np.full((n, k), -1, dtype=np.int64)
    dist = np.zeros((n, k), dtype=np.float64)
    for q in range(n):
        order = sorted(
            (d, j) for j, d in enumerate(distances[q]) if j != q and np.isfinite(d)
        )[:k]
        for slot, (d, j) in enumerate(order):
            index[q, slot], dist[q, slot] = j, d
    return index, dist


@requires_scipy
def test_geodesic_matches_dense_dijkstra_with_weighted_edges():
    pos, edge_index = grid(6, 5, spacing=(1.0, 0.7))
    k = 7
    result = geodesic_knn(pos, edge_index, k)
    index, dist = dense_geodesic_reference(pos, edge_index, k)
    assert result.mask.all()
    torch.testing.assert_close(result.index, torch.from_numpy(index))
    torch.testing.assert_close(
        result.distance.double(), torch.from_numpy(dist), atol=1e-6, rtol=1e-6
    )


def test_geodesic_ties_prefer_smaller_node_id():
    pos, edge_index = grid(3, 3)
    result = geodesic_knn(pos, edge_index, 6)
    # Center node 4: four neighbors at distance 1, then diagonals at distance 2.
    assert result.index[4].tolist() == [1, 3, 5, 7, 0, 2]
    torch.testing.assert_close(
        result.distance[4], torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    )


@requires_scipy
def test_geodesic_stays_on_the_sheet_but_euclidean_crosses_the_gap():
    lower_pos, lower_edges = grid(4, 4, spacing=(10.0, 10.0), z=0.0)
    upper_pos, upper_edges = grid(4, 4, spacing=(10.0, 10.0), z=1.0, offset=16)
    pos = torch.cat((lower_pos, upper_pos))
    edge_index = torch.cat((lower_edges, upper_edges), dim=1)
    k = 5
    geodesic = geodesic_knn(pos, edge_index, k)
    euclidean = euclidean_knn(pos, k)
    assert geodesic.mask.all() and euclidean.mask.all()
    assert bool((geodesic.index[:16] < 16).all()) and bool(
        (geodesic.index[16:] >= 16).all()
    )
    # Every lower node's nearest Euclidean neighbor is the node 1 mm above it.
    torch.testing.assert_close(euclidean.index[:16, 0], torch.arange(16, 32))
    torch.testing.assert_close(euclidean.distance[:16, 0], torch.ones(16))
    dispatched = static_knn(pos, k, metric="geodesic", edge_index=edge_index)
    torch.testing.assert_close(dispatched.index, geodesic.index)
    torch.testing.assert_close(
        static_knn(pos, k, metric="euclidean").index, euclidean.index
    )


@requires_scipy
def test_euclidean_matches_a_brute_force_ranking_without_ties():
    """Random points: the KD-tree result equals a float64 all-pairs ranking."""
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
    assert result.index.numpy().tolist() == order.tolist()
    np.testing.assert_allclose(
        result.distance.numpy(),
        np.take_along_axis(distances, order, axis=1),
        rtol=1e-6,
        atol=1e-6,
    )


@requires_scipy
def test_euclidean_ties_at_the_k_boundary_follow_the_kd_tree():
    """Documents the tie rule: boundary ties are the KD-tree's choice, returned
    candidates at equal distance are ordered by node ID, and runs are
    deterministic."""
    pos, _ = grid(5, 5)
    first, second = euclidean_knn(pos, 1), euclidean_knn(pos, 1)
    torch.testing.assert_close(first.index, second.index)
    # Corner node 0 has nodes 1 and 5 at distance 1: one of them is returned.
    assert first.index[0].item() in (1, 5)
    torch.testing.assert_close(first.distance[0], torch.tensor([1.0]))
    # Centre node 12 has four nodes at distance 1, listed by node ID.
    assert euclidean_knn(pos, 4).index[12].tolist() == [7, 11, 13, 17]


@pytest.mark.skipif(_has_scipy(), reason="documents the error without SciPy")
def test_euclidean_names_scipy_when_it_is_missing():
    with pytest.raises(Exception, match="SciPy"):
        euclidean_knn(torch.zeros(3, 3), 2)


PADDED_POSITIONS = torch.tensor(
    [[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [50, 0, 0], [51, 0, 0]]
)


def test_small_components_are_padded():
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    result = geodesic_knn(PADDED_POSITIONS, edge_index, 4)
    assert result.mask.sum(1).tolist() == [2, 2, 2, 1, 1]
    assert result.index[3].tolist() == [4, 3, 3, 3]  # padding repeats the query
    torch.testing.assert_close(result.distance[3], torch.tensor([1.0, 0, 0, 0]))


@requires_scipy
def test_small_point_sets_are_padded():
    few = euclidean_knn(PADDED_POSITIONS[:3], 5)
    assert few.mask.sum(1).tolist() == [2, 2, 2]
    assert few.index[0].tolist() == [1, 2, 0, 0, 0]


def test_self_loops_duplicates_and_zero_length_edges():
    pos, edge_index = grid(3, 2)
    noisy = torch.cat(
        (edge_index, torch.tensor([[0, 0, 1], [0, 1, 0]]), edge_index[:, :3]), dim=1
    )
    clean = geodesic_knn(pos, edge_index, 3)
    with_noise = geodesic_knn(pos, noisy, 3)
    torch.testing.assert_close(with_noise.index, clean.index)
    torch.testing.assert_close(with_noise.distance, clean.distance)
    coincident = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    result = geodesic_knn(coincident, torch.tensor([[0, 1], [1, 2]]), 2)
    assert result.index[0].tolist() == [1, 2]
    torch.testing.assert_close(result.distance[0], torch.tensor([0.0, 1.0]))


def test_determinism_and_validation():
    pos, edge_index = grid(5, 4, spacing=(0.3, 1.1))
    first, second = geodesic_knn(pos, edge_index, 6), geodesic_knn(pos, edge_index, 6)
    torch.testing.assert_close(first.index, second.index)
    with pytest.raises(ValueError, match="metric"):
        static_knn(pos, 3, metric="hops", edge_index=edge_index)
    with pytest.raises(ValueError, match="edge_index"):
        static_knn(pos, 3, metric="geodesic")
    with pytest.raises(ValueError, match="positive integer"):
        geodesic_knn(pos, edge_index, 0)
    with pytest.raises(ValueError, match="CPU float"):
        geodesic_knn(pos.half(), edge_index, 3)
    with pytest.raises(ValueError, match="int64"):
        geodesic_knn(pos, edge_index.int(), 3)
    with pytest.raises(ValueError, match="outside"):
        geodesic_knn(pos, torch.tensor([[0], [99]]), 3)
    assert geodesic_knn(
        pos[:0], torch.empty((2, 0), dtype=torch.long), 3
    ).index.shape == (0, 3)
