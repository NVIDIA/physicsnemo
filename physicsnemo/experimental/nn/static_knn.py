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
"""Static k-nearest reference neighborhoods for mesh nodes: geodesic or Euclidean.

Neighborhoods are computed once on the initial (reference) geometry and reused
at every rollout step, so they describe material locality rather than the
deformed state. ``metric="geodesic"`` ranks candidates by shortest-path length
along the supplied structural edges, weighted by reference Euclidean edge
lengths; it never crosses between disconnected parts and ranks folded or
stacked sheets by their path distance. ``metric="euclidean"`` ranks by
straight-line reference distance and can therefore connect spatially adjacent
but materially separate shells. Selection is discrete and carries no autograd
graph. Ties are broken by the smaller node ID so results are deterministic.
"""

from __future__ import annotations

import heapq
import math
from typing import Literal, NamedTuple

import numpy as np
import torch

from physicsnemo.core.version_check import OptionalImport

_scipy_spatial = OptionalImport(
    "scipy.spatial",
    package_hint=(
        "euclidean static neighborhoods require SciPy. Install with "
        "pip install 'nvidia-physicsnemo[nn-extras]' or pip install scipy."
    ),
)

NeighborhoodMetric = Literal["geodesic", "euclidean"]


class StaticNeighborhood(NamedTuple):
    r"""Per-node neighbor IDs, reference distances, and validity mask.

    ``index`` has shape :math:`(N, k)` and excludes the query node itself. Rows
    with fewer than ``k`` reachable neighbors are padded with the query's own
    ID, zero distance, and ``mask=False``; consumers must honor the mask.
    """

    index: torch.Tensor
    distance: torch.Tensor
    mask: torch.Tensor


def _validate_positions(reference_positions: torch.Tensor) -> np.ndarray:
    if (
        not isinstance(reference_positions, torch.Tensor)
        or reference_positions.device.type != "cpu"
        or reference_positions.dtype not in (torch.float32, torch.float64)
        or reference_positions.ndim != 2
        or reference_positions.shape[1] != 3
        or not torch.isfinite(reference_positions).all()
    ):
        raise ValueError("reference_positions must be finite CPU float [N,3]")
    return reference_positions.detach().numpy().astype(np.float64)


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return k


def _padded(
    n: int, k: int, rows: list[list[int]], dists: list[list[float]]
) -> StaticNeighborhood:
    index = np.arange(n, dtype=np.int64)[:, None].repeat(k, axis=1)
    distance = np.zeros((n, k), dtype=np.float32)
    mask = np.zeros((n, k), dtype=bool)
    for q, (ids, ds) in enumerate(zip(rows, dists)):
        count = len(ids)
        if count:
            index[q, :count] = ids
            distance[q, :count] = ds
            mask[q, :count] = True
    return StaticNeighborhood(
        torch.from_numpy(index), torch.from_numpy(distance), torch.from_numpy(mask)
    )


@torch.no_grad()
def euclidean_knn(reference_positions: torch.Tensor, k: int) -> StaticNeighborhood:
    r"""Return the ``k`` nearest other nodes by straight-line reference distance.

    Coincident duplicate nodes are ordinary neighbors at zero distance; only the
    query's own ID is removed. Fewer than ``k`` other nodes are padded. Among
    the returned candidates, equal distances are ordered by node ID.
    """
    xyz = _validate_positions(reference_positions)
    k = _validate_k(k)
    n = len(xyz)
    if n == 0:
        return _padded(0, k, [], [])
    cKDTree = _scipy_spatial.cKDTree
    queries = min(k + 1, n)
    distance, index = cKDTree(xyz).query(xyz, k=queries)
    distance = np.asarray(distance, dtype=np.float64).reshape(n, queries)
    index = np.asarray(index, dtype=np.int64).reshape(n, queries)
    own = np.arange(n, dtype=np.int64)[:, None]
    # Move the query itself (if returned) last, then order the remaining
    # candidates by (distance, node ID) so ties resolve like the geodesic path.
    order = np.lexsort((index, distance, index == own), axis=1)
    index = np.take_along_axis(index, order, axis=1)[:, :k]
    distance = np.take_along_axis(distance, order, axis=1)[:, :k]
    valid = (index != own) & (index < n) & np.isfinite(distance)
    rows = [index[q, valid[q]].tolist() for q in range(n)]
    dists = [distance[q, valid[q]].tolist() for q in range(n)]
    return _padded(n, k, rows, dists)


def _reference_adjacency(xyz: np.ndarray, edge_index: torch.Tensor, n: int):
    if (
        not isinstance(edge_index, torch.Tensor)
        or edge_index.device.type != "cpu"
        or edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError("edge_index must be CPU int64 [2,E]")
    edges = edge_index.detach().numpy()
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge_index refers to nodes outside the reference mesh")
    src, dst = edges
    keep = src != dst
    low = np.minimum(src[keep], dst[keep])
    high = np.maximum(src[keep], dst[keep])
    pairs = (
        np.unique(np.stack((low, high), axis=1), axis=0)
        if low.size
        else np.empty((0, 2), dtype=np.int64)
    )
    lengths = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("reference edge length overflow")
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    weights = np.concatenate((lengths, lengths))
    order = np.lexsort((cols, rows))
    rows, cols, weights = rows[order], cols[order], weights[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    # Python lists are several times faster than NumPy scalars in the loop.
    return indptr.tolist(), cols.tolist(), weights.tolist()


@torch.no_grad()
def geodesic_knn(
    reference_positions: torch.Tensor, edge_index: torch.Tensor, k: int
) -> StaticNeighborhood:
    r"""Return the ``k`` nearest other nodes by reference edge-path distance.

    Distances are shortest paths along the undirected graph of ``edge_index``
    (self-loops and duplicate edges are ignored), with each edge weighted by its
    reference Euclidean length. Bounded Dijkstra stops after ``k`` nodes settle,
    so no dense distance matrix is allocated. Nodes in components smaller than
    ``k + 1`` receive fewer neighbors and are padded. Zero-length edges keep
    coincident connected nodes at distance zero.
    """
    xyz = _validate_positions(reference_positions)
    k = _validate_k(k)
    n = len(xyz)
    indptr, cols, weights = _reference_adjacency(xyz, edge_index, n)
    heappush, heappop = heapq.heappush, heapq.heappop
    rows: list[list[int]] = []
    dists: list[list[float]] = []
    inf = math.inf
    for source in range(n):
        heap = [(0.0, source)]
        best = {source: 0.0}
        settled: set[int] = set()
        found_ids: list[int] = []
        found_distances: list[float] = []
        while heap:
            distance, vertex = heappop(heap)
            if vertex in settled:
                continue
            settled.add(vertex)
            if vertex != source:
                found_ids.append(vertex)
                found_distances.append(distance)
                if len(found_ids) == k:
                    break
            for slot in range(indptr[vertex], indptr[vertex + 1]):
                neighbor = cols[slot]
                if neighbor in settled:
                    continue
                candidate = distance + weights[slot]
                if candidate < best.get(neighbor, inf):
                    best[neighbor] = candidate
                    heappush(heap, (candidate, neighbor))
        rows.append(found_ids)
        dists.append(found_distances)
    return _padded(n, k, rows, dists)


@torch.no_grad()
def static_knn(
    reference_positions: torch.Tensor,
    k: int,
    *,
    metric: NeighborhoodMetric = "geodesic",
    edge_index: torch.Tensor | None = None,
) -> StaticNeighborhood:
    r"""Dispatch to :func:`geodesic_knn` or :func:`euclidean_knn` by ``metric``."""
    if metric == "geodesic":
        if edge_index is None:
            raise ValueError("geodesic neighborhoods require edge_index")
        return geodesic_knn(reference_positions, edge_index, k)
    if metric == "euclidean":
        return euclidean_knn(reference_positions, k)
    raise ValueError("metric must be 'geodesic' or 'euclidean'")
