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
"""Geodesic k-nearest reference neighborhoods of mesh nodes.

Neighborhoods are computed once on the initial (reference) geometry and reused
at every rollout step, so they describe material locality rather than the
deformed state. Candidates are ranked by shortest-path length along the
supplied structural edges, weighted by reference Euclidean edge lengths: the
query never crosses between disconnected parts and ranks folded or stacked
sheets by their path distance, where a straight-line query would connect
spatially adjacent but materially separate shells. Straight-line (Euclidean)
neighborhoods are the ordinary kNN problem and come from
:func:`physicsnemo.nn.functional.knn`. Selection is discrete and carries no
autograd graph. Results are deterministic: ties are broken by the smaller node
ID, at the ``k`` boundary as well.
"""

from __future__ import annotations

import heapq
import math
from typing import NamedTuple

import numpy as np
import torch


class ReferenceNeighborhood(NamedTuple):
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
) -> ReferenceNeighborhood:
    index = np.arange(n, dtype=np.int64)[:, None].repeat(k, axis=1)
    distance = np.zeros((n, k), dtype=np.float32)
    mask = np.zeros((n, k), dtype=bool)
    for q, (ids, ds) in enumerate(zip(rows, dists)):
        count = len(ids)
        if count:
            index[q, :count] = ids
            distance[q, :count] = ds
            mask[q, :count] = True
    return ReferenceNeighborhood(
        torch.from_numpy(index), torch.from_numpy(distance), torch.from_numpy(mask)
    )


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
) -> ReferenceNeighborhood:
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
