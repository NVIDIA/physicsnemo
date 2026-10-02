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

"""Static, gap-scaled exclusions using initial material-edge path distances."""

import heapq
import math

import numpy as np
import torch

from physicsnemo.core.version_check import OptionalImport

# EXT-004: delay optional imports until this preprocessing feature is called.
_scipy_sparse = OptionalImport(
    "scipy.sparse",
    package_hint=(
        "reference_geodesic_exclusions requires SciPy. Install with "
        "pip install 'nvidia-physicsnemo[nn-extras]' or pip install scipy."
    ),
)


@torch.no_grad()
def reference_geodesic_exclusions(
    reference_positions: torch.Tensor,
    faces: torch.Tensor,
    node_gap: torch.Tensor,
    face_gap: torch.Tensor,
    *,
    distance_scale: float = math.sqrt(2.0),
    gap_scale: float = 1.0,
    gap_min: float = 0.0,
    max_pairs: int = 16_000_000,
) -> torch.Tensor:
    """Return sorted unique CPU [node, facet] exclusions, including incidence.

    For q outside facet f, exclude iff min_v_in_f d0(q,v) is STRICTLY less than
    distance_scale * max(gap_min, gap_scale * (node_gap[q] + face_gap[f])).
    d0 is shortest-path length along ORIGINAL triangle/quad PERIMETER edges,
    weighted by initial physical-coordinate Euclidean lengths. This is an
    edge-geodesic approximation, not a continuous-surface geodesic or an exact
    reproduction of a solver's unavailable contact deck. Disconnected sheets
    have infinite distance even when coincident. No artificial quad diagonal,
    spatial kNN edge, future frame, or normalized coordinate is used.

    Caller supplies nonnegative half-gap vectors, e.g. half nodal thickness and
    half max facet-vertex thickness. They are unrelated to a predictive message
    band. distance_scale=0 disables the optional filter, keeping incidence only.
    This frozen eligibility preprocessing intentionally has no autograd graph;
    live contact geometry/weights remain differentiable downstream.

    Bounded Dijkstra visits only nodes within each query's maximum possible
    pair threshold. No dense N-by-N distances are allocated. A storage budget
    overflow raises; it never silently drops exclusions or contact pairs.
    """
    coo_matrix = _scipy_sparse.coo_matrix

    if (
        reference_positions.device.type != "cpu"
        or reference_positions.dtype not in (torch.float32, torch.float64)
        or reference_positions.ndim != 2
        or reference_positions.shape[1] != 3
        or not torch.isfinite(reference_positions).all()
    ):
        raise ValueError("reference_positions must be finite CPU float [N,3]")
    n = len(reference_positions)
    if (
        faces.device.type != "cpu"
        or faces.dtype != torch.long
        or faces.ndim != 2
        or faces.shape[1] not in (3, 4)
        or (faces.numel() and (faces.min() < 0 or faces.max() >= n))
    ):
        raise ValueError("faces must be CPU int64 [F,3|4] with valid node IDs")
    for value, size in ((node_gap, n), (face_gap, len(faces))):
        if (
            value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.float64)
            or value.shape != (size,)
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError("gap vectors must be finite nonnegative CPU floats")
    for value in (distance_scale, gap_scale, gap_min):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(
                "gap/distance scales and gap_min must be finite nonnegative"
            )
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer")
    if not len(faces):
        return torch.empty((2, 0), dtype=torch.long)
    xyz = reference_positions.detach().numpy().astype(np.float64)
    facets = faces.numpy()
    ng = node_gap.detach().numpy().astype(np.float64)
    fg = face_gap.detach().numpy().astype(np.float64)
    # A padded triangle has a repeated terminal corner, not a fourth vertex.
    ordered = np.sort(facets, axis=1)
    repeats = np.diff(ordered, axis=1) == 0
    if faces.shape[1] == 3:
        valid = ~repeats.any(axis=1)
    else:
        valid = (~repeats.any(axis=1)) | (
            (facets[:, 2] == facets[:, 3]) & (repeats.sum(axis=1) == 1)
        )
    if not valid.all():
        raise ValueError("invalid repeated facet vertices")
    incidence = coo_matrix(
        (
            np.ones(facets.size, dtype=bool),
            (facets.ravel(), np.repeat(np.arange(len(faces)), faces.shape[1])),
        ),
        shape=(n, len(faces)),
    ).tocsr()
    if incidence.nnz > max_pairs:
        raise RuntimeError(
            "reference-geodesic exclusion budget exceeded; no pairs truncated"
        )
    if distance_scale == 0:
        q, f = incidence.nonzero()
        return torch.from_numpy(np.stack((q, f)).astype(np.int64))
    edges = np.stack((facets, np.roll(facets, -1, axis=1)), axis=-1).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("reference edge length overflow")
    adjacency = coo_matrix(
        (
            np.tile(lengths, 2),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(n, n),
    ).tocsr()
    # Preserve zero-length material edges. Distinct coincident sheets still have
    # no connecting edge. Duplicate edge weights were removed before CSR build.
    bounds = distance_scale * np.maximum(gap_min, gap_scale * (ng + fg.max()))
    if not np.isfinite(bounds).all():
        raise ValueError("reference-geodesic radius overflow")
    output = []
    total = 0
    for q in range(n):
        incident = incidence.indices[incidence.indptr[q] : incidence.indptr[q + 1]]
        excluded = set(incident.tolist())
        distances = {q: 0.0}
        heap = [(0.0, q)]
        bound = bounds[q]
        while heap:
            distance, vertex = heapq.heappop(heap)
            if distance != distances[vertex] or distance >= bound:
                continue
            adjacent_faces = incidence.indices[
                incidence.indptr[vertex] : incidence.indptr[vertex + 1]
            ]
            thresholds = distance_scale * np.maximum(
                gap_min, gap_scale * (ng[q] + fg[adjacent_faces])
            )
            excluded.update(adjacent_faces[distance < thresholds].tolist())
            for j in range(adjacency.indptr[vertex], adjacency.indptr[vertex + 1]):
                neighbor = int(adjacency.indices[j])
                candidate = distance + adjacency.data[j]
                if candidate < bound and candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
        total += len(excluded)
        if total > max_pairs:
            raise RuntimeError(
                "reference-geodesic exclusion budget exceeded; no pairs truncated"
            )
        if excluded:
            ids = np.array(sorted(excluded), dtype=np.int64)
            output.append(np.stack((np.full(len(ids), q, dtype=np.int64), ids)))
    pairs = (
        np.concatenate(output, axis=1) if output else np.empty((2, 0), dtype=np.int64)
    )
    return torch.from_numpy(pairs)
