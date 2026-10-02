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

"""Public topology-filtered contact-search functional."""

import torch
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.neighbors.radius_search.radius_search import (
    RadiusSearch,
)

from ._torch_impl import contact_search as contact_search_torch
from ._warp_impl import contact_search as contact_search_warp


class ContactSearch(FunctionSpec):
    r"""Find broad-phase contact candidates while excluding graph-local pairs.

    ``contact_search`` performs a Euclidean radius search in the current
    (world-space) coordinates and fuses a per-query sparse exclusion test into
    the neighbor selection. It is intended for self-contact and learned-contact
    graph construction, where points that are close through the material mesh
    should not become contact edges merely because they are also close in world
    space.

    The exclusion relation uses compressed sparse row (CSR) storage. For query
    ``q``, point indices in
    ``exclude_neighbors[exclude_offsets[q]:exclude_offsets[q + 1]]`` are
    discarded. Each row **must be sorted in ascending order**, because the Warp
    backend uses binary search. Include the query point itself in its exclusion
    row when self-pairs must be removed.

    For batched point clouds, a CSR with ``Q`` rows is shared by every batch
    element, which is efficient for trajectories with common mesh topology. A
    CSR with ``B * Q`` rows can instead provide distinct exclusions for every
    batch. Per-batch rows are flattened in ``(batch, query)`` order, and CSR
    point indices remain local to their batch element.

    The CSR is deliberately supplied rather than constructed inside this
    functional. Mesh topology and reference-space geodesic neighborhoods are
    normally static, so precomputing them avoids graph traversal in every
    simulation step. The relation may represent direct mesh adjacency, k-hop
    neighborhoods, a length-weighted geodesic ball, or any other pairs that
    should be suppressed.

    Parameters
    ----------
    points : torch.Tensor
        Candidate point coordinates with shape ``(N, 3)`` or ``(B, N, 3)``.
        Warp supports ``float16``, ``bfloat16``, and ``float32``; use
        ``implementation="torch"`` for ``float64`` coordinates. Warp rejects
        unsupported dtypes rather than silently reducing their precision.
    queries : torch.Tensor
        Query coordinates with shape ``(Q, 3)`` or ``(B, Q, 3)``.
        Must have the same dtype and device as ``points``.
    radius : float
        Positive Euclidean world-space search radius.
    exclude_neighbors : torch.Tensor
        Sorted excluded point indices in flattened CSR storage. Must be a
        one-dimensional ``torch.int32`` tensor on the same device as ``points``.
    exclude_offsets : torch.Tensor
        CSR row offsets. Must be a one-dimensional ``torch.int32`` tensor with
        length ``Q + 1`` (shared topology) or ``B * Q + 1`` (per-batch
        topology), starting at zero and ending at
        ``len(exclude_neighbors)``.
    max_points : int or None, optional
        Maximum candidates returned per query. ``None`` returns all candidates
        as a dynamic COO edge list. A positive integer returns the nearest
        eligible candidates within ``radius``, ordered by increasing distance,
        in static padded tensors compatible with ``torch.compile``. Exclusions
        are applied before nearest-k selection. Default is ``None``.
    return_dists : bool, optional
        Return Euclidean world-space candidate distances. Default is ``False``.
    return_points : bool, optional
        Return candidate point coordinates. Default is ``False``.
    implementation : {"warp", "torch"} or None, optional
        Backend override. ``None`` selects the accelerated Warp backend when
        available and otherwise falls back to PyTorch.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, ...]
        Candidate indices are returned first. With ``max_points=None``, an
        unbatched search returns ``(2, E)`` query/point COO indices and a
        batched search returns ``(3, E)`` batch/query/point indices. With
        ``max_points`` set, indices have shape ``(Q, max_points)`` or
        ``(B, Q, max_points)`` and unused indices are ``-1``. Use
        ``indices >= 0`` as the validity mask, and mask out invalid slots before
        indexing into point data. Requested points and distances follow the
        same layout and are zero-filled for invalid slots. A zero distance is
        valid for coincident contact points and must not be used as a padding
        test.

    Notes
    -----
    This is broad-phase candidate generation, not a narrow-phase contact
    solver. Vertex-face or edge-edge tests are still needed for robust physical
    contact constraints. Dynamic COO order is unspecified. For capped outputs,
    ties at equal distance may be ordered or selected differently by the two
    backends. The Warp capped search scans all spatial candidates while retaining
    only ``max_points`` entries per query in a bounded heap; it does not allocate
    a dense pairwise distance matrix or materialize every contact edge. Candidate
    selection is non-differentiable. Only returned point coordinates have
    backend-consistent gradient support; do not rely on gradients through
    returned distances.
    """

    _BENCHMARK_CASES = (
        ("small-p1024-q512-r0p1-m32-x5", 1, 1024, 512, 0.1, 32, 5),
        ("medium-p4096-q2048-r0p1-m32-x5", 1, 4096, 2048, 0.1, 32, 5),
        ("large-p8192-q4096-r0p1-m32-x5", 1, 8192, 4096, 0.1, 32, 5),
        ("batched-b4-p1024-q512-r0p1-m32-x5", 4, 1024, 512, 0.1, 32, 5),
    )
    _BACKWARD_BENCHMARK_CASES = _BENCHMARK_CASES

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        queries: Float[torch.Tensor, "*batch num_queries 3"],
        radius: float,
        exclude_neighbors: Int[torch.Tensor, " num_exclusions"],
        exclude_offsets: Int[torch.Tensor, " num_exclusion_rows_plus_one"],
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Warp hash-grid implementation with fused CSR filtering."""
        return contact_search_warp(
            points,
            queries,
            radius,
            exclude_neighbors,
            exclude_offsets,
            max_points,
            return_dists,
            return_points,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        queries: Float[torch.Tensor, "*batch num_queries 3"],
        radius: float,
        exclude_neighbors: Int[torch.Tensor, " num_exclusions"],
        exclude_offsets: Int[torch.Tensor, " num_exclusion_rows_plus_one"],
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Pure-PyTorch reference implementation via ``torch.cdist``."""
        return contact_search_torch(
            points,
            queries,
            radius,
            exclude_neighbors,
            exclude_offsets,
            max_points,
            return_dists,
            return_points,
        )

    @staticmethod
    def _benchmark_inputs(
        batch_size: int,
        num_points: int,
        num_queries: int,
        exclusion_width: int,
        device: torch.device,
        requires_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.rand(
            batch_size,
            num_points,
            3,
            device=device,
            requires_grad=requires_grad,
        )
        queries = torch.rand(
            batch_size,
            num_queries,
            3,
            device=device,
            requires_grad=requires_grad,
        )
        half_width = exclusion_width // 2
        relative = torch.arange(
            -half_width,
            exclusion_width - half_width,
            dtype=torch.int32,
            device=device,
        )
        query_ids = torch.arange(
            num_queries, dtype=torch.int32, device=device
        ).unsqueeze(1)
        exclude_neighbors = torch.clamp(
            query_ids + relative.unsqueeze(0), 0, num_points - 1
        ).flatten()
        exclude_offsets = torch.arange(
            0,
            (num_queries + 1) * exclusion_width,
            exclusion_width,
            dtype=torch.int32,
            device=device,
        )
        if batch_size == 1:
            points = points.squeeze(0)
            queries = queries.squeeze(0)
        return points, queries, exclude_neighbors, exclude_offsets

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative topology-filtered radius-search workloads."""
        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            num_queries,
            radius,
            max_points,
            exclusion_width,
        ) in cls._BENCHMARK_CASES:
            points, queries, neighbors, offsets = cls._benchmark_inputs(
                batch_size,
                num_points,
                num_queries,
                exclusion_width,
                device,
                requires_grad=False,
            )
            yield (
                label,
                (points, queries, radius, neighbors, offsets),
                {
                    "max_points": max_points,
                    "return_dists": True,
                    "return_points": True,
                },
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield workloads that exercise selected-point gradients."""
        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            num_queries,
            radius,
            max_points,
            exclusion_width,
        ) in cls._BACKWARD_BENCHMARK_CASES:
            points, queries, neighbors, offsets = cls._benchmark_inputs(
                batch_size,
                num_points,
                num_queries,
                exclusion_width,
                device,
                requires_grad=True,
            )
            yield (
                f"{label}-bwd",
                (points, queries, radius, neighbors, offsets),
                {
                    "max_points": max_points,
                    "return_dists": False,
                    "return_points": True,
                },
            )

    @classmethod
    def compare_forward(cls, output: tuple, reference: tuple) -> None:
        """Compare backends without relying on hash-grid neighbor ordering."""
        RadiusSearch.compare_forward(output, reference)

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare gradients propagated through selected point coordinates."""
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)


contact_search = ContactSearch.make_function("contact_search")

__all__ = ["ContactSearch", "contact_search"]
