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

"""Validation helpers shared by contact-search backends."""

import math

import torch

from physicsnemo.nn.functional.neighbors.radius_search.utils import validate_inputs


def validate_contact_inputs(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    max_points: int | None,
) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
    """Validate contact-search inputs and normalize point tensors.

    The exclusion relation uses compressed sparse row (CSR) storage. Rows are
    ordered like flattened query batches. A CSR with ``Q`` rows is shared by
    every batch element; a CSR with ``B * Q`` rows supplies distinct exclusions
    for every ``(batch, query)`` pair.

    Returns
    -------
    points, queries, was_unbatched, exclusion_row_stride
        Normalized point tensors have shape ``(B, N, 3)`` and ``(B, Q, 3)``.
        ``exclusion_row_stride`` is zero for shared CSR and ``Q`` for per-batch
        CSR.
    """
    if points.ndim not in (2, 3) or queries.ndim not in (2, 3):
        raise ValueError(
            "points and queries must be 2D (N, 3) or 3D (B, N, 3), "
            f"got {points.ndim}D and {queries.ndim}D"
        )
    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.dtype != queries.dtype:
        raise ValueError("points and queries must have the same dtype")
    if not points.is_floating_point() or not queries.is_floating_point():
        raise ValueError("points and queries must have floating-point dtype")
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        raise ValueError("the last dimension of points and queries must be 3")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be positive and finite, got {radius}")
    if max_points is not None and max_points <= 0:
        raise ValueError(f"max_points must be positive or None, got {max_points}")

    if (
        exclude_neighbors.device != points.device
        or exclude_offsets.device != points.device
    ):
        raise ValueError(
            "exclude_neighbors, exclude_offsets, points, and queries must be on "
            "the same device"
        )
    if exclude_neighbors.ndim != 1 or exclude_offsets.ndim != 1:
        raise ValueError("exclude_neighbors and exclude_offsets must be 1D CSR tensors")
    if exclude_neighbors.dtype != torch.int32 or exclude_offsets.dtype != torch.int32:
        raise ValueError(
            "exclude_neighbors and exclude_offsets must have dtype torch.int32"
        )

    points, queries, was_unbatched = validate_inputs(points, queries)
    batch_size, num_queries = queries.shape[:2]
    num_rows = exclude_offsets.numel() - 1

    if num_rows == num_queries:
        exclusion_row_stride = 0
    elif num_rows == batch_size * num_queries:
        exclusion_row_stride = num_queries
    else:
        raise ValueError(
            "exclude_offsets must describe either Q shared CSR rows or B * Q "
            f"per-batch CSR rows; got {num_rows} rows for B={batch_size}, "
            f"Q={num_queries}"
        )

    return points, queries, was_unbatched, exclusion_row_stride


__all__ = ["validate_contact_inputs"]
