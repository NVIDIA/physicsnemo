# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input normalization and reduction helpers for deformation energies."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float, Int

_Reduction = Literal["none", "sum", "mean"]


def _validate_coordinate_tensor(
    tensor: Float[torch.Tensor, "*batch num_points num_dims"],
    name: str,
) -> None:
    """Validate an unbatched or aligned-batch coordinate tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"{name} must have shape (N, D) or (B, N, D), got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] < 1:
        raise ValueError(f"{name} coordinate dimension must be at least 1")
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"{name} must have dtype torch.float32 or torch.float64, got {tensor.dtype}"
        )


def _validate_aligned_coordinates(
    points: Float[torch.Tensor, "*batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
) -> None:
    """Validate current and reference coordinates without broadcasting."""

    _validate_coordinate_tensor(points, "points")
    _validate_coordinate_tensor(reference_points, "reference_points")
    if points.shape != reference_points.shape:
        raise ValueError(
            "points and reference_points must have identical shapes, got "
            f"{tuple(points.shape)} and {tuple(reference_points.shape)}"
        )
    if points.device != reference_points.device:
        raise ValueError(
            "points and reference_points must be on the same device, got "
            f"{points.device} and {reference_points.device}"
        )
    if points.dtype != reference_points.dtype:
        raise TypeError(
            "points and reference_points must have the same dtype, got "
            f"{points.dtype} and {reference_points.dtype}"
        )


def _validate_topology(
    topology: Int[torch.Tensor, "num_primitives vertices_per_primitive"],
    points: Float[torch.Tensor, "*batch num_points num_dims"],
    *,
    name: str,
    vertices_per_primitive: int | None = None,
) -> Int[torch.Tensor, "num_primitives vertices_per_primitive"]:
    """Validate shared connectivity and normalize integer indices to int64."""

    if not isinstance(topology, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(topology).__name__}")
    if topology.ndim != 2:
        raise ValueError(f"{name} must have shape (M, K), got {tuple(topology.shape)}")
    if topology.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"{name} must have dtype torch.int32 or torch.int64, got {topology.dtype}"
        )
    if topology.device != points.device:
        raise ValueError(
            f"{name} and points must be on the same device, got "
            f"{topology.device} and {points.device}"
        )
    if (
        vertices_per_primitive is not None
        and topology.shape[1] != vertices_per_primitive
    ):
        raise ValueError(
            f"{name} must have shape (M, {vertices_per_primitive}), "
            f"got {tuple(topology.shape)}"
        )
    return topology.to(dtype=torch.int64)


def _as_batched(
    tensor: Float[torch.Tensor, "*batch num_points num_dims"],
) -> tuple[Float[torch.Tensor, "batch num_points num_dims"], bool]:
    """Normalize ``(N, D)`` coordinates to ``(1, N, D)`` without copying."""

    if tensor.ndim == 2:
        return tensor.unsqueeze(0), True
    return tensor, False


def normalize_simplex_inputs(
    points: Float[torch.Tensor, "*batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "*batch num_points num_dims"],
    cells: Int[torch.Tensor, "num_cells vertices_per_cell"],
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"],
    Float[torch.Tensor, "batch num_points num_dims"],
    Int[torch.Tensor, "num_cells vertices_per_cell"],
    bool,
]:
    """Validate simplex inputs and normalize coordinates to ``(B, N, D)``.

    The topology is shared across aligned batches. Cells may be edges,
    triangles, or tetrahedra, and their manifold dimension may not exceed the
    coordinate dimension.
    """

    _validate_aligned_coordinates(points, reference_points)
    cells = _validate_topology(cells, points, name="cells")
    simplex_dimension = cells.shape[1] - 1
    if simplex_dimension not in (1, 2, 3):
        raise ValueError(
            "cells must contain 2, 3, or 4 vertex indices per primitive, "
            f"got {cells.shape[1]}"
        )
    coordinate_dimension = points.shape[-1]
    if coordinate_dimension < simplex_dimension:
        raise ValueError(
            "the coordinate dimension must be at least the simplex dimension, "
            f"got D={coordinate_dimension} and simplex dimension "
            f"{simplex_dimension}"
        )

    points_b3, was_unbatched = _as_batched(points)
    reference_b3, _ = _as_batched(reference_points)
    return points_b3, reference_b3, cells, was_unbatched


def normalize_hinge_inputs(
    points: Float[torch.Tensor, "*batch num_points 3"],
    reference_points: Float[torch.Tensor, "*batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"],
    Float[torch.Tensor, "batch num_points 3"],
    Int[torch.Tensor, "num_hinges 4"],
    bool,
]:
    """Validate oriented triangle-hinge inputs and normalize their coordinates.

    Each row ``(i, j, k, l)`` represents the adjacent oriented faces
    ``(i, j, k)`` and ``(j, i, l)``.
    """

    _validate_aligned_coordinates(points, reference_points)
    if points.shape[-1] != 3:
        raise ValueError(
            f"surface bending requires 3D coordinates, got D={points.shape[-1]}"
        )
    hinges = _validate_topology(
        hinges,
        points,
        name="hinges",
        vertices_per_primitive=4,
    )
    points_b3, was_unbatched = _as_batched(points)
    reference_b3, _ = _as_batched(reference_points)
    return points_b3, reference_b3, hinges, was_unbatched


def normalize_closed_surface_inputs(
    points: Float[torch.Tensor, "*batch num_points 3"],
    reference_points: Float[torch.Tensor, "*batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"],
    Float[torch.Tensor, "batch num_points 3"],
    Int[torch.Tensor, "num_triangles 3"],
    bool,
]:
    """Validate oriented triangle-surface inputs and normalize coordinates."""

    _validate_aligned_coordinates(points, reference_points)
    if points.shape[-1] != 3:
        raise ValueError(
            f"closed-surface volume requires 3D coordinates, got D={points.shape[-1]}"
        )
    triangles = _validate_topology(
        triangles,
        points,
        name="triangles",
        vertices_per_primitive=3,
    )
    points_b3, was_unbatched = _as_batched(points)
    reference_b3, _ = _as_batched(reference_points)
    return points_b3, reference_b3, triangles, was_unbatched


def restore_batch(
    tensor: Float[torch.Tensor, "batch num_primitives"],
    was_unbatched: bool,
) -> Float[torch.Tensor, "*batch num_primitives"]:
    """Remove the normalization batch dimension when the input was unbatched."""

    return tensor.squeeze(0) if was_unbatched else tensor


def reduce_energy_terms(
    terms: Float[torch.Tensor, "batch num_primitives"],
    reduction: _Reduction,
    was_unbatched: bool,
) -> Float[torch.Tensor, "..."]:
    """Reduce normalized per-primitive energy terms and restore input rank.

    An empty primitive set has zero ``sum`` and ``mean`` energy. This convention
    keeps optional regularizers neutral when a mesh has no applicable
    primitives, such as a triangle surface with no interior hinges.
    """

    if reduction == "none":
        return restore_batch(terms, was_unbatched)
    elif reduction == "sum":
        return terms.sum()
    elif reduction == "mean":
        return terms.sum() / max(terms.numel(), 1)
    else:
        raise ValueError(
            f"reduction must be 'none', 'sum', or 'mean', got {reduction!r}"
        )
