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

from __future__ import annotations

import torch


def validate_inputs(
    points: torch.Tensor,
    cells: torch.Tensor,
    neighbors: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Validate mesh, adjacency, and value tensors for Green-Gauss reconstruction."""
    ### Validate mesh point coordinates and supported spatial dimensionality.
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n_points, dims), got points.shape={points.shape}"
        )
    if points.shape[1] not in (2, 3):
        raise ValueError(
            f"mesh_green_gauss_gradient supports dims in {{2, 3}}, got {points.shape[1]}"
        )
    if not torch.is_floating_point(points):
        raise TypeError("points must be a floating-point tensor")

    ### Validate simplicial connectivity and compatibility with spatial dims.
    if cells.ndim != 2:
        raise ValueError(
            f"cells must have shape (n_cells, n_vertices), got cells.shape={cells.shape}"
        )
    expected_vertices = points.shape[1] + 1
    if cells.shape[1] != expected_vertices:
        raise ValueError(
            f"cells must contain {expected_vertices} vertices per simplex for dims={points.shape[1]}, "
            f"got {cells.shape[1]}"
        )
    if cells.dtype not in (torch.int32, torch.int64):
        raise TypeError("cells must be int32 or int64")

    ### Validate precomputed cell-neighbor tensor.
    if neighbors.ndim != 2:
        raise ValueError(
            "neighbors must have shape (n_cells, n_faces), "
            f"got neighbors.shape={neighbors.shape}"
        )
    n_cells = cells.shape[0]
    n_faces = cells.shape[1]
    if neighbors.shape != (n_cells, n_faces):
        raise ValueError(
            "neighbors shape must match (n_cells, n_faces): "
            f"expected ({n_cells}, {n_faces}), got {tuple(neighbors.shape)}"
        )
    if neighbors.dtype not in (torch.int32, torch.int64):
        raise TypeError("neighbors must be int32 or int64")

    ### Validate cell-centered values tensor.
    if values.ndim < 1:
        raise ValueError(
            f"values must have shape (n_cells, ...), got values.shape={values.shape}"
        )
    if values.shape[0] != n_cells:
        raise ValueError(
            f"values leading dimension must match n_cells: {values.shape[0]} != {n_cells}"
        )
    if not torch.is_floating_point(values):
        raise TypeError("values must be a floating-point tensor")

    ### Validate co-located tensors and index range invariants.
    if (
        points.device != cells.device
        or points.device != neighbors.device
        or points.device != values.device
    ):
        raise ValueError(
            "points, cells, neighbors, and values must be on the same device"
        )
    if cells.numel() > 0:
        idx_min = int(cells.min().item())
        idx_max = int(cells.max().item())
        if idx_min < 0 or idx_max >= points.shape[0]:
            raise ValueError(
                f"cells indices must satisfy 0 <= index < n_points ({points.shape[0]})"
            )
    if neighbors.numel() > 0:
        neigh_min = int(neighbors.min().item())
        neigh_max = int(neighbors.max().item())
        if neigh_min < -1 or neigh_max >= n_cells:
            raise ValueError(
                "neighbors entries must satisfy -1 <= index < n_cells "
                f"({n_cells}); got [{neigh_min}, {neigh_max}]"
            )


def build_neighbors(cells: torch.Tensor) -> torch.Tensor:
    """Build simplicial face-neighbor adjacency as ``(n_cells, n_faces)``.

    Face ``f`` corresponds to the simplex face opposite local vertex ``f``.
    Boundary faces are marked with ``-1``.
    """
    if cells.ndim != 2:
        raise ValueError(
            f"cells must have shape (n_cells, n_vertices), got {cells.shape=}"
        )
    if cells.dtype not in (torch.int32, torch.int64):
        raise TypeError("cells must be int32 or int64")

    n_cells, n_vertices = cells.shape
    neighbors = torch.full(
        (n_cells, n_vertices), -1, device=cells.device, dtype=torch.int64
    )

    open_faces: dict[tuple[int, ...], tuple[int, int]] = {}
    cells_cpu = cells.to(dtype=torch.int64).detach().cpu().tolist()

    for cell_idx, cell in enumerate(cells_cpu):
        for face_idx in range(n_vertices):
            face_verts = tuple(int(cell[v]) for v in range(n_vertices) if v != face_idx)
            key = tuple(sorted(face_verts))
            if key in open_faces:
                other_cell, other_face = open_faces.pop(key)
                neighbors[cell_idx, face_idx] = other_cell
                neighbors[other_cell, other_face] = cell_idx
            else:
                open_faces[key] = (cell_idx, face_idx)

    return neighbors
