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

"""Input validation for point-to-mesh distance backends."""

from __future__ import annotations

import torch


def _assert_tensor_contract(condition: torch.Tensor, message: str) -> None:
    """Assert a scalar tensor predicate without a device-to-host synchronization."""

    torch._assert_async(condition, message)


def normalize_point_to_mesh_inputs(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    squared: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    """Validate inputs and flatten the query prefix.

    The target topology is normalized to ``int64`` for native-Torch indexing,
    while coordinate tensors keep their original dtype and autograd history.
    """

    if mesh_vertices.ndim != 2 or mesh_vertices.shape[-1] != 3:
        raise ValueError(
            "mesh_vertices must have shape (num_vertices, 3), got "
            f"{tuple(mesh_vertices.shape)}"
        )
    if mesh_vertices.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "mesh_vertices must have dtype torch.float32 or torch.float64, got "
            f"{mesh_vertices.dtype}"
        )
    if mesh_vertices.shape[0] == 0:
        raise ValueError("mesh_vertices must contain at least one vertex")

    if mesh_indices.ndim != 2 or mesh_indices.shape[-1] != 3:
        raise ValueError(
            "mesh_indices must have shape (num_faces, 3), got "
            f"{tuple(mesh_indices.shape)}"
        )
    if mesh_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "mesh_indices must have dtype torch.int32 or torch.int64, got "
            f"{mesh_indices.dtype}"
        )
    if mesh_indices.shape[0] == 0:
        raise ValueError("mesh_indices must contain at least one triangle")

    if input_points.ndim < 1 or input_points.shape[-1] != 3:
        raise ValueError(
            f"input_points must have shape (..., 3), got {tuple(input_points.shape)}"
        )
    if input_points.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "input_points must have dtype torch.float32 or torch.float64, got "
            f"{input_points.dtype}"
        )
    if input_points.dtype != mesh_vertices.dtype:
        raise TypeError(
            "mesh_vertices and input_points must have the same dtype, got "
            f"{mesh_vertices.dtype} and {input_points.dtype}"
        )
    if input_points.device != mesh_vertices.device:
        raise ValueError(
            "mesh_vertices and input_points must be on the same device, got "
            f"{mesh_vertices.device} and {input_points.device}"
        )
    if mesh_indices.device != mesh_vertices.device:
        raise ValueError(
            "mesh_vertices and mesh_indices must be on the same device, got "
            f"{mesh_vertices.device} and {mesh_indices.device}"
        )
    if not isinstance(squared, bool):
        raise TypeError(f"squared must be a bool, got {type(squared).__name__}")

    _assert_tensor_contract(
        torch.isfinite(mesh_vertices.detach()).all(),
        "mesh_vertices must contain only finite coordinates",
    )
    _assert_tensor_contract(
        torch.isfinite(input_points.detach()).all(),
        "input_points must contain only finite coordinates",
    )

    faces = mesh_indices.to(torch.long)
    num_vertices = mesh_vertices.shape[0]
    indices_in_bounds = ((faces >= 0) & (faces < num_vertices)).all()
    _assert_tensor_contract(
        indices_in_bounds,
        "mesh_indices values must satisfy 0 <= index < num_vertices",
    )

    # Clamp solely to keep downstream gathers memory-safe while a failed CUDA
    # assertion is reported asynchronously. Valid connectivity is unchanged.
    safe_faces = faces.clamp(0, num_vertices - 1)
    triangles = mesh_vertices.detach()[safe_faces]
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    area_vector = torch.linalg.cross(first_edge, second_edge, dim=-1)
    finite_area = torch.isfinite(area_vector).all(dim=-1)
    nonzero_area = (area_vector != 0).any(dim=-1)
    _assert_tensor_contract(
        (finite_area & nonzero_area).all(),
        "mesh triangles must be nondegenerate with finite area",
    )

    input_shape = input_points.shape
    return (
        mesh_vertices,
        safe_faces,
        input_points.reshape(-1, 3),
        input_shape,
    )


__all__ = ["normalize_point_to_mesh_inputs"]
