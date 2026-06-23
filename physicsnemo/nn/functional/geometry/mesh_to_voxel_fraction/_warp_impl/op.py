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

from collections.abc import Sequence

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ._kernels import (
    _voxel_mesh_intersection_kernel,
    _voxel_open_mesh_intersection_kernel,
)

wp.init()
wp.config.quiet = True


# ----------------------------------------------------------------------------
# Input Normalization Helpers
# ----------------------------------------------------------------------------
def _normalize_mesh_indices(
    mesh_indices: torch.Tensor,
    *,
    n_vertices: int | None = None,
) -> torch.Tensor:
    # Mesh connectivity must use an integer dtype.
    if mesh_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("mesh_indices must use an integer dtype")

    # Accept either flattened indices or (n_faces, 3) connectivity.
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError("mesh_indices with rank 2 must have shape (n_faces, 3)")
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError(
            "mesh_indices must be either rank-1 flattened indices or rank-2 (n_faces, 3)"
        )

    # Flattened connectivity must contain complete triangle triplets.
    if mesh_indices.numel() == 0 or mesh_indices.numel() % 3 != 0:
        raise ValueError(
            "mesh_indices must contain a positive number of triangle-triplet indices"
        )

    # Validate index bounds when vertex count is provided.
    if n_vertices is not None:
        min_index = int(mesh_indices.min().item())
        max_index = int(mesh_indices.max().item())
        if min_index < 0 or max_index >= n_vertices:
            raise ValueError("mesh_indices values must satisfy 0 <= index < n_vertices")
    return mesh_indices


def _normalize_origin(
    origin: torch.Tensor | Sequence[float],
    *,
    device: torch.device,
) -> torch.Tensor:
    # Convert origin input to a float32 tensor on the target device.
    if torch.is_tensor(origin):
        origin_tensor = origin.to(device=device, dtype=torch.float32)
    else:
        origin_tensor = torch.tensor(origin, device=device, dtype=torch.float32)

    # Origin must be a length-3 coordinate vector.
    if origin_tensor.ndim != 1 or origin_tensor.numel() != 3:
        raise ValueError("origin must be a length-3 vector")
    return origin_tensor


def _normalize_grid_dims(
    grid_dims: torch.Tensor | Sequence[int],
) -> tuple[int, int, int]:
    # Convert grid dimensions to a Python integer triplet.
    if torch.is_tensor(grid_dims):
        if grid_dims.ndim != 1 or grid_dims.numel() != 3:
            raise ValueError("grid_dims must contain exactly three values")
        dims = (
            int(grid_dims[0].item()),
            int(grid_dims[1].item()),
            int(grid_dims[2].item()),
        )
    else:
        if len(grid_dims) != 3:
            raise ValueError("grid_dims must contain exactly three values")
        dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    # Grid resolution in each axis must be positive.
    if dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0:
        raise ValueError("grid_dims values must be strictly positive")
    return dims


# ----------------------------------------------------------------------------
# Warp Custom Operator
# ----------------------------------------------------------------------------
@torch.library.custom_op("physicsnemo::mesh_to_voxel_fraction_warp", mutates_args=())
def mesh_to_voxel_fraction_impl(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    n_samples: int = 64,
    seed: int = 42,
    open_mesh: bool = False,
    winding_number_threshold: float = 0.5,
    winding_number_accuracy: float = 2.0,
) -> torch.Tensor:
    """Execute the Warp voxel-fraction kernel on normalized tensor inputs.

    Parameters are already expanded to scalar grid dimensions and a tensor
    ``origin`` so this function can serve as the low-level custom op entrypoint.
    """
    # Validate mesh and parameter inputs.
    if mesh_vertices.device != mesh_indices.device:
        raise ValueError("mesh_vertices and mesh_indices must be on the same device")
    if mesh_vertices.device != origin.device:
        raise ValueError("mesh_vertices and origin must be on the same device")
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[-1] != 3:
        raise ValueError("mesh_vertices must have shape (n_vertices, 3)")
    if mesh_indices.ndim != 1:
        raise ValueError("mesh_indices must be flattened (rank-1) in the custom op")
    if mesh_indices.numel() == 0 or mesh_indices.numel() % 3 != 0:
        raise ValueError(
            "mesh_indices must contain a positive number of triangle-triplet indices"
        )
    min_index = int(mesh_indices.min().item())
    max_index = int(mesh_indices.max().item())
    if min_index < 0 or max_index >= mesh_vertices.shape[0]:
        raise ValueError("mesh_indices values must satisfy 0 <= index < n_vertices")
    if origin.ndim != 1 or origin.numel() != 3:
        raise ValueError("origin must be a length-3 tensor")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be strictly positive")
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, and nz must be strictly positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be strictly positive")
    if winding_number_accuracy <= 0.0:
        raise ValueError("winding_number_accuracy must be strictly positive")

    # Normalize dtype/layout for Warp mesh and kernel launches.
    mesh_vertices = mesh_vertices.to(dtype=torch.float32).contiguous()
    mesh_indices = mesh_indices.to(dtype=torch.int32).contiguous()
    origin = origin.to(dtype=torch.float32).contiguous()

    # Allocate flattened output buffer and launch the appropriate kernel.
    output = torch.empty(nx * ny * nz, device=mesh_vertices.device, dtype=torch.float32)
    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(mesh_vertices)

    with wp.ScopedStream(wp_launch_stream):
        wp_vertices = wp.from_torch(mesh_vertices, dtype=wp.vec3)
        wp_indices = wp.from_torch(mesh_indices, dtype=wp.int32)
        wp_output = wp.from_torch(output, return_ctype=True)

        mesh = wp.Mesh(
            points=wp_vertices,
            indices=wp_indices,
            support_winding_number=open_mesh,
        )
        origin_vec = wp.vec3f(
            float(origin[0].item()),
            float(origin[1].item()),
            float(origin[2].item()),
        )

        if open_mesh:
            wp.launch(
                kernel=_voxel_open_mesh_intersection_kernel,
                dim=(nx, ny, nz),
                inputs=[
                    mesh.id,
                    origin_vec,
                    float(voxel_size),
                    nx,
                    ny,
                    nz,
                    n_samples,
                    seed,
                    float(winding_number_threshold),
                    float(winding_number_accuracy),
                    wp_output,
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )
        else:
            wp.launch(
                kernel=_voxel_mesh_intersection_kernel,
                dim=(nx, ny, nz),
                inputs=[
                    mesh.id,
                    origin_vec,
                    float(voxel_size),
                    nx,
                    ny,
                    nz,
                    n_samples,
                    seed,
                    wp_output,
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )

    # Match the original voxelizer output convention: (nz, ny, nx).
    return output.reshape(nz, ny, nx)


@mesh_to_voxel_fraction_impl.register_fake
def mesh_to_voxel_fraction_impl_fake(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    n_samples: int = 64,
    seed: int = 42,
    open_mesh: bool = False,
    winding_number_threshold: float = 0.5,
    winding_number_accuracy: float = 2.0,
) -> torch.Tensor:
    """Return a fake output tensor for tracing/compilation shape propagation."""
    if mesh_vertices.device != mesh_indices.device:
        raise ValueError("mesh_vertices and mesh_indices must be on the same device")
    if mesh_vertices.device != origin.device:
        raise ValueError("mesh_vertices and origin must be on the same device")
    return torch.empty((nz, ny, nx), device=mesh_vertices.device, dtype=torch.float32)


def mesh_to_voxel_fraction_warp(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    origin: torch.Tensor | Sequence[float],
    voxel_size: float,
    grid_dims: Sequence[int] | torch.Tensor,
    n_samples: int = 64,
    seed: int = 42,
    open_mesh: bool = False,
    winding_number_threshold: float = 0.5,
    winding_number_accuracy: float = 2.0,
) -> torch.Tensor:
    """Normalize inputs and execute the Warp mesh-to-voxel custom operator."""
    mesh_indices = _normalize_mesh_indices(
        mesh_indices,
        n_vertices=mesh_vertices.shape[0],
    )
    origin_tensor = _normalize_origin(origin, device=mesh_vertices.device)
    nx, ny, nz = _normalize_grid_dims(grid_dims)

    return mesh_to_voxel_fraction_impl(
        mesh_vertices=mesh_vertices,
        mesh_indices=mesh_indices,
        origin=origin_tensor,
        voxel_size=float(voxel_size),
        nx=nx,
        ny=ny,
        nz=nz,
        n_samples=int(n_samples),
        seed=int(seed),
        open_mesh=bool(open_mesh),
        winding_number_threshold=float(winding_number_threshold),
        winding_number_accuracy=float(winding_number_accuracy),
    )


__all__ = [
    "mesh_to_voxel_fraction_warp",
    "mesh_to_voxel_fraction_impl",
]
