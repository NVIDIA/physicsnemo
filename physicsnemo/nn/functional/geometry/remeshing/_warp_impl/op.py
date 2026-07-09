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

"""Torch custom-op registration for Warp remeshing."""

import torch

from .._config import WarpRemeshOptions
from .launch_forward import launch_remeshing


@torch.library.custom_op(
    "physicsnemo::remeshing_warp",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_bitwise,),
)
def remeshing_warp(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    n_clusters: int,
    max_iterations: int,
    search_radius_scale: float,
    voxel_width_scale: float,
    hash_grid_resolution: int,
    farthest_point_threshold: int,
    farthest_point_oversampling: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute Warp remeshing as an opaque, non-differentiable Torch op."""
    options = WarpRemeshOptions(
        search_radius_scale=search_radius_scale,
        voxel_width_scale=voxel_width_scale,
        hash_grid_resolution=hash_grid_resolution,
        farthest_point_threshold=farthest_point_threshold,
        farthest_point_oversampling=farthest_point_oversampling,
    )
    with torch.no_grad():
        output_vertices, output_indices = launch_remeshing(
            mesh_vertices.detach(),
            mesh_indices.detach(),
            n_clusters,
            max_iterations=None if max_iterations < 0 else max_iterations,
            options=options,
        )

    if not 3 <= output_vertices.shape[0] <= n_clusters:
        raise RuntimeError(
            "Warp remeshing returned an invalid number of output vertices"
        )
    if output_indices.shape[0] < 1:
        raise RuntimeError("Warp remeshing returned no output triangles")
    return output_vertices, output_indices


@remeshing_warp.register_fake
def _remeshing_warp_fake(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    n_clusters: int,
    max_iterations: int,
    search_radius_scale: float,
    voxel_width_scale: float,
    hash_grid_resolution: int,
    farthest_point_threshold: int,
    farthest_point_oversampling: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe the data-dependent remeshing output shapes for FakeTensor."""
    del (
        max_iterations,
        search_radius_scale,
        voxel_width_scale,
        hash_grid_resolution,
        farthest_point_threshold,
        farthest_point_oversampling,
    )
    context = torch.library.get_ctx()
    output_vertex_count = context.new_dynamic_size(min=3, max=n_clusters)
    output_face_count = context.new_dynamic_size(min=1)
    output_vertices = mesh_vertices.new_empty((output_vertex_count, 3))
    output_indices = mesh_indices.new_empty(
        (output_face_count, 3),
        dtype=torch.int64,
    )
    return output_vertices, output_indices


__all__ = ["remeshing_warp"]
