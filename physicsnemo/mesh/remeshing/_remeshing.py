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

"""Public Mesh API for Warp-accelerated surface remeshing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from physicsnemo.nn.functional.geometry.remeshing import remeshing

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int = 4,
    search_radius_scale: float = 1.6,
    voxel_width_scale: float = 1.15,
    hash_grid_resolution: int = 128,
    farthest_point_threshold: int = 256,
    farthest_point_oversampling: int = 4,
) -> Mesh:
    """Uniformly remesh a triangle surface using Warp on CPU or CUDA.

    Warp performs area-weighted centroidal clustering, projects cluster centers
    back to the source surface with a bounding volume hierarchy, and
    reconstructs compact triangle connectivity.

    Parameters
    ----------
    mesh : Mesh
        Input triangle surface. Only 2D triangle manifolds embedded in 3D are
        supported.
    n_clusters : int
        Target output vertex count. Cleanup can produce slightly fewer vertices.
        Must be between 3 and the input point count, inclusive.
    max_iterations : int, optional
        Maximum centroid-relaxation iterations. Default is ``4``. Values must
        be non-negative.
    search_radius_scale : float, optional
        Hash-grid query radius relative to
        ``sqrt(surface_area / n_clusters)``. Default is ``1.6``.
    voxel_width_scale : float, optional
        Spatial-stratification voxel width relative to
        ``sqrt(surface_area / n_clusters)``. Default is ``1.15``.
    hash_grid_resolution : int, optional
        Resolution of each axis of the sparse centroid hash grid. Must be at
        most ``256``. Default is ``128``.
    farthest_point_threshold : int, optional
        Use farthest-point initialization when ``n_clusters`` is at most this
        value. Set to ``0`` to always use voxel initialization. Default is
        ``256``.
    farthest_point_oversampling : int, optional
        Area-weighted farthest-point candidate-pool size as a multiple of
        ``n_clusters``. Default is ``4``.

    Returns
    -------
    Mesh
        Geometry-only remeshed surface on the input device. Point and cell data
        are discarded because topology changes; global data is preserved.

    Raises
    ------
    TypeError
        If counts, tuning parameters, or point coordinates have invalid types.
    ValueError
        If a count is out of range or coordinates or connectivity are invalid.
    NotImplementedError
        If ``mesh`` is not a 2D triangle surface embedded in 3D.
    ImportError
        If Warp is unavailable.
    RuntimeError
        If cleanup cannot reconstruct a nonempty manifold triangle surface.

    Notes
    -----
    Remeshing is intentionally non-differentiable. Warp computes in centered
    and scaled coordinates in float32, then restores the input point dtype and
    coordinate frame. Because Warp clusters by spatial distance rather than
    mesh connectivity, very close disconnected sheets can be assigned to a
    common cluster.
    """
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise TypeError(
            f"n_clusters must be an integer, got {type(n_clusters).__name__}"
        )
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 3D; got "
            f"n_manifold_dims={mesh.n_manifold_dims} and "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )
    if n_clusters < 3:
        raise ValueError(f"n_clusters must be at least 3, got {n_clusters}")
    if n_clusters > mesh.n_points:
        raise ValueError(
            "n_clusters cannot exceed the input point count; got "
            f"n_clusters={n_clusters} and mesh.n_points={mesh.n_points}"
        )
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError(
            f"max_iterations must be an integer, got {type(max_iterations).__name__}"
        )
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")
    if mesh.n_cells == 0:
        raise ValueError("remesh requires at least one triangle")
    if not torch.is_floating_point(mesh.points):
        raise TypeError(
            f"mesh points must use a floating-point dtype, got {mesh.points.dtype}"
        )

    output_points, output_cells = remeshing(
        mesh.points,
        mesh.cells,
        n_clusters,
        max_iterations=max_iterations,
        search_radius_scale=search_radius_scale,
        voxel_width_scale=voxel_width_scale,
        hash_grid_resolution=hash_grid_resolution,
        farthest_point_threshold=farthest_point_threshold,
        farthest_point_oversampling=farthest_point_oversampling,
    )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=output_points,
        cells=output_cells,
        global_data=mesh.global_data.clone(),
    )


__all__ = ["remesh"]
