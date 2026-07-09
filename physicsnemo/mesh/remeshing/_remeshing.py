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

from physicsnemo.nn.functional.geometry.remeshing import WarpRemeshOptions, remeshing

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _validate_remesh_inputs(
    mesh: Mesh,
    n_clusters: int,
    max_iterations: int | None,
) -> None:
    """Validate Mesh-level remeshing invariants."""
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
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError(
                "max_iterations must be an integer or None, got "
                f"{type(max_iterations).__name__}"
            )
        if max_iterations < 0:
            raise ValueError(
                f"max_iterations must be non-negative, got {max_iterations}"
            )

    if mesh.n_cells == 0:
        raise ValueError("remesh requires at least one triangle")
    if not torch.is_floating_point(mesh.points):
        raise TypeError(
            f"mesh points must use a floating-point dtype, got {mesh.points.dtype}"
        )


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int | None = None,
    warp_options: WarpRemeshOptions | None = None,
) -> Mesh:
    """Uniformly remesh a CUDA triangle surface using Warp.

    Warp performs area-weighted centroidal clustering, projects cluster centers
    back to the source surface with a GPU bounding volume hierarchy, and
    reconstructs compact triangle connectivity.

    Parameters
    ----------
    mesh : Mesh
        Input triangle surface. Only 2D triangle manifolds embedded in 3D are
        supported.
    n_clusters : int
        Target output vertex count. Cleanup can produce slightly fewer vertices.
        Must be between 3 and the input point count, inclusive.
    max_iterations : int | None, optional
        Maximum centroid-relaxation iterations. ``None`` uses four iterations.
        Values must be non-negative.
    warp_options : WarpRemeshOptions | None, optional
        Performance and initialization controls for Warp.

    Returns
    -------
    Mesh
        Geometry-only remeshed surface on the input device. Point and cell data
        are discarded because topology changes; global data is preserved.

    Raises
    ------
    TypeError
        If counts, options, or point coordinates have invalid types.
    ValueError
        If the mesh is not on CUDA, a count is out of range, or coordinates or
        connectivity are invalid.
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
    if warp_options is not None and not isinstance(warp_options, WarpRemeshOptions):
        raise TypeError(
            "warp_options must be a WarpRemeshOptions instance or None, got "
            f"{type(warp_options).__name__}"
        )

    _validate_remesh_inputs(mesh, n_clusters, max_iterations)

    if not mesh.points.is_cuda:
        raise ValueError(
            "remesh requires a CUDA mesh; move the mesh to CUDA before calling it"
        )
    output_points, output_cells = remeshing(
        mesh.points,
        mesh.cells,
        n_clusters,
        max_iterations=max_iterations,
        warp_options=warp_options,
    )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=output_points,
        cells=output_cells,
        global_data=mesh.global_data.clone(),
    )


__all__ = ["WarpRemeshOptions", "remesh"]
