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

from physicsnemo.nn.functional.geometry.remeshing import remeshing

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int = 4,
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

    Returns
    -------
    Mesh
        Geometry-only remeshed surface on the input device. Point and cell data
        are discarded because topology changes. Global data is preserved.

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
    coordinate frame. Because clustering uses spatial distance rather than
    mesh connectivity, sheets or thin features separated by less than the mean
    cluster spacing can be assigned to a common cluster and welded together.
    Projection can map distinct cluster centroids to the same surface position.
    Output vertices are compacted by connectivity but are not welded by
    position. Backend-specific tuning remains available through
    :func:`physicsnemo.nn.functional.remeshing`. These advanced parameters may
    change as the implementation evolves.
    """
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 3D. Got "
            f"n_manifold_dims={mesh.n_manifold_dims} and "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )

    output_points, output_cells = remeshing(
        mesh.points,
        mesh.cells,
        n_clusters,
        max_iterations=max_iterations,
    )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=output_points,
        cells=output_cells,
        global_data=mesh.global_data.clone(),
    )


__all__ = ["remesh"]
