# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Main remeshing entry point.

This module wires together all components of the remeshing pipeline.
"""

from typing import TYPE_CHECKING

from physicsnemo.core.version_check import require_version_spec

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


@require_version_spec("pyacvd")
def remesh(
    mesh: "Mesh",
    n_clusters: int,
) -> "Mesh":
    """Uniform remeshing via clustering (dimension-agnostic).

    Creates a simplified mesh with approximately n_clusters cells uniformly
    distributed across the geometry. Uses the ACVD (Approximate Centroidal
    Voronoi Diagram) clustering algorithm.

    The algorithm:
    1. Weights vertices by their dual volumes (Voronoi areas)
    2. Initializes clusters via area-based region growing
    3. Minimizes energy by iteratively reassigning vertices
    4. Reconstructs a simplified mesh from cluster adjacency

    This works for arbitrary manifold dimensions (1D curves, 2D surfaces,
    3D volumes, etc.) in arbitrary embedding spaces.

    Args:
        mesh: Input mesh to remesh
        n_clusters: Target number of output cells. The actual number may vary
            slightly depending on mesh topology.

    Returns:
        Remeshed mesh with approximately n_clusters cells. The vertices are
        cluster centroids, and cells connect adjacent clusters.

    Raises:
        ValueError: If n_clusters <= 0 or weights have wrong shape

    Example:
        >>> # Remesh a triangle mesh to ~1000 triangles
        >>> from physicsnemo.mesh.remeshing import remesh
        >>> simplified = remesh(mesh, n_clusters=1000)
        >>> print(f"Original: {mesh.n_cells} cells, {mesh.n_points} points")
        >>> print(f"Remeshed: {simplified.n_cells} cells, {simplified.n_points} points")
        >>>
        >>> # With custom weights to preserve detail in certain regions
        >>> weights = torch.ones(mesh.n_points)
        >>> weights[important_region_mask] = 10.0  # 10x more clusters here
        >>> detailed = remesh(mesh, n_clusters=500, weights=weights)

    Note:
        - Works for 1D, 2D, 3D, and higher-dimensional manifolds
        - Preserves mesh topology qualitatively but not quantitatively
        - Point and cell data are not transferred (topology changes fundamentally)
        - Output cell orientation may differ from input
    """
    import importlib

    from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista
    from physicsnemo.mesh.repair import repair_mesh

    pyacvd = importlib.import_module("pyacvd")
    clustering = pyacvd.Clustering(to_pyvista(mesh))
    clustering.cluster(n_clusters)
    new_mesh = from_pyvista(clustering.create_mesh())
    new_mesh, stats = repair_mesh(new_mesh)
    return new_mesh
