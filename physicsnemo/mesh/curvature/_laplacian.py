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

"""Direct cotangent Laplacian computation for mean curvature.

Computes the cotangent Laplacian applied to point positions without building
the full matrix, for memory efficiency and performance.

L @ points gives the mean curvature normal (times area).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_laplacian_at_points(mesh: "Mesh") -> torch.Tensor:
    """Compute cotangent Laplacian applied to point positions directly.

    Computes L @ points where L is the cotangent Laplacian matrix, but
    without explicitly building L (more efficient).

    For each vertex i:
        (L @ points)_i = Σ_neighbors_j w_ij * (p_j - p_i)

    where w_ij are cotangent weights that depend on manifold dimension.

    Args:
        mesh: Input mesh (must be codimension-1 for mean curvature)

    Returns:
        Tensor of shape (n_points, n_spatial_dims) representing Laplacian
        applied to point coordinates.

    Raises:
        ValueError: If codimension != 1 (mean curvature requires normals)

    Example:
        >>> laplacian_coords = compute_laplacian_at_points(mesh)
        >>> # Use for mean curvature: H = ||laplacian_coords|| / (2 * voronoi_area)
    """
    ### Validate codimension
    if mesh.codimension != 1:
        raise ValueError(
            f"Cotangent Laplacian for mean curvature requires codimension-1 manifolds.\n"
            f"Got {mesh.n_manifold_dims=} and {mesh.n_spatial_dims=}, {mesh.codimension=}.\n"
            f"Mean curvature is only defined for hypersurfaces (codimension-1)."
        )

    device = mesh.points.device
    n_points = mesh.n_points

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return torch.zeros(
            (n_points, mesh.n_spatial_dims),
            dtype=mesh.points.dtype,
            device=device,
        )

    ### Extract unique edges
    from physicsnemo.mesh.subdivision._topology import extract_unique_edges

    unique_edges, _ = extract_unique_edges(mesh)  # (n_edges, 2)

    ### Compute cotangent weights for each edge
    cotangent_weights = compute_cotangent_weights(mesh, unique_edges)  # (n_edges,)

    ### Apply cotangent Laplacian operator to point coordinates using shared utility
    from physicsnemo.mesh.calculus.laplacian import _apply_cotan_laplacian_operator

    laplacian_coords = _apply_cotan_laplacian_operator(
        n_vertices=n_points,
        edges=unique_edges,
        cotan_weights=cotangent_weights,
        data=mesh.points,
        device=device,
    )

    return laplacian_coords


def compute_cotangent_weights(mesh: "Mesh", edges: torch.Tensor) -> torch.Tensor:
    """Compute cotangent weights for edges in the mesh.

    For 2D manifolds (triangles):
        w_ij = (1/2) × (cot α + cot β)
    where α, β are opposite angles in the two adjacent triangles.

    For 3D manifolds (tets):
        w_ij = (1/2) × (cot θ_1 + cot θ_2 + ...)
    where θ_k are dihedral angles at the edge in adjacent tets.

    For boundary edges (only one adjacent cell):
        w_ij = (1/2) × cot α
    where α is the angle in the single adjacent triangle.

    Args:
        mesh: Input mesh
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
        Tensor of shape (n_edges,) containing cotangent weights

    Example:
        >>> weights = compute_cotangent_weights(mesh, edges)
        >>> # Use in Laplacian: L_ij = w_ij if connected, else 0
    """
    from physicsnemo.mesh.calculus._circumcentric_dual import (
        compute_cotan_weights_triangle_mesh,
    )

    # Use the merged implementation (now uses correct formula by default)
    return compute_cotan_weights_triangle_mesh(
        mesh,
        edges=edges,
        return_edges=False,
    )
