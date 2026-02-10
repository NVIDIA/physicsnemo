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

"""Interior angle computation for simplicial meshes.

Computes generalized interior angles at vertices of n-simplices using a
dimension-agnostic formula based on correlation (normalized Gram) matrices.
The formula unifies planar angles (triangles), solid angles (tetrahedra),
and higher-dimensional generalizations into a single expression.

This module provides two levels of abstraction:

- :func:`compute_vertex_angles`: Per-cell-per-vertex angles, shape
  ``(n_cells, n_vertices_per_cell)``. This is the fundamental geometric
  primitive, used by normal weighting and quality metrics.

- :func:`compute_vertex_angle_sums`: Per-vertex angle sums, shape
  ``(n_points,)``. This aggregates angles across all incident cells,
  used by Gaussian curvature (angle defect method).

Reference:
    Van Oosterom, A. & Strackee, J. (1983). "The Solid Angle of a Plane
    Triangle." IEEE Trans. Biomed. Eng. BME-30(2):125-126.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_vertex_angles(mesh: "Mesh") -> torch.Tensor:
    """Compute generalized interior angles at each vertex of each cell.

    For an n-simplex, the "angle" at a vertex is computed using the unified
    formula that generalizes to arbitrary dimensions:

        Omega = 2 * arctan(sqrt(det(C)) / (1 + sum_{i<j} C_ij))

    where C is the correlation (normalized Gram) matrix of edge vectors::

        C_ij = (e_i . e_j) / (|e_i| |e_j|)

    This formula reduces to:

    - For triangles (n=2): the planar interior angle theta
    - For tetrahedra (n=3): the solid angle Omega (steradians)
    - For higher n: the generalized solid angle

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh. Must have at least 1-dimensional cells
        (edges or higher).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_cells, n_vertices_per_cell)`` containing the
        generalized angle at each vertex of each cell.

    Notes
    -----
    This formula is derived by recognizing that both the planar angle formula
    and the Van Oosterom & Strackee (1983) solid angle formula follow the
    same pattern when expressed in terms of the correlation matrix.

    The formula uses ``atan2`` for numerical stability when the denominator
    approaches zero (nearly degenerate simplices). Intermediate computations
    are performed in float64 to avoid catastrophic cancellation in the
    correlation matrix when vertex angles approach 0 or pi.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Equilateral triangle: all angles should be pi/3
    >>> pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866025]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh = Mesh(points=pts, cells=cells)
    >>> angles = compute_vertex_angles(mesh)
    >>> angles.shape
    torch.Size([1, 3])
    """
    n_edges = mesh.n_manifold_dims  # edges emanating from each vertex
    input_dtype = mesh.points.dtype

    ### Upcast to float64 for the correlation matrix and determinant
    # The correlation matrix C_ij = cos(angle_ij) suffers catastrophic
    # cancellation in float32 when angles are near 0 or pi, because
    # cos(epsilon) = 1 - epsilon^2/2 is indistinguishable from 1.0.
    # Computing in float64 avoids this with negligible overhead.
    cell_vertices = mesh.points[mesh.cells].double()

    ### Build edge vectors for all vertices simultaneously
    # For each vertex k, edges[k, i] = v_{(k+i+1) mod n_verts} - v_k
    # Shape: (n_cells, n_verts, n_edges, n_spatial_dims)
    edges = torch.stack(
        [
            torch.roll(cell_vertices, shifts=-(i + 1), dims=1) - cell_vertices
            for i in range(n_edges)
        ],
        dim=2,
    )

    ### Compute edge lengths: (n_cells, n_verts, n_edges)
    edge_lengths = edges.norm(dim=-1)

    ### Compute normalized edges: (n_cells, n_verts, n_edges, n_spatial_dims)
    edges_normalized = edges / edge_lengths.unsqueeze(-1).clamp(
        min=safe_eps(torch.float64)
    )

    ### Compute correlation matrix C for each vertex of each cell
    # C[i,j] = normalized_edge_i . normalized_edge_j
    # Shape: (n_cells, n_verts, n_edges, n_edges)
    corr_matrix = torch.einsum("cvid,cvjd->cvij", edges_normalized, edges_normalized)

    ### Compute det(C) for each vertex: (n_cells, n_verts)
    det_C = torch.linalg.det(corr_matrix)

    ### Compute sum of upper-triangle off-diagonal elements: sum_{i<j} C_ij
    triu_mask = torch.triu(
        torch.ones(n_edges, n_edges, device=mesh.points.device, dtype=torch.bool),
        diagonal=1,
    )
    sum_off_diag = corr_matrix[:, :, triu_mask].sum(dim=-1)  # (n_cells, n_verts)

    ### Compute angle: Omega = 2 * arctan2(sqrt(|det(C)|), 1 + sum_{i<j} C_ij)
    denominator = 1.0 + sum_off_diag
    numerator = det_C.abs().sqrt()
    angles = 2.0 * torch.atan2(numerator, denominator)

    return angles.to(input_dtype)


def compute_vertex_angle_sums(mesh: "Mesh") -> torch.Tensor:
    """Compute the sum of interior angles at each vertex over all incident cells.

    For each vertex, sums the generalized interior angle contributed by every
    cell incident to that vertex. This is the quantity used in the angle defect
    formula for discrete Gaussian curvature:

        K_v = (full_angle - angle_sum_v) / voronoi_area_v

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_points,)`` containing the summed angle at each
        vertex. Isolated vertices (no incident cells) have angle sum 0.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Two triangles sharing an edge: interior vertex has angle sum = 2*pi/3 + 2*pi/3
    >>> pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866025], [0.5, -0.866025]])
    >>> cells = torch.tensor([[0, 1, 2], [0, 3, 1]])
    >>> mesh = Mesh(points=pts, cells=cells)
    >>> sums = compute_vertex_angle_sums(mesh)
    >>> sums.shape
    torch.Size([4])
    """
    ### Compute per-cell-per-vertex angles
    angles = compute_vertex_angles(mesh)  # (n_cells, n_verts_per_cell)

    ### Scatter-add angles to their corresponding vertex indices
    angle_sums = torch.zeros(
        mesh.n_points, dtype=mesh.points.dtype, device=mesh.points.device
    )
    angle_sums.scatter_add_(0, mesh.cells.reshape(-1), angles.reshape(-1))

    return angle_sums
