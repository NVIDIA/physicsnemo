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

"""Barycentric interpolation functions and their gradients for DEC.

Barycentric (or Whitney 0-form) interpolation functions φ_{v,cell} are the standard
linear shape functions used in finite elements. For a simplex with vertices v₀,...,vₙ,
the function φ_v is 1 at vertex v and 0 at all other vertices, linearly interpolated.

The gradients of these functions are needed for the discrete sharp operator in DEC.

Key properties (Hirani Rem. 2.7.2, lines 1260-1288):
- ∇φ_{v,cell} is constant in the cell interior
- ∇φ_{v,cell} is perpendicular to the face opposite to v
- ||∇φ_{v,cell}|| = 1/h where h is the height of v above opposite face
- Σ_{vertices v in cell} ∇φ_{v,cell} = 0 (gradients sum to zero)

References:
    Hirani (2003) Section 2.7, Remark 2.7.2
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_barycentric_gradients(
    mesh: "Mesh",
) -> torch.Tensor:
    """Compute gradients of barycentric interpolation functions.

    For each cell and each of its vertices, computes ∇φ_{v,cell}, the gradient
    of the barycentric interpolation function that is 1 at vertex v and 0 at
    all other vertices of the cell.

    These gradients are needed for the PP-sharp operator (Hirani Eq. 5.8.1).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh (2D or 3D)

    Returns
    -------
    torch.Tensor
        Gradients of shape (n_cells, n_vertices_per_cell, n_spatial_dims)

        gradients[cell_i, local_vertex_j, :] = ∇φ_{v_j, cell_i}

        where v_j is the j-th vertex of cell_i (in local indexing).

    Algorithm:
        For n-simplex with vertices v₀, ..., vₙ:
        1. The gradient ∇φ_{v₀,cell} is perpendicular to face [v₁,...,vₙ]
        2. Points from face centroid toward v₀
        3. Has magnitude 1/height

        Efficient computation:
        - Use barycentric coordinate derivatives
        - For vertex i: ∇φᵢ = ∇(volume ratio) = normal to opposite face / height

    Properties:
        - Σᵢ ∇φᵢ = 0 (constraint: barycentric coords sum to 1)
        - ∇φᵢ · (vⱼ - vᵢ) = -1 for j ≠ i (decrease along edge away from i)
        - ∇φᵢ · (vᵢ - vⱼ) = +1 for j ≠ i (increase along edge toward i)

    Reference:
        Hirani Remark 2.7.2 (lines 1260-1288)

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> grads = compute_barycentric_gradients(mesh)
        >>> # grads[i, j, :] is ∇φ for j-th vertex of i-th cell
        >>> # Use in sharp operator with α♯(v) = Σ α(edge) × weight × grad
    """
    n_cells = mesh.n_cells
    n_manifold_dims = mesh.n_manifold_dims
    n_spatial_dims = mesh.n_spatial_dims
    n_vertices_per_cell = n_manifold_dims + 1

    device = mesh.points.device
    dtype = mesh.points.dtype

    ### Initialize output
    gradients = torch.zeros(
        (n_cells, n_vertices_per_cell, n_spatial_dims),
        dtype=dtype,
        device=device,
    )

    ### Handle empty mesh
    if n_cells == 0:
        return gradients

    ### Get cell vertices
    cell_vertices = mesh.points[
        mesh.cells
    ]  # (n_cells, n_vertices_per_cell, n_spatial_dims)

    if n_manifold_dims == 2:
        ### 2D triangles: Efficient closed-form solution
        # For triangle with vertices v₀, v₁, v₂:
        # ∇φ₀ is perpendicular to edge [v₁, v₂] and points toward v₀
        #
        # Standard formula from finite elements:
        # For 2D triangle, ∇φᵢ = perpendicular to opposite edge / (2 × area)
        #
        # More precisely: ∇φ₀ = (v₂ - v₁)^⊥ / (2 × signed_area)
        # where ^⊥ rotates 90° counterclockwise in 2D

        ### Extract vertices
        v0 = cell_vertices[:, 0, :]  # (n_cells, n_spatial_dims)
        v1 = cell_vertices[:, 1, :]
        v2 = cell_vertices[:, 2, :]

        ### Compute 2× signed area for each triangle
        # Using cross product: 2A = (v1-v0) × (v2-v0)
        edge1 = v1 - v0
        edge2 = v2 - v0

        if n_spatial_dims == 2:
            # 2D: cross product gives z-component (scalar)
            twice_signed_area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
            twice_signed_area = twice_signed_area.unsqueeze(-1)  # (n_cells, 1)
        elif n_spatial_dims == 3:
            # 3D: cross product encodes both normal direction and twice-area
            cross = torch.linalg.cross(edge1, edge2)  # (n_cells, 3)
        else:
            # Higher dimensions: use Gram determinant
            raise NotImplementedError(
                f"Barycentric gradients for n_spatial_dims={n_spatial_dims} not yet implemented"
            )

        ### Compute gradients of barycentric functions for each vertex
        # In 2D: ∇φᵢ = perpendicular(opposite_edge) / (2 × signed_area)
        #   where perpendicular(x, y) = (-y, x) is a fixed 90° CCW rotation,
        #   and the signed area corrects the direction for CW-oriented cells.
        # In 3D: ∇φᵢ = cross × opposite_edge / |cross|²
        #   where cross = (v₁-v₀) × (v₂-v₀). This formula is inherently
        #   orientation-independent (no signed area needed) because flipping
        #   two vertices negates both cross and the opposite edge, leaving
        #   the quotient unchanged.

        if n_spatial_dims == 2:
            ### 2D case: direct perpendicular
            edge_v2_v1 = v2 - v1  # (n_cells, 2)
            edge_v0_v2 = v0 - v2
            edge_v1_v0 = v1 - v0

            # Perpendicular: (x,y) → (-y, x)
            perp_v2_v1 = torch.stack([-edge_v2_v1[:, 1], edge_v2_v1[:, 0]], dim=1)
            perp_v0_v2 = torch.stack([-edge_v0_v2[:, 1], edge_v0_v2[:, 0]], dim=1)
            perp_v1_v0 = torch.stack([-edge_v1_v0[:, 1], edge_v1_v0[:, 0]], dim=1)

            gradients[:, 0, :] = perp_v2_v1 / twice_signed_area
            gradients[:, 1, :] = perp_v0_v2 / twice_signed_area
            gradients[:, 2, :] = perp_v1_v0 / twice_signed_area

        elif n_spatial_dims == 3:
            ### 3D case: ∇φᵢ = (cross × opposite_edge) / |cross|²
            # Equivalent to the textbook n̂ × edge / (2A), since
            # n̂ = cross/|cross| and 2A = |cross|, giving cross×edge / |cross|².

            # Opposite edges
            edge_v2_v1 = v2 - v1
            edge_v0_v2 = v0 - v2
            edge_v1_v0 = v1 - v0

            # |cross|² = (2A)²; clamp for degenerate triangles (zero area)
            cross_norm_sq = (
                (cross * cross)
                .sum(dim=-1, keepdim=True)
                .clamp(min=safe_eps(cross.dtype))
            )

            gradients[:, 0, :] = torch.linalg.cross(cross, edge_v2_v1) / cross_norm_sq
            gradients[:, 1, :] = torch.linalg.cross(cross, edge_v0_v2) / cross_norm_sq
            gradients[:, 2, :] = torch.linalg.cross(cross, edge_v1_v0) / cross_norm_sq

    elif n_manifold_dims == 3:
        ### 3D tetrahedra: Use dual basis / perpendicular to opposite face
        # ∇φᵢ is perpendicular to the triangular face opposite to vertex i
        # and has magnitude 1/(height from i to opposite face)

        ### For each vertex, compute gradient
        for local_v_idx in range(4):
            ### Get opposite face (3 vertices excluding current one)
            other_indices = [j for j in range(4) if j != local_v_idx]
            opposite_face_vertices = cell_vertices[
                :, other_indices, :
            ]  # (n_cells, 3, n_spatial_dims)

            ### Compute normal to opposite face
            # Face has 3 vertices: compute normal via cross product
            face_v0 = opposite_face_vertices[:, 0, :]
            face_v1 = opposite_face_vertices[:, 1, :]
            face_v2 = opposite_face_vertices[:, 2, :]

            face_edge1 = face_v1 - face_v0
            face_edge2 = face_v2 - face_v0

            face_normal = torch.linalg.cross(face_edge1, face_edge2)  # (n_cells, 3)
            face_area = (
                torch.norm(face_normal, dim=-1, keepdim=True) / 2.0
            )  # (n_cells, 1)

            ### Normalize face normal
            face_normal_unit = face_normal / (2.0 * face_area).clamp(
                min=safe_eps(face_area.dtype)
            )

            ### Height from vertex to opposite face
            vertex_pos = cell_vertices[:, local_v_idx, :]
            vec_to_face = face_v0 - vertex_pos
            height = torch.abs(
                (vec_to_face * face_normal_unit).sum(dim=-1, keepdim=True)
            )  # (n_cells, 1)

            ### Gradient: normal direction with magnitude 1/height
            # Direction: toward vertex (opposite of normal if on other side)
            sign = torch.sign(
                (vec_to_face * face_normal_unit).sum(dim=-1, keepdim=True)
            )
            grad = -sign * face_normal_unit / height.clamp(min=safe_eps(height.dtype))

            gradients[:, local_v_idx, :] = grad.squeeze(-1)

    else:
        raise NotImplementedError(
            f"Barycentric gradients not implemented for {n_manifold_dims=}D. "
            f"Currently supported: 2D (triangles), 3D (tetrahedra)."
        )

    return gradients
