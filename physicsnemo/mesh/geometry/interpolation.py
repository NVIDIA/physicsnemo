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

r"""Barycentric interpolation functions and their gradients for DEC.

Barycentric (or Whitney 0-form) interpolation functions
:math:`\varphi_{v, \sigma}` are the standard linear shape functions used in
finite elements. For a simplex :math:`\sigma` with vertices
:math:`v_0, \ldots, v_n`, the function :math:`\varphi_v` is 1 at vertex
:math:`v` and 0 at all other vertices of :math:`\sigma`, linearly
interpolated.

The gradients of these functions are needed for the discrete sharp operator
in DEC.

Key properties (Hirani 2003, *Discrete Exterior Calculus*, Remark 2.7.2):

- :math:`\nabla \varphi_{v, \sigma}` is constant in the interior of
  :math:`\sigma`.
- :math:`\nabla \varphi_{v, \sigma}` is perpendicular to the face of
  :math:`\sigma` opposite to :math:`v`.
- :math:`\|\nabla \varphi_{v, \sigma}\| = 1 / h`, where :math:`h` is the
  height of :math:`v` above the opposite face.
- :math:`\sum_{v \in \sigma} \nabla \varphi_{v, \sigma} = 0` (gradients sum
  to zero).

References
----------
Hirani, A. N. (2003). *Discrete Exterior Calculus*. PhD thesis, California
Institute of Technology. §2.7 (Interpolation Functions), Remark 2.7.2.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_barycentric_gradients(
    mesh: "Mesh",
) -> Float[torch.Tensor, "n_cells n_vertices_per_cell n_spatial_dims"]:
    r"""Compute gradients of barycentric interpolation functions.

    For each cell :math:`\sigma` and each of its vertices :math:`v`, computes
    :math:`\nabla \varphi_{v, \sigma}`, the gradient of the barycentric
    interpolation function that is 1 at :math:`v` and 0 at all other vertices
    of :math:`\sigma`.

    These gradients are needed for the primal-primal sharp (PP-sharp)
    operator (Hirani 2003, *Discrete Exterior Calculus*, Eq. 5.8.1).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh (2D or 3D).

    Returns
    -------
    Float[torch.Tensor, "n_cells n_vertices_per_cell n_spatial_dims"]
        Per-vertex barycentric gradients. ``gradients[i, j, :]`` is
        :math:`\nabla \varphi_{v_j,\, \sigma_i}`, where :math:`v_j` is the
        :math:`j`-th vertex of cell :math:`\sigma_i` in local indexing.

    Notes
    -----
    For an :math:`n`-simplex :math:`\sigma` with vertices
    :math:`v_0, \ldots, v_n`:

    1. :math:`\nabla \varphi_{v_0, \sigma}` is perpendicular to the face
       :math:`[v_1, \ldots, v_n]` opposite :math:`v_0`.
    2. It points from the face centroid toward :math:`v_0`.
    3. It has magnitude :math:`1 / h`, where :math:`h` is the height of
       :math:`v_0` above the opposite face.

    Efficient computation uses barycentric coordinate derivatives: for
    vertex :math:`v_i`, :math:`\nabla \varphi_{v_i, \sigma}` equals the
    inward unit normal to the opposite face, divided by :math:`h_i`.

    Properties:

    - :math:`\sum_i \nabla \varphi_{v_i, \sigma} = 0`
      (barycentric coords sum to 1).
    - :math:`\nabla \varphi_{v_i, \sigma} \cdot (v_j - v_i) = -1`
      for :math:`j \neq i`.
    - :math:`\nabla \varphi_{v_i, \sigma} \cdot (v_i - v_j) = +1`
      for :math:`j \neq i`.

    References
    ----------
    Hirani (2003), *Discrete Exterior Calculus* (PhD thesis), Remark 2.7.2.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> grads = compute_barycentric_gradients(mesh)
    >>> # grads[i, j, :] is grad phi for j-th vertex of i-th cell
    >>> # Use in sharp operator with sharp(alpha)(v) = sum alpha(edge) * weight * grad
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
