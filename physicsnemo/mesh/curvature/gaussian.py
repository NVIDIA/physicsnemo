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

"""Gaussian curvature computation for simplicial meshes.

Implements intrinsic Gaussian curvature using angle defect method.
Works for any codimension (intrinsic property).

For 2D surfaces: K = k1 * k2 where k1, k2 are principal curvatures
For 1D curves: K represents discrete turning angle
For 3D volumes: K represents volumetric angle defect

Reference: Meyer et al. (2003), Discrete Gauss-Bonnet theorem
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.curvature._angles import compute_angles_at_vertices
from physicsnemo.mesh.curvature._utils import compute_full_angle_n_sphere
from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0
from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def gaussian_curvature_vertices(mesh: "Mesh") -> torch.Tensor:
    """Compute intrinsic Gaussian curvature at mesh vertices.

    Uses the angle defect formula from discrete differential geometry:
        K_vertex = angle_defect / voronoi_area
    where:
        angle_defect = full_angle(n) - Σ(angles at vertex in incident cells)

    This is an intrinsic measure of curvature that works for any codimension,
    as it depends only on distances measured within the manifold (Theorema Egregium).

    Signed curvature:
    - Positive: Elliptic point (sphere-like)
    - Zero: Flat/parabolic point (plane-like)
    - Negative: Hyperbolic point (saddle-like)

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh (1D, 2D, or 3D manifold)

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_points,) containing signed Gaussian curvature at each vertex.
        For isolated vertices (no incident cells), curvature is NaN.

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Sphere of radius r has K = 1/r² everywhere
        >>> sphere_mesh = sphere_icosahedral.load(radius=2.0, subdivisions=3)
        >>> K = gaussian_curvature_vertices(sphere_mesh)
        >>> # K.mean() ≈ 0.25 (= 1/(2.0)²)

    Notes
    -----
    Satisfies discrete Gauss-Bonnet theorem:
        Σ_vertices (K_i * A_i) = 2π * χ(M)
    where χ(M) is the Euler characteristic.
    """
    n_manifold_dims = mesh.n_manifold_dims

    ### Compute angle sums at each vertex
    angle_sums = compute_angles_at_vertices(mesh)  # (n_points,)

    ### Compute full angle for this manifold dimension
    full_angle = compute_full_angle_n_sphere(n_manifold_dims)

    ### Compute angle defect
    # angle_defect = full_angle - sum_of_angles
    # Positive defect = positive curvature
    angle_defect = full_angle - angle_sums  # (n_points,)

    ### Compute dual volumes (Voronoi areas)
    dual_volumes = compute_dual_volumes_0(mesh)  # (n_points,)

    ### Compute Gaussian curvature
    # K = angle_defect / dual_volume
    # For isolated vertices (dual_volume = 0), this gives inf/nan
    # Clamp areas to avoid division by zero, use inf for zero areas
    dual_volumes_safe = torch.clamp(dual_volumes, min=safe_eps(dual_volumes.dtype))

    gaussian_curvature = angle_defect / dual_volumes_safe

    # Set isolated vertices (zero dual volume) to NaN
    gaussian_curvature = torch.where(
        dual_volumes > 0,
        gaussian_curvature,
        torch.full_like(gaussian_curvature, float("nan")),
    )

    return gaussian_curvature


def gaussian_curvature_cells(mesh: "Mesh") -> torch.Tensor:
    """Compute Gaussian curvature at cell centers.

    Averages the intrinsic vertex-based Gaussian curvature (angle defect)
    over each cell's vertices. This gives a cell-centered curvature field
    that is consistent with the vertex-based curvature and inherits its
    mathematical properties (intrinsic, satisfies discrete Gauss-Bonnet).

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_cells,) containing Gaussian curvature at each cell.
        Cells whose vertices all have NaN curvature are set to NaN.

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> sphere_mesh = sphere_icosahedral.load(subdivisions=2)
        >>> K_cells = gaussian_curvature_cells(sphere_mesh)
        >>> # Should be positive for sphere
    """
    device = mesh.points.device
    n_cells = mesh.n_cells

    ### Handle empty mesh
    if n_cells == 0:
        return torch.zeros(0, dtype=mesh.points.dtype, device=device)

    ### Compute vertex Gaussian curvature (intrinsic, angle-defect based)
    K_vertices = gaussian_curvature_vertices(mesh)  # (n_points,)

    ### Average vertex curvature to cell centers
    # For each cell, take the mean of its vertices' curvatures, ignoring NaN.
    cell_vertex_K = K_vertices[mesh.cells]  # (n_cells, n_verts_per_cell)

    # Mask NaN vertices (isolated vertices with no incident cells)
    valid = ~torch.isnan(cell_vertex_K)  # (n_cells, n_verts_per_cell)
    n_valid = valid.sum(dim=1)  # (n_cells,)

    # Replace NaN with 0 for summation, then divide by count of valid vertices
    cell_vertex_K_safe = torch.where(
        valid, cell_vertex_K, torch.zeros_like(cell_vertex_K)
    )
    K_sum = cell_vertex_K_safe.sum(dim=1)  # (n_cells,)
    K_cells = K_sum / n_valid.clamp(min=1).to(K_sum.dtype)

    # Set cells with no valid vertices to NaN
    K_cells = torch.where(
        n_valid > 0,
        K_cells,
        torch.full_like(K_cells, float("nan")),
    )

    ### Set boundary cells to NaN.
    # Boundary vertices have incomplete angular neighborhoods, giving
    # spuriously large curvature values. Cells that touch the boundary
    # inherit those values and should be excluded.
    from physicsnemo.mesh.boundaries._detection import get_boundary_vertices

    is_boundary_vertex = get_boundary_vertices(mesh)  # (n_points,)
    cell_touches_boundary = is_boundary_vertex[mesh.cells].any(dim=1)  # (n_cells,)
    K_cells = torch.where(
        cell_touches_boundary,
        torch.full_like(K_cells, float("nan")),
        K_cells,
    )

    return K_cells
