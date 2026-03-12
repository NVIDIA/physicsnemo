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

"""Mean curvature computation for simplicial meshes.

Implements extrinsic mean curvature using the cotangent Laplace-Beltrami operator.
Only works for codimension-1 manifolds (surfaces with well-defined normals).

For 2D surfaces: H = (k1 + k2) / 2 where k1, k2 are principal curvatures
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.curvature._laplacian import compute_laplacian_at_points
from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0
from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def mean_curvature_vertices(
    mesh: "Mesh",
    include_boundary: bool = False,
) -> torch.Tensor:
    """Compute extrinsic mean curvature at mesh vertices.

    Uses the cotangent Laplace-Beltrami operator:
        H_vertex = (1/2) * ||L @ points|| / voronoi_area

    where L is the cotangent Laplacian. The Laplacian of the embedding coordinates
    gives the mean curvature normal vector, whose magnitude is the mean curvature.

    Mean curvature is an extrinsic measure (depends on embedding in ambient space)
    and is only defined for codimension-1 manifolds where normals exist.

    Signed curvature:
    - Sign determined by normal orientation
    - Positive: Convex (outward bulging like sphere exterior)
    - Negative: Concave (inward curving like sphere interior)
    - Zero: Minimal surface (soap film)

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must be codimension-1)
    include_boundary : bool, optional
        If False, boundary vertices are set to NaN (default).
        If True, computes curvature at boundary vertices using available
        neighbors (Neumann-like boundary condition). This may be less accurate
        at boundaries but provides complete coverage.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_points,) containing signed mean curvature at each vertex.
        For isolated vertices, mean curvature is NaN.
        For boundary vertices, NaN if include_boundary=False, otherwise computed.

    Raises
    ------
    ValueError
        If mesh is not codimension-1

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Sphere of radius r has H = 1/r everywhere
        >>> sphere = sphere_icosahedral.load(radius=2.0, subdivisions=3)
        >>> H = mean_curvature_vertices(sphere)
        >>> # H.mean() ≈ 0.5 (= 1/2.0)

    Notes
    -----
    For a sphere with outward normals, H > 0.
    For minimal surfaces (soap films), H = 0.
    """
    ### Validate codimension (done in compute_laplacian_at_points)

    ### Compute Laplacian applied to points
    laplacian_coords = compute_laplacian_at_points(mesh)  # (n_points, n_spatial_dims)

    ### Compute magnitude of mean curvature normal
    # ||L @ points|| gives 2 * H * voronoi_area
    laplacian_magnitude = torch.norm(laplacian_coords, dim=-1)  # (n_points,)

    ### Compute dual volumes (Voronoi areas)
    dual_volumes = compute_dual_volumes_0(mesh)  # (n_points,)

    ### Compute mean curvature
    # H = ||L @ points|| / (2 * dual_volume)
    dual_volumes_safe = torch.clamp(dual_volumes, min=safe_eps(dual_volumes.dtype))
    mean_curvature = laplacian_magnitude / (2.0 * dual_volumes_safe)

    ### Determine sign using normal orientation
    # The mean curvature normal is: H * n = (1/2) * L @ points
    # For a sphere with outward normals, L @ points points INWARD (toward center)
    # But we want H > 0 for convex surfaces, so:
    # sign = -sign(L · n) to flip the convention

    point_normals = mesh.point_normals  # (n_points, n_spatial_dims)

    # Normalize laplacian_coords first
    laplacian_normalized = torch.nn.functional.normalize(
        laplacian_coords, dim=-1, eps=1e-12
    )

    # Sign from dot product (NEGATIVE of dot product for correct convention)
    # Positive curvature when Laplacian opposes normal (convex like sphere)
    sign = -torch.sign((laplacian_normalized * point_normals).sum(dim=-1))

    # Handle zero magnitude case (flat regions)
    sign = torch.where(
        laplacian_magnitude > 1e-10,
        sign,
        torch.ones_like(sign),  # Default to positive for zero curvature
    )

    # Apply sign
    mean_curvature = mean_curvature * sign

    ### Set isolated vertices to NaN
    mean_curvature = torch.where(
        dual_volumes > 0,
        mean_curvature,
        torch.full_like(mean_curvature, float("nan")),
    )

    ### Handle boundary vertices
    # The cotangent Laplacian formula assumes a complete neighborhood around each vertex.
    # For boundary vertices, we can either:
    # 1. Set to NaN (conservative, default)
    # 2. Compute using available neighbors (Neumann-like boundary condition)

    if not include_boundary:
        from physicsnemo.mesh.boundaries import get_boundary_vertices

        is_boundary_vertex = get_boundary_vertices(mesh)

        # Set boundary vertices to NaN
        mean_curvature = torch.where(
            is_boundary_vertex,
            torch.full_like(mean_curvature, float("nan")),
            mean_curvature,
        )

    return mean_curvature
