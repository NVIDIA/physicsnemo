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
    device = mesh.points.device
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
    dual_volumes_safe = torch.clamp(dual_volumes, min=1e-30)

    gaussian_curvature = angle_defect / dual_volumes_safe

    # Set isolated vertices (zero dual volume) to NaN
    gaussian_curvature = torch.where(
        dual_volumes > 0,
        gaussian_curvature,
        torch.full_like(gaussian_curvature, float("nan")),
    )

    return gaussian_curvature


def gaussian_curvature_cells(mesh: "Mesh") -> torch.Tensor:
    """Compute Gaussian curvature at cell centers using dual mesh concept.

    Treats cell centroids as vertices of a dual mesh and computes curvature
    based on angles between connections to adjacent cell centroids.

    This provides a cell-based curvature measure complementary to vertex curvature.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_cells,) containing Gaussian curvature at each cell.

    Algorithm
    ---------
    1. Get cell-to-cell adjacency (cells sharing facets)
    2. Compute "dual angles" between adjacent cell centroids
    3. Apply angle defect formula on dual mesh

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> sphere_mesh = sphere_icosahedral.load(subdivisions=2)
        >>> K_cells = gaussian_curvature_cells(sphere_mesh)
        >>> # Should be positive for sphere
    """
    device = mesh.points.device
    n_cells = mesh.n_cells
    n_manifold_dims = mesh.n_manifold_dims

    ### Handle empty mesh
    if n_cells == 0:
        return torch.zeros(0, dtype=mesh.points.dtype, device=device)

    ### Get cell centroids (reuse existing computation)
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    ### Get cell-to-cell adjacency
    from physicsnemo.mesh.neighbors import get_cell_to_cells_adjacency

    # Cells are adjacent if they share a codimension-1 facet
    adjacency = get_cell_to_cells_adjacency(mesh, adjacency_codimension=1)

    ### Compute angles in dual mesh (fully vectorized)
    # For each cell, sum angles between all pairs of vectors to adjacent cell centroids
    angle_sums = torch.zeros(n_cells, dtype=mesh.points.dtype, device=device)

    ### Get valences (number of neighbors per cell)
    valences = adjacency.offsets[1:] - adjacency.offsets[:-1]

    ### Build source cell indices for each neighbor relationship
    # Shape: (total_neighbors,)
    source_cell_indices = torch.repeat_interleave(
        torch.arange(n_cells, dtype=torch.int64, device=device),
        valences,
    )

    ### Get vectors from each cell to each of its neighbors
    # adjacency.indices contains the neighbor cell indices
    # source_cell_indices contains the source cell index for each entry in adjacency.indices
    # Shape: (total_neighbors, n_spatial_dims)
    source_centroids = cell_centroids[source_cell_indices]
    neighbor_centroids = cell_centroids[adjacency.indices]
    vectors = neighbor_centroids - source_centroids

    ### For each cell, compute pairwise angles between all neighbor vectors
    # We need to process cells with different numbers of neighbors
    # For efficiency, batch cells by valence
    unique_valences = torch.unique(valences[valences >= 2])

    for val in unique_valences:
        ### Get cells with this valence
        cells_with_valence = torch.where(valences == val)[0]
        n_cells_val = len(cells_with_valence)

        if n_cells_val == 0:
            continue

        ### Extract vectors for these cells using vectorized indexing
        # Shape: (n_cells_val, val, n_spatial_dims)

        # Build gather indices vectorized: broadcast offsets with arange
        # Shape: (n_cells_val, val)
        start_indices = adjacency.offsets[cells_with_valence]  # (n_cells_val,)
        offset_range = torch.arange(val, device=device)  # (val,)
        gather_indices = start_indices.unsqueeze(1) + offset_range.unsqueeze(
            0
        )  # (n_cells_val, val)

        # Gather vectors
        # Shape: (n_cells_val, val, n_spatial_dims)
        cell_vectors = vectors[gather_indices.flatten()].reshape(
            n_cells_val, val, mesh.n_spatial_dims
        )

        ### Generate all pairwise combinations (i, j) where i < j (vectorized)
        # For val neighbors, we have C(val, 2) = val*(val-1)/2 pairs
        n_pairs = (val * (val - 1)) // 2

        # Vectorized pair generation
        # Create indices 0 to val-1 and get all pairs where i < j
        val_int = int(val)
        indices = torch.arange(val_int, device=device)
        # Generate pairs using broadcasting trick
        i_idx = indices.unsqueeze(1).expand(val_int, val_int)
        j_idx = indices.unsqueeze(0).expand(val_int, val_int)
        # Get upper triangle (i < j)
        mask = i_idx < j_idx
        pair_i = i_idx[mask]
        pair_j = j_idx[mask]

        ### Compute angles for all pairs across all cells
        # Shape: (n_cells_val, n_pairs, n_spatial_dims)
        vectors_i = cell_vectors[:, pair_i, :]  # (n_cells_val, n_pairs, n_spatial_dims)
        vectors_j = cell_vectors[:, pair_j, :]  # (n_cells_val, n_pairs, n_spatial_dims)

        ### Compute angles using stable_angle_between_vectors
        # Reshape to (n_cells_val * n_pairs, n_spatial_dims) for batch computation
        from physicsnemo.mesh.curvature._utils import stable_angle_between_vectors

        vectors_i_flat = vectors_i.reshape(-1, mesh.n_spatial_dims)
        vectors_j_flat = vectors_j.reshape(-1, mesh.n_spatial_dims)

        angles_flat = stable_angle_between_vectors(vectors_i_flat, vectors_j_flat)
        angles = angles_flat.reshape(n_cells_val, n_pairs)

        ### Sum angles for each cell
        angle_sums[cells_with_valence] = angles.sum(dim=1)

    ### Compute angle defect
    full_angle = compute_full_angle_n_sphere(n_manifold_dims)
    angle_defect = full_angle - angle_sums

    ### Approximate "dual Voronoi area" using cell area
    # For dual mesh, use cell area as approximate measure
    cell_areas = mesh.cell_areas

    ### Compute curvature
    gaussian_curvature = angle_defect / torch.clamp(cell_areas, min=1e-30)

    # Set isolated cells to NaN
    gaussian_curvature = torch.where(
        cell_areas > 0,
        gaussian_curvature,
        torch.full_like(gaussian_curvature, float("nan")),
    )

    return gaussian_curvature
