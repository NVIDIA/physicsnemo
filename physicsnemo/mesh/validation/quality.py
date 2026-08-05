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

"""Quality metrics for mesh cells.

Computes geometric quality metrics for simplicial cells including aspect ratio,
skewness, and angles. Higher quality = better shaped cells.
"""

import math
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.mesh.geometry._angles import compute_vertex_angles
from physicsnemo.mesh.geometry._cell_areas import compute_cell_areas
from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_cell_edge_lengths(
    mesh: "Mesh",
) -> Float[torch.Tensor, "n_cells n_edges_per_cell"]:
    """Compute all pairwise edge lengths within each cell.

    For an n-simplex with (n+1) vertices, there are C(n+1, 2) edges per cell.
    Returns a tensor of all edge lengths, vectorized across all cells.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose cells to measure.

    Returns
    -------
    torch.Tensor
        Edge lengths, shape ``(n_cells, n_edges_per_cell)`` where
        ``n_edges_per_cell = C(n_manifold_dims + 1, 2)``.
        Returns an empty ``(0, 0)`` tensor if the mesh has no cells.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> lengths = compute_cell_edge_lengths(mesh)
    >>> lengths.shape
    torch.Size([1, 3])
    """
    if mesh.n_cells == 0:
        return torch.zeros((0, 0), dtype=mesh.points.dtype, device=mesh.points.device)

    cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, n_dims)
    n_verts_per_cell = mesh.n_manifold_dims + 1

    # All (i, j) pairs with i < j via upper-triangular indices
    i_indices, j_indices = torch.triu_indices(
        n_verts_per_cell,
        n_verts_per_cell,
        offset=1,
        device=mesh.points.device,
    )
    # Edge vectors and their lengths: (n_cells, n_edges_per_cell)
    edge_vectors = cell_vertices[:, j_indices] - cell_vertices[:, i_indices]
    return torch.linalg.vector_norm(edge_vectors, dim=-1)


def _compute_simplex_altitudes(
    mesh: "Mesh",
    cell_measures: Float[torch.Tensor, " n_cells"],
) -> Float[torch.Tensor, "n_cells n_vertices_per_cell"]:
    """Compute the altitude opposite every simplex vertex."""
    n_manifold_dims = mesh.n_manifold_dims
    n_vertices_per_cell = n_manifold_dims + 1

    if n_manifold_dims == 1:
        # A segment's two opposing facets are points with unit 0-volume.
        return cell_measures[:, None].expand(-1, n_vertices_per_cell)

    cell_vertices = mesh.points[mesh.cells]
    vertex_indices = torch.arange(n_vertices_per_cell, device=mesh.cells.device)
    facet_mask = ~torch.eye(
        n_vertices_per_cell,
        dtype=torch.bool,
        device=mesh.cells.device,
    )
    facet_indices = vertex_indices.expand(n_vertices_per_cell, -1)[facet_mask].reshape(
        n_vertices_per_cell, n_manifold_dims
    )
    facet_vertices = cell_vertices[:, facet_indices]
    facet_relative_vectors = facet_vertices[..., 1:, :] - facet_vertices[..., :1, :]
    facet_measures = compute_cell_areas(facet_relative_vectors.flatten(0, 1)).unflatten(
        0, (mesh.n_cells, n_vertices_per_cell)
    )

    # An absolute epsilon is not dimensionally valid here: facet measures have
    # units of length ** (d - 1), while the common tolerance is length-valued.
    # Guard exact zeros without changing any positive, representable measure.
    facet_denominators = torch.where(
        facet_measures > 0,
        facet_measures,
        torch.ones_like(facet_measures),
    )
    return n_manifold_dims * cell_measures[:, None] / facet_denominators


def compute_quality_metrics(mesh: "Mesh") -> TensorDict:
    """Compute geometric quality metrics for all cells.

    Returns TensorDict with per-cell quality metrics:

    - aspect_ratio: normalized max_edge / min_altitude
      (lower is better, 1.0 is a regular simplex)
    - min_angle: Minimum interior angle in radians
    - max_angle: Maximum interior angle in radians
    - edge_length_ratio: max_edge / min_edge (1.0 is a regular simplex)
    - quality_score: Combined metric in [0,1] (1.0 is a regular simplex)

    Parameters
    ----------
    mesh : Mesh
        Mesh to analyze

    Returns
    -------
    TensorDict
        TensorDict of shape (n_cells,) with quality metrics

    Notes
    -----
    A 0-simplex has no shape to distort, so its aspect ratio, edge-length
    ratio, and quality score are 1. Its edge lengths and angles are undefined
    and reported as ``NaN``.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> metrics = compute_quality_metrics(mesh)
    >>> assert "quality_score" in metrics.keys()
    """
    if mesh.n_cells == 0:
        return TensorDict(
            {},
            batch_size=torch.Size([0]),
            device=mesh.points.device,
        )

    device = mesh.points.device
    dtype = mesh.points.dtype
    n_cells = mesh.n_cells

    if mesh.n_manifold_dims == 0:
        # A point has no shape to distort, so its dimensionless quality values
        # are ideal. Edge lengths and angles are undefined for a 0-simplex.
        ideal = torch.ones((n_cells,), dtype=dtype, device=device)
        undefined = torch.full(
            (n_cells,),
            float("nan"),
            dtype=dtype,
            device=device,
        )
        return TensorDict(
            {
                "aspect_ratio": ideal.clone(),
                "edge_length_ratio": ideal.clone(),
                "min_angle": undefined.clone(),
                "max_angle": undefined.clone(),
                "min_edge_length": undefined.clone(),
                "max_edge_length": undefined.clone(),
                "quality_score": ideal,
            },
            batch_size=torch.Size([n_cells]),
            device=device,
        )

    ### Compute edge lengths for each cell
    edge_lengths = compute_cell_edge_lengths(mesh)  # (n_cells, n_edges_per_cell)

    max_edge = edge_lengths.max(dim=1).values
    min_edge = edge_lengths.min(dim=1).values

    eps = safe_eps(dtype)
    min_edge_denominator = torch.where(
        min_edge > 0,
        min_edge,
        torch.ones_like(min_edge),
    )
    edge_length_ratio = max_edge / min_edge_denominator
    edge_length_ratio = torch.where(
        min_edge > 0,
        edge_length_ratio,
        torch.full_like(edge_length_ratio, float("inf")),
    )

    ### Compute a dimensionless, scale-invariant simplex aspect ratio
    cell_measures = mesh.cell_areas
    min_altitude = _compute_simplex_altitudes(mesh, cell_measures).min(dim=1).values
    min_altitude_denominator = torch.where(
        min_altitude > 0,
        min_altitude,
        torch.ones_like(min_altitude),
    )
    raw_aspect_ratio = max_edge / min_altitude_denominator

    # A regular d-simplex has max_edge / min_altitude = sqrt(2d / (d + 1)).
    # Normalize by that value so 1.0 is ideal in every manifold dimension.
    regular_simplex_ratio = math.sqrt(
        2 * mesh.n_manifold_dims / (mesh.n_manifold_dims + 1)
    )
    aspect_ratio = (raw_aspect_ratio / regular_simplex_ratio).clamp(min=1.0)
    aspect_ratio = torch.where(
        (cell_measures > 0) & (min_altitude > 0),
        aspect_ratio,
        torch.full_like(aspect_ratio, float("inf")),
    )

    ### Compute interior angles at each vertex of each cell
    if mesh.n_manifold_dims >= 2:
        # Unified formula works for triangles, tetrahedra, and higher simplices
        all_angles = compute_vertex_angles(mesh)  # (n_cells, n_verts_per_cell)
        min_angle = all_angles.min(dim=1).values
        max_angle = all_angles.max(dim=1).values
    else:
        # For 1D manifolds (edges), interior angles are not meaningful
        min_angle = torch.full((n_cells,), float("nan"), dtype=dtype, device=device)
        max_angle = torch.full((n_cells,), float("nan"), dtype=dtype, device=device)

    ### Compute combined quality score
    # Perfect simplex has:
    # - edge_length_ratio = 1.0 (all edges equal)
    # - all vertex angles are equal
    # - aspect_ratio = 1.0

    # Quality score combines multiple metrics
    # Each component in [0, 1] where 1 is perfect

    # Edge uniformity: 1 / edge_length_ratio (clamped)
    edge_uniformity = 1.0 / torch.clamp(edge_length_ratio, min=1.0, max=10.0)

    # Aspect ratio quality: 1 / aspect_ratio (clamped)
    aspect_quality = 1.0 / torch.clamp(aspect_ratio, min=1.0, max=10.0)

    # Angle quality: measure how uniform the vertex angles are within each cell
    if mesh.n_manifold_dims == 2:
        # For triangles: compare against equilateral ideal (pi/3)
        ideal_angle = torch.pi / 3
        min_angle_quality = torch.clamp(min_angle / ideal_angle, max=1.0)
        max_angle_quality = torch.clamp(ideal_angle / max_angle, max=1.0)
        angle_quality = (min_angle_quality + max_angle_quality) / 2
    elif mesh.n_manifold_dims >= 3:
        # For tets and higher: use min/max ratio (1.0 for regular simplex)
        angle_quality = torch.clamp(min_angle / max_angle.clamp(min=eps), max=1.0)
    else:
        angle_quality = torch.ones((n_cells,), dtype=dtype, device=device)

    # Combined score (geometric mean)
    quality_score = (edge_uniformity * aspect_quality * angle_quality) ** (1 / 3)

    return TensorDict(
        {
            "aspect_ratio": aspect_ratio,
            "edge_length_ratio": edge_length_ratio,
            "min_angle": min_angle,
            "max_angle": max_angle,
            "min_edge_length": min_edge,
            "max_edge_length": max_edge,
            "quality_score": quality_score,
        },
        batch_size=torch.Size([n_cells]),
        device=device,
    )
