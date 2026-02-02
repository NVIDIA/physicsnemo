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

"""Boundary detection for simplicial meshes.

Provides functions to identify boundary vertices, edges, and cells in meshes.
A facet is on the boundary if it appears in only one cell (non-watertight/manifold-with-boundary).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def get_boundary_vertices(mesh: "Mesh") -> torch.Tensor:
    """Identify vertices that lie on the mesh boundary.

    A vertex is on the boundary if it is incident to at least one boundary edge
    (for 2D+ manifolds) or is an endpoint of the chain (for 1D manifolds).

    Args:
        mesh: Input simplicial mesh

    Returns:
        Boolean tensor of shape (n_points,) where True indicates boundary vertices

    Example:
        >>> from physicsnemo.mesh.primitives.surfaces import cylinder_open
        >>> # Cylinder with open ends
        >>> mesh = cylinder_open.load(n_circ=32, n_height=16)
        >>> is_boundary = get_boundary_vertices(mesh)
        >>> # Top and bottom circles are boundary vertices
        >>> assert is_boundary.sum() == 2 * 32  # 64 boundary vertices

    Note:
        For closed manifolds (watertight meshes), returns all False.
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    device = mesh.cells.device
    n_points = mesh.n_points

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return torch.zeros(n_points, dtype=torch.bool, device=device)

    ### Extract boundary edges (codimension-1 facets that appear in only 1 cell)
    # For n-manifolds, a boundary edge is an (n-1)-facet with only 1 adjacent cell
    candidate_edges, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    # Get boundary edges (appear exactly once)
    boundary_edges, _, _ = categorize_facets_by_count(
        candidate_edges, target_counts="boundary"
    )

    ### Mark all vertices incident to boundary edges
    is_boundary_vertex = torch.zeros(n_points, dtype=torch.bool, device=device)
    if len(boundary_edges) > 0:
        is_boundary_vertex.scatter_(0, boundary_edges.flatten(), True)

    return is_boundary_vertex


def get_boundary_cells(
    mesh: "Mesh",
    boundary_codimension: int = 1,
) -> torch.Tensor:
    """Identify cells that have at least one facet on the mesh boundary.

    A cell is on the boundary if it contains at least one k-codimension facet
    that appears in no other cell.

    Args:
        mesh: Input simplicial mesh
        boundary_codimension: Codimension of facets defining boundary membership.
            - 1 (default): Cells with at least one codim-1 boundary facet (most restrictive)
              For 2D: triangles with at least one edge on boundary
              For 3D: tets with at least one face on boundary
            - 2: Cells with at least one codim-2 boundary facet (more permissive)
              For 3D: tets with at least one edge on boundary
            - k: Cells with at least one codim-k boundary facet

    Returns:
        Boolean tensor of shape (n_cells,) where True indicates boundary cells

    Example:
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> # Two triangles sharing an edge, with 4 boundary edges total
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> is_boundary = get_boundary_cells(mesh, boundary_codimension=1)
        >>> assert is_boundary.all()  # Both triangles touch boundary edges

    Note:
        For closed manifolds (watertight meshes), returns all False.
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    device = mesh.cells.device
    n_cells = mesh.n_cells

    ### Handle empty mesh
    if n_cells == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    ### Validate boundary_codimension
    if boundary_codimension < 1 or boundary_codimension > mesh.n_manifold_dims:
        raise ValueError(
            f"Invalid {boundary_codimension=}. "
            f"Must be in range [1, {mesh.n_manifold_dims}] for {mesh.n_manifold_dims=}"
        )

    ### Extract all k-codimension facets from cells
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=boundary_codimension,
    )

    ### Find boundary facets (appear exactly once)
    _, inverse_indices, _ = categorize_facets_by_count(
        candidate_facets, target_counts="boundary"
    )

    ### Map back to candidate facets
    candidate_is_boundary = inverse_indices >= 0

    ### Mark cells that contain at least one boundary facet
    is_boundary_cell = torch.zeros(n_cells, dtype=torch.bool, device=device)
    boundary_parent_cells = parent_cell_indices[candidate_is_boundary]

    if len(boundary_parent_cells) > 0:
        is_boundary_cell.scatter_(0, boundary_parent_cells, True)

    return is_boundary_cell


def get_boundary_edges(mesh: "Mesh") -> torch.Tensor:
    """Get edges that lie on the mesh boundary.

    An edge is on the boundary if it is a codimension-1 facet that appears in
    only one cell.

    Args:
        mesh: Input simplicial mesh

    Returns:
        Tensor of shape (n_boundary_edges, 2) containing boundary edge connectivity.
        Returns empty tensor of shape (0, 2) for watertight meshes.

    Example:
        >>> from physicsnemo.mesh.primitives.surfaces import cylinder_open
        >>> # Cylinder with open ends
        >>> mesh = cylinder_open.load(n_circ=32, n_height=16)
        >>> boundary_edges = get_boundary_edges(mesh)
        >>> # Top and bottom circles each have 32 edges = 64 total
        >>> assert len(boundary_edges) == 64

    Note:
        For closed manifolds (watertight meshes), returns empty tensor.
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    device = mesh.cells.device

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return torch.zeros((0, 2), dtype=torch.int64, device=device)

    ### Extract all edges (with duplicates)
    candidate_edges, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    # Get boundary edges (appear exactly once)
    boundary_edges, _, _ = categorize_facets_by_count(
        candidate_edges, target_counts="boundary"
    )

    return boundary_edges
