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

"""Boundary detection for simplicial meshes.

Provides functions to identify boundary vertices, edges, and cells in meshes.
A facet is on the boundary if it appears in only one cell (non-watertight /
manifold-with-boundary).
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _extract_boundary_facets(
    mesh: "Mesh",
    manifold_codimension: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract boundary facets at a given codimension.

    Shared helper that avoids duplicating the extract-then-categorize pattern
    across the public detection functions.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh (must have n_cells > 0).
    manifold_codimension : int
        Codimension of facets to extract and filter to boundary.

    Returns
    -------
    boundary_facets : torch.Tensor
        Unique facets appearing in exactly one cell,
        shape (n_boundary_facets, n_verts_per_facet).
    parent_cell_indices : torch.Tensor
        Parent cell index for every *candidate* facet (before deduplication),
        shape (n_candidates,).
    boundary_candidate_mask : torch.Tensor
        Boolean mask over candidates: True where the candidate maps to a
        kept boundary facet, shape (n_candidates,).
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    candidate_facets, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=manifold_codimension,
    )
    boundary_facets, inverse_indices, _ = categorize_facets_by_count(
        candidate_facets,
        target_counts="boundary",
    )
    boundary_candidate_mask = inverse_indices >= 0
    return boundary_facets, parent_cell_indices, boundary_candidate_mask


def get_boundary_vertices(mesh: "Mesh") -> torch.Tensor:
    """Identify vertices that lie on the mesh boundary.

    A vertex is on the boundary if it belongs to at least one boundary facet
    (a codimension-1 sub-simplex appearing in exactly one cell).

    For 1D manifolds the boundary facets are vertices (0-simplices), for 2D
    they are edges, and for 3D they are faces.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_points,) where True indicates boundary vertices

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import cylinder_open
    >>> # Cylinder with open ends
    >>> mesh = cylinder_open.load(n_circ=32, n_height=16)
    >>> is_boundary = get_boundary_vertices(mesh)
    >>> # Top and bottom circles are boundary vertices
    >>> assert is_boundary.sum() == 2 * 32  # 64 boundary vertices

    Notes
    -----
    For closed manifolds (watertight meshes), returns all False.
    """
    device = mesh.cells.device
    n_points = mesh.n_points

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return torch.zeros(n_points, dtype=torch.bool, device=device)

    ### Extract codimension-1 boundary facets and mark their vertices
    boundary_facets, _, _ = _extract_boundary_facets(mesh, manifold_codimension=1)

    is_boundary_vertex = torch.zeros(n_points, dtype=torch.bool, device=device)
    if len(boundary_facets) > 0:
        is_boundary_vertex.scatter_(0, boundary_facets.flatten(), True)

    return is_boundary_vertex


def get_boundary_cells(
    mesh: "Mesh",
    boundary_codimension: int = 1,
) -> torch.Tensor:
    """Identify cells that have at least one facet on the mesh boundary.

    A cell is on the boundary if it contains at least one k-codimension facet
    that appears in no other cell.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh
    boundary_codimension : int, optional
        Codimension of facets defining boundary membership.

        - 1 (default): cells with at least one codim-1 boundary facet
        - 2: cells with at least one codim-2 boundary facet (more permissive)
        - k: cells with at least one codim-k boundary facet

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_cells,) where True indicates boundary cells

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Two triangles sharing an edge, with 4 boundary edges total
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> is_boundary = get_boundary_cells(mesh, boundary_codimension=1)
    >>> assert is_boundary.all()  # Both triangles touch boundary edges

    Notes
    -----
    For closed manifolds (watertight meshes), returns all False.
    """
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

    ### Extract boundary facets and determine which parent cells they belong to
    _, parent_cell_indices, boundary_candidate_mask = _extract_boundary_facets(
        mesh,
        manifold_codimension=boundary_codimension,
    )

    ### Mark cells that contain at least one boundary facet
    is_boundary_cell = torch.zeros(n_cells, dtype=torch.bool, device=device)
    boundary_parent_cells = parent_cell_indices[boundary_candidate_mask]

    if len(boundary_parent_cells) > 0:
        is_boundary_cell.scatter_(0, boundary_parent_cells, True)

    return is_boundary_cell


def get_boundary_edges(mesh: "Mesh") -> torch.Tensor:
    """Get edges that lie on the mesh boundary.

    For 2D manifolds, boundary edges are codimension-1 facets appearing in only
    one cell. For 1D manifolds (edge meshes), the boundary consists of vertices
    rather than edges, so an empty tensor is returned. For 3D+ manifolds, boundary
    edges are those belonging to at least one boundary face (codimension-1 facet
    appearing in only one cell).

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_boundary_edges, 2) containing boundary edge connectivity.
        Returns empty tensor of shape (0, 2) for watertight meshes or 1D manifolds.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import cylinder_open
    >>> # Cylinder with open ends
    >>> mesh = cylinder_open.load(n_circ=32, n_height=16)
    >>> boundary_edges = get_boundary_edges(mesh)
    >>> # Top and bottom circles each have 32 edges = 64 total
    >>> assert len(boundary_edges) == 64

    Notes
    -----
    For closed manifolds (watertight meshes), returns empty tensor.
    """
    device = mesh.cells.device

    ### Handle empty mesh or 1D manifolds (whose boundary consists of vertices, not edges)
    if mesh.n_cells == 0 or mesh.n_manifold_dims < 2:
        return torch.zeros((0, 2), dtype=torch.int64, device=device)

    ### For 2D manifolds, boundary edges are codim-1 facets appearing in exactly 1 cell.
    ### For 3D+ manifolds, extract edges of boundary faces.
    if mesh.n_manifold_dims == 2:
        # Edges are codim-1 facets; boundary = appear in exactly 1 cell
        boundary_edges, _, _ = _extract_boundary_facets(mesh, manifold_codimension=1)
        return boundary_edges

    # For 3D+ manifolds: find boundary faces (codim-1), then extract their edges
    boundary_faces, _, _ = _extract_boundary_facets(mesh, manifold_codimension=1)

    if len(boundary_faces) == 0:
        return torch.zeros((0, 2), dtype=torch.int64, device=device)

    # Extract unique edges from boundary faces
    n_verts_per_face = boundary_faces.shape[1]
    from itertools import combinations

    combo_indices = list(combinations(range(n_verts_per_face), 2))
    combo_tensor = torch.tensor(
        combo_indices, dtype=torch.long, device=device
    )  # (n_combos, 2)

    # Gather edges from boundary faces: (n_boundary_faces, n_combos, 2)
    all_edges = boundary_faces[:, combo_tensor]
    # Reshape to (n_boundary_faces * n_combos, 2)
    all_edges = all_edges.reshape(-1, 2)
    # Sort each edge for canonical ordering
    all_edges = torch.sort(all_edges, dim=-1).values
    # Deduplicate
    boundary_edges = torch.unique(all_edges, dim=0)

    return boundary_edges
