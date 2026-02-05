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

"""Topology validation for simplicial meshes.

This module provides functions to check topological properties of meshes:
- Watertight checking: mesh has no boundary (all facets shared by exactly 2 cells)
- Manifold checking: mesh is a valid topological manifold
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def is_watertight(mesh: "Mesh") -> bool:
    """Check if mesh is watertight (has no boundary).

    A mesh is watertight if every codimension-1 facet is shared by exactly 2 cells.
    This means the mesh forms a closed surface/volume with no holes or gaps.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh to check

    Returns
    -------
    bool
        True if mesh is watertight (no boundary facets), False otherwise

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral, cylinder_open
    >>> # Closed sphere is watertight
    >>> sphere = sphere_icosahedral.load(subdivisions=3)
    >>> assert is_watertight(sphere) == True
    >>>
    >>> # Open cylinder with holes at ends
    >>> cylinder = cylinder_open.load()
    >>> assert is_watertight(cylinder) == False
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    ### Empty mesh is considered watertight
    if mesh.n_cells == 0:
        return True

    ### Extract all codimension-1 facets
    candidate_facets, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Deduplicate and get counts
    _, _, counts = categorize_facets_by_count(candidate_facets, target_counts="all")

    ### Watertight iff all facets appear exactly twice
    # Each facet should be shared by exactly 2 cells
    return bool(torch.all(counts == 2))


def is_manifold(
    mesh: "Mesh",
    check_level: Literal["facets", "edges", "full"] = "full",
) -> bool:
    """Check if mesh is a valid topological manifold.

    A mesh is a manifold if it locally looks like Euclidean space at every point.
    This function checks various topological constraints depending on the check level.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh to check
    check_level : {"facets", "edges", "full"}, optional
        Level of checking to perform:
        - "facets": Only check codimension-1 facets (each appears 1-2 times)
        - "edges": Check facets + edge neighborhoods (for 2D/3D meshes)
        - "full": Complete manifold validation (default)

    Returns
    -------
    bool
        True if mesh passes the specified manifold checks, False otherwise

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral, cylinder_open
    >>> # Valid manifold (sphere)
    >>> sphere = sphere_icosahedral.load(subdivisions=3)
    >>> assert is_manifold(sphere) == True
    >>>
    >>> # Manifold with boundary (open cylinder)
    >>> cylinder = cylinder_open.load()
    >>> assert is_manifold(cylinder) == True  # manifold with boundary is OK

    Notes
    -----
    This function checks topological constraints but does not check for
    geometric self-intersections (which would require expensive spatial queries).
    """
    ### Empty mesh is considered a valid manifold
    if mesh.n_cells == 0:
        return True

    ### Check facets (codimension-1)
    if not _check_facets_manifold(mesh):
        return False

    if check_level == "facets":
        return True

    ### Check edges (for 2D and 3D meshes)
    if mesh.n_manifold_dims >= 2:
        if not _check_edges_manifold(mesh):
            return False

    if check_level == "edges":
        return True

    ### Full check includes vertices (for 2D and 3D meshes)
    if mesh.n_manifold_dims >= 2:
        if not _check_vertices_manifold(mesh):
            return False

    return True


def _check_facets_manifold(mesh: "Mesh") -> bool:
    """Check if facets satisfy manifold constraints.

    For a manifold (possibly with boundary), each codimension-1 facet must appear
    in at most 2 cells. Facets appearing once are on the boundary; facets appearing
    twice are interior.

    Parameters
    ----------
    mesh : Mesh
        Input mesh

    Returns
    -------
    bool
        True if facets satisfy manifold constraints
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    ### Extract all codimension-1 facets
    candidate_facets, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Deduplicate and get counts
    _, _, counts = categorize_facets_by_count(candidate_facets, target_counts="all")

    ### For manifold: each facet appears at most twice (1 = boundary, 2 = interior)
    # If any facet appears 3+ times, it's a non-manifold edge
    return bool(torch.all(counts <= 2))


def _check_edges_manifold(mesh: "Mesh") -> bool:
    """Check if edges satisfy manifold constraints.

    For 2D manifolds (triangles): Each edge should be shared by at most 2 triangles.
    For 3D manifolds (tetrahedra): Each edge should have a valid "link" - the set of
    facets (triangles) incident to the edge should form a topological disk or circle.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must have n_manifold_dims >= 2)

    Returns
    -------
    bool
        True if edges satisfy manifold constraints
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    ### For 2D meshes, edges are codimension-1, already checked in _check_facets_manifold
    if mesh.n_manifold_dims == 2:
        return True

    ### For 3D meshes, extract edges (codimension-2 facets)
    if mesh.n_manifold_dims == 3:
        candidate_edges, parent_cell_indices = extract_candidate_facets(
            mesh.cells,
            manifold_codimension=2,
        )

        ### Find unique edges and their parent cells
        unique_edges, inverse_indices = torch.unique(
            candidate_edges,
            dim=0,
            return_inverse=True,
        )

        ### For each edge, check that the cells around it form a valid configuration
        # In a manifold, the triangular faces around an edge should form a cycle
        # (for interior edges) or a fan (for boundary edges)

        ### Simple check: count cells per edge
        # In a 3D manifold, an edge can be shared by any number of tetrahedra,
        # but the triangular faces around the edge must form a valid fan/cycle

        ### For now, we do a simpler check: ensure each edge appears in at least one cell
        # A more sophisticated check would require analyzing the link of the edge
        edge_counts = torch.zeros(
            len(unique_edges), dtype=torch.int64, device=mesh.cells.device
        )
        edge_counts.scatter_add_(
            dim=0,
            index=inverse_indices,
            src=torch.ones_like(inverse_indices),
        )

        ### All edges should be used by at least one cell
        if torch.any(edge_counts == 0):
            return False

        ### Additional check: extract the triangular faces around each edge
        # and verify they form a topological disk or circle
        # This is more complex and requires analyzing face adjacency
        # For now, we rely on the facet check which catches most non-manifold cases

        return True

    ### For higher dimensions, we don't have specific checks yet
    return True


def _check_vertices_manifold(mesh: "Mesh") -> bool:
    """Check if vertices satisfy manifold constraints.

    For a manifold, the link of each vertex (the set of cells incident to the vertex)
    must form a valid topological structure:
    - For 2D: The edges around each vertex form a single cycle or fan
    - For 3D: The faces around each vertex form a single connected surface

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must have n_manifold_dims >= 2)

    Returns
    -------
    bool
        True if vertices satisfy manifold constraints
    """
    ### For 2D meshes, check that edges around each vertex form a valid fan/cycle
    if mesh.n_manifold_dims == 2:
        return _check_2d_vertex_manifold(mesh)

    ### For 3D meshes, check that faces around each vertex form a connected surface
    if mesh.n_manifold_dims == 3:
        return _check_3d_vertex_manifold(mesh)

    ### For other dimensions, no specific check
    return True


def _check_2d_vertex_manifold(mesh: "Mesh") -> bool:
    """Check vertex manifold constraints for 2D meshes.

    For a 2D triangular mesh to be manifold at a vertex, the triangles around the
    vertex must form a single fan (for boundary vertices) or a complete cycle
    (for interior vertices).

    Parameters
    ----------
    mesh : Mesh
        2D triangular mesh

    Returns
    -------
    bool
        True if all vertices satisfy 2D manifold constraints
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    ### Extract edges (codimension-1 for 2D)
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Find unique edges
    unique_edges, inverse_indices, edge_counts = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    ### For each vertex, count how many boundary edges are incident
    # In a manifold with boundary, each boundary vertex should have exactly 2 boundary edges
    # In a closed manifold, no vertex should have boundary edges

    boundary_edge_mask = edge_counts == 1
    boundary_edges = unique_edges[boundary_edge_mask]

    if len(boundary_edges) > 0:
        ### Count boundary edges per vertex
        vertex_boundary_count = torch.zeros(
            mesh.n_points, dtype=torch.int64, device=mesh.cells.device
        )
        vertex_boundary_count.scatter_add_(
            dim=0,
            index=boundary_edges.flatten(),
            src=torch.ones(
                boundary_edges.numel(), dtype=torch.int64, device=mesh.cells.device
            ),
        )

        ### Each boundary vertex should have exactly 2 boundary edges (forms a chain)
        # Non-boundary vertices should have 0
        valid_counts = (vertex_boundary_count == 0) | (vertex_boundary_count == 2)
        if not torch.all(valid_counts):
            return False

    return True


def _check_3d_vertex_manifold(mesh: "Mesh") -> bool:
    """Check vertex manifold constraints for 3D tetrahedral meshes.

    For a 3D mesh to be manifold at vertex v, the **link** of v must be
    connected. The link at v consists of one triangular face per incident
    tetrahedron (the face opposite to v, formed by the tet's other 3
    vertices). Two link faces are adjacent if their parent tets share a
    triangular face that contains v - equivalently, if their non-v vertex
    sets share exactly 2 vertices (an edge).

    A disconnected link indicates a **pinch point**: two groups of
    tetrahedra meeting at a single vertex without sharing any face that
    contains it. This is the primary non-manifold vertex configuration not
    caught by the facet and edge checks.

    Algorithm:
        For each vertex v with 2+ incident tetrahedra:

        1. Extract the link face (3 non-v vertices, sorted) from each
           incident tet.
        2. For each edge of each link face, record which local tet owns it.
           When two local tets share a link-face edge, they are adjacent in
           the link graph.
        3. Use union-find on the local tets, merging via shared edges.
        4. Verify that all local tets end up in one connected component.

    Parameters
    ----------
    mesh : Mesh
        Input 3D tetrahedral mesh.

    Returns
    -------
    bool
        True if all vertices have connected links (manifold at all vertices).
    """
    from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

    p2c = get_point_to_cells_adjacency(mesh)

    ### Move to CPU numpy for the per-vertex connectivity check
    offsets = p2c.offsets.cpu().numpy()
    p2c_indices = p2c.indices.cpu().numpy()
    cells_np = mesh.cells.cpu().numpy()

    for v in range(mesh.n_points):
        start, end = offsets[v], offsets[v + 1]
        n_incident = end - start

        ### Vertices with 0 or 1 incident tets have trivially connected links
        if n_incident <= 1:
            continue

        incident_tet_verts = cells_np[p2c_indices[start:end]]  # (n_incident, 4)

        ### Extract sorted link faces: the 3 non-v vertices of each tet
        link_faces: list[tuple[int, int, int]] = []
        for row in incident_tet_verts:
            non_v = row[row != v]
            if len(non_v) != 3:
                continue  # Degenerate tet with duplicate vertex v
            non_v.sort()
            link_faces.append((int(non_v[0]), int(non_v[1]), int(non_v[2])))

        n_faces = len(link_faces)
        if n_faces <= 1:
            continue

        ### Union-find: merge local tets whose link faces share an edge
        # For each edge of each link face, the first tet to claim that edge
        # becomes the representative; subsequent tets are unioned with it.
        parent = list(range(n_faces))
        edge_to_first: dict[tuple[int, int], int] = {}

        for face_idx, (a, b, c) in enumerate(link_faces):
            for edge in ((a, b), (a, c), (b, c)):
                if edge in edge_to_first:
                    _uf_union(parent, edge_to_first[edge], face_idx)
                else:
                    edge_to_first[edge] = face_idx

        ### Verify single connected component
        root = _uf_find(parent, 0)
        for i in range(1, n_faces):
            if _uf_find(parent, i) != root:
                return False

    return True


def _uf_find(parent: list[int], i: int) -> int:
    """Find root of element *i* with path halving (union-find helper)."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _uf_union(parent: list[int], i: int, j: int) -> None:
    """Merge the components containing *i* and *j* (union-find helper)."""
    ri, rj = _uf_find(parent, i), _uf_find(parent, j)
    if ri != rj:
        parent[ri] = rj
