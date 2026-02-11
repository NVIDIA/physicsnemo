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

"""Topology validation for simplicial meshes.

This module provides functions to check topological properties of meshes:
- Watertight checking: mesh has no boundary (all facets shared by exactly 2 cells)
- Manifold checking: mesh is a valid topological manifold
"""

from typing import TYPE_CHECKING, Literal

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
    For 3D manifolds (tetrahedra): Each edge should have a valid "link" - the
    codimension-1 faces (triangles) incident to the edge must form a single
    connected chain (boundary edge) or cycle (interior edge). Two faces
    containing the same edge are adjacent when their parent tets share a
    codimension-1 facet that also contains the edge - equivalently, when the
    two faces share a vertex besides the edge's two endpoints.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must have n_manifold_dims >= 2)

    Returns
    -------
    bool
        True if edges satisfy manifold constraints
    """
    ### For 2D meshes, edges are codimension-1, already checked in _check_facets_manifold
    if mesh.n_manifold_dims == 2:
        return True

    ### For 3D meshes, verify edge-link connectivity
    if mesh.n_manifold_dims == 3:
        return _check_3d_edge_link_connectivity(mesh)

    ### For higher dimensions, we don't have specific checks yet
    return True


def _check_3d_edge_link_connectivity(mesh: "Mesh") -> bool:
    """Check that the face-link around every edge is connected (3D meshes).

    For each edge in a 3D tetrahedral mesh, collect every codimension-1
    face (triangle) that contains the edge. Two such faces are "link-
    adjacent" when they share a third vertex (the non-edge vertex of each
    face) via their parent tets sharing a triangular face that contains
    the edge. All the faces around an edge must form a single connected
    component; a disconnected link indicates a non-manifold edge.

    The implementation is fully vectorized via union-find, following the
    same pattern used in ``_check_3d_vertex_manifold``.

    Parameters
    ----------
    mesh : Mesh
        Input 3D tetrahedral mesh.

    Returns
    -------
    bool
        True if every edge has a connected face-link.
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    device = mesh.cells.device

    ### Step 1: Extract candidate edges and their parent tets
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=2,
    )
    # candidate_edges: (n_candidate_edges, 2), parent_cell_indices: (n_candidate_edges,)

    ### Step 2: Map each candidate edge to a unique edge index
    unique_edges, edge_inverse = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )
    n_unique_edges = len(unique_edges)
    n_candidates = len(candidate_edges)

    if n_unique_edges == 0:
        return True

    ### Step 3: For each candidate edge, find the "third vertex" of the parent
    # tet that is NOT one of the edge's two endpoints.  In a tet (4 vertices)
    # with an edge occupying 2, there are 2 remaining vertices.  Each such
    # (edge, third_vertex) triple identifies one face of the tet that contains
    # the edge.  Two candidate edges that map to the same unique edge and
    # share a third vertex came from tets that share a face containing the
    # edge - making their link-faces adjacent.

    tet_verts = mesh.cells[parent_cell_indices]  # (n_candidates, 4)
    edge_v0 = candidate_edges[:, 0].unsqueeze(1)  # (n_candidates, 1)
    edge_v1 = candidate_edges[:, 1].unsqueeze(1)  # (n_candidates, 1)

    # Mask: which columns of each tet are NOT part of the edge
    not_edge = (tet_verts != edge_v0) & (tet_verts != edge_v1)  # (n_candidates, 4)

    # Extract the two non-edge vertices per candidate.
    # not_edge has exactly 2 True values per row (for non-degenerate tets).
    # Collect them into a (n_candidates, 2) tensor.
    # Use a scatter trick: for each row, the True columns give the third vertices.
    third_verts = tet_verts[not_edge].reshape(n_candidates, -1)  # (n_candidates, 2)

    # If a tet is degenerate (edge vertex appears >2 times), third_verts may
    # have fewer than 2 columns.  Skip the check for such edges.
    if third_verts.shape[1] < 2:
        return True

    ### Step 4: Build adjacency between candidates that share both the same
    # unique edge AND at least one third vertex.  Two candidates (c_a, c_b)
    # are adjacent when edge_inverse[c_a] == edge_inverse[c_b] and they share
    # a third vertex.
    #
    # Strategy: for each (unique_edge_id, third_vertex) pair, group the
    # candidates that have that pair and union consecutive members.

    # Expand: each candidate contributes two (edge_id, third_vert) keys.
    # Use repeat_interleave so that entries i*2 and i*2+1 both correspond
    # to candidate i (matching the interleaved layout of flatten()).
    edge_ids_2x = edge_inverse.repeat_interleave(2)  # (2 * n_candidates,)
    third_verts_flat = third_verts.flatten()  # (2 * n_candidates,)
    cand_indices_2x = torch.arange(n_candidates, device=device).repeat_interleave(2)

    # Use lexicographic sort via two stable argsorts to avoid integer overflow
    order = torch.argsort(third_verts_flat.long(), stable=True)
    order = order[torch.argsort(edge_ids_2x[order].long(), stable=True)]

    sorted_cand = cand_indices_2x[order]
    sorted_edge_ids = edge_ids_2x[order]
    sorted_third_verts = third_verts_flat[order]

    # Consecutive entries with the same (edge_id, third_vertex) pair
    # are candidates sharing a link-adjacent face.
    same_key = (sorted_edge_ids[:-1] == sorted_edge_ids[1:]) & (
        sorted_third_verts[:-1] == sorted_third_verts[1:]
    )
    pair_a = sorted_cand[:-1][same_key]
    pair_b = sorted_cand[1:][same_key]

    ### Step 5: Union-find over candidates to find connected components per edge
    from physicsnemo.mesh.utilities._duplicate_detection import (
        vectorized_connected_components,
    )

    pairs = torch.stack([pair_a, pair_b], dim=1)
    labels = vectorized_connected_components(pairs, n_candidates)

    ### Step 6: Check single connected component per unique edge.
    # For each unique edge, all candidate entries must share the same root.
    min_labels = torch.full(
        (n_unique_edges,),
        n_candidates,
        dtype=torch.long,
        device=device,
    )
    max_labels = torch.zeros(n_unique_edges, dtype=torch.long, device=device)
    min_labels.scatter_reduce_(0, edge_inverse, labels, reduce="amin")
    max_labels.scatter_reduce_(0, edge_inverse, labels, reduce="amax")

    # Count candidates per edge to identify edges with 2+ tets (others are
    # trivially connected since they have a single candidate).
    edge_counts = torch.zeros(n_unique_edges, dtype=torch.long, device=device)
    edge_counts.scatter_add_(0, edge_inverse, torch.ones_like(edge_inverse))
    multi = edge_counts >= 2

    return bool(torch.all(min_labels[multi] == max_labels[multi]))


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

    This implementation is fully vectorized (no Python loops over vertices),
    following the union-find pattern from
    :func:`~physicsnemo.mesh.utilities._duplicate_detection.compute_canonical_indices`.

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

    device = mesh.cells.device

    p2c = get_point_to_cells_adjacency(mesh)

    ### Step 1: Expand p2c to all (vertex_id, tet_id) pairs
    vertex_ids, tet_ids = p2c.expand_to_pairs()  # both shape (total_pairs,)
    total = len(vertex_ids)
    if total == 0:
        return True

    ### Step 2: Extract link faces (3 non-v vertices per incident tet)
    tet_verts = mesh.cells[tet_ids]  # (total, 4)

    # Find which column of each tet holds the vertex v
    # (total, 4) == (total, 1) -> bool mask, argmax gives first True column
    v_col = (tet_verts == vertex_ids.unsqueeze(1)).byte().argmax(dim=1)  # (total,)

    # Detect degenerate tets (vertex appears more than once)
    v_occurrence_count = (tet_verts == vertex_ids.unsqueeze(1)).sum(dim=1)
    is_degenerate = v_occurrence_count > 1

    # Precomputed lookup: given v is at column c, take columns COL_MAP[c]
    _COL_MAP = torch.tensor(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=torch.long,
        device=device,
    )
    gather_cols = _COL_MAP[v_col]  # (total, 3)
    link_faces = torch.gather(tet_verts, 1, gather_cols)  # (total, 3)

    # Sort each link face for canonical ordering
    link_faces, _ = torch.sort(link_faces, dim=1)

    ### Step 3: Generate all edges of all link faces
    # Each sorted face (a, b, c) gives edges: (a,b), (a,c), (b,c)
    edge_ab = link_faces[:, :2]  # (total, 2) -> columns 0,1
    edge_ac = link_faces[:, [0, 2]]  # (total, 2) -> columns 0,2
    edge_bc = link_faces[:, 1:]  # (total, 2) -> columns 1,2

    all_edges = torch.cat([edge_ab, edge_ac, edge_bc], dim=0)  # (3*total, 2)

    # Matching vertex_ids and face indices, repeated 3x
    face_indices = torch.arange(total, device=device)
    vertex_ids_3x = vertex_ids.repeat(3)  # (3*total,)
    face_indices_3x = face_indices.repeat(3)  # (3*total,)

    ### Step 4: Find pairs of link faces sharing an edge at the same vertex
    # Sort by composite key (vertex_id, edge_v0, edge_v1) to group identical
    # (vertex, edge) tuples together. Consecutive entries sharing a key are
    # face pairs connected via that edge.
    # Use lexicographic sort via three stable argsorts to avoid integer overflow
    order = torch.argsort(all_edges[:, 1], stable=True)
    order = order[torch.argsort(all_edges[order, 0], stable=True)]
    order = order[torch.argsort(vertex_ids_3x[order], stable=True)]

    sorted_face_idx = face_indices_3x[order]
    sorted_vertex_ids = vertex_ids_3x[order]
    sorted_edges = all_edges[order]

    # Adjacent entries with the same (vertex_id, edge_v0, edge_v1) are face pairs
    same_key = (
        (sorted_vertex_ids[:-1] == sorted_vertex_ids[1:])
        & (sorted_edges[:-1, 0] == sorted_edges[1:, 0])
        & (sorted_edges[:-1, 1] == sorted_edges[1:, 1])
    )
    pair_f1 = sorted_face_idx[:-1][same_key]
    pair_f2 = sorted_face_idx[1:][same_key]

    if len(pair_f1) == 0:
        # No shared edges at all - each link face is isolated.
        # Vertices with 0 or 1 incident tet are trivially manifold.
        # Vertices with 2+ incident tets but no shared edges are non-manifold.
        incident_counts = p2c.counts  # (n_points,)
        return bool(torch.all(incident_counts <= 1))

    ### Step 5: Vectorized union-find (using shared utility)
    from physicsnemo.mesh.utilities._duplicate_detection import (
        vectorized_connected_components,
    )

    pairs = torch.stack([pair_f1, pair_f2], dim=1)
    labels = vectorized_connected_components(pairs, total)

    ### Step 6: Check single connected component per vertex
    # For each vertex, all its link faces must share the same root label.
    # Use scatter to find min and max label per vertex.
    min_labels = torch.full((mesh.n_points,), total, dtype=torch.long, device=device)
    max_labels = torch.zeros(mesh.n_points, dtype=torch.long, device=device)

    min_labels.scatter_reduce_(0, vertex_ids, labels, reduce="amin")
    max_labels.scatter_reduce_(0, vertex_ids, labels, reduce="amax")

    # Only check vertices with 2+ incident tets (others are trivially manifold)
    incident_counts = p2c.counts  # (n_points,)
    multi_incident = incident_counts >= 2

    # Also exclude degenerate vertices from the check
    degenerate_per_vertex = torch.zeros(mesh.n_points, dtype=torch.bool, device=device)
    degenerate_per_vertex.scatter_(0, vertex_ids[is_degenerate], True)

    check_mask = multi_incident & ~degenerate_per_vertex
    return bool(torch.all(min_labels[check_mask] == max_labels[check_mask]))
