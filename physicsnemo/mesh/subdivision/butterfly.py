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

"""Butterfly subdivision for simplicial meshes.

Butterfly is an interpolating subdivision scheme where original vertices remain
fixed and new edge midpoints are computed using weighted stencils of neighboring
vertices. This produces smoother surfaces than linear subdivision.

The classical butterfly scheme is designed for 2D manifolds (triangular meshes).
This implementation provides the standard 2D butterfly and extensions/fallbacks
for other dimensions.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.subdivision._data import propagate_cell_data_to_children
from physicsnemo.mesh.subdivision._topology import (
    extract_unique_edges,
    generate_child_cells,
    get_subdivision_pattern,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _build_edge_to_triangle_pairs(
    candidate_edges: torch.Tensor,
    parent_cell_indices: torch.Tensor,
    n_unique_edges: int,
    unique_edge_hashes: torch.Tensor,
    max_vertex: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a (n_unique_edges, 2) tensor mapping each edge to its parent triangles.

    Parameters
    ----------
    candidate_edges : torch.Tensor
        All triangle half-edges (with duplicates), shape (n_candidates, 2).
    parent_cell_indices : torch.Tensor
        Parent triangle index for each candidate, shape (n_candidates,).
    n_unique_edges : int
        Number of unique edges in the mesh.
    unique_edge_hashes : torch.Tensor
        Hash of each unique edge (sorted order), shape (n_unique_edges,).
    max_vertex : int
        Value used for hash computation: ``hash = v0 * max_vertex + v1``.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Shape (n_unique_edges, 2). ``result[i]`` holds the (up to 2) parent
        triangle indices for unique edge ``i``; boundary edges have ``-1`` in
        the second slot.
    """
    ### Hash candidate edges (canonicalized) and map to unique edge indices
    sorted_cands, _ = torch.sort(candidate_edges, dim=1)
    cand_hash = sorted_cands[:, 0] * max_vertex + sorted_cands[:, 1]

    sorted_unique_hash, unique_sort_perm = torch.sort(unique_edge_hashes)
    positions = torch.searchsorted(sorted_unique_hash, cand_hash)
    positions = positions.clamp(max=len(sorted_unique_hash) - 1)
    edge_idx = unique_sort_perm[positions]  # Maps each candidate to a unique-edge index

    ### Fill pair table by scattering parent triangle indices
    pair_table = torch.full((n_unique_edges, 2), -1, dtype=torch.long, device=device)

    # First pass fills slot 0 for every edge
    pair_table[:, 0].scatter_(0, edge_idx, parent_cell_indices)

    # Second pass fills slot 1 for edges with 2 parents.  Mask out candidates
    # whose parent already occupies slot 0 so we only write the *other* parent.
    slot0_parent = pair_table[edge_idx, 0]
    is_second_parent = parent_cell_indices != slot0_parent
    second_idx = edge_idx[is_second_parent]
    second_parents = parent_cell_indices[is_second_parent]
    pair_table[:, 1].scatter_(0, second_idx, second_parents)

    return pair_table


def compute_butterfly_weights_2d(
    mesh: "Mesh",
    unique_edges: torch.Tensor,
) -> torch.Tensor:
    r"""Compute butterfly weighted positions for edge midpoints in 2D manifolds.

    For triangular meshes, uses the classical 8-point butterfly stencil
    (Dyn, Gregory & Levin 1990):

    - Edge vertices v0, v1: weight 1/2 each
    - Opposite vertices a, b in the two adjacent triangles: weight 1/8 each
    - Four "wing" vertices c, d, e, f: weight -1/16 each

    These weights sum to exactly 1.0 when all wing vertices exist. For
    interior edges adjacent to boundary edges (missing some wing vertices),
    the missing wing contributions are omitted (treated as zero weight).

    Boundary edges use the simple midpoint average of the two endpoints.

    Parameters
    ----------
    mesh : Mesh
        Input 2D manifold mesh (triangular).
    unique_edges : torch.Tensor
        Unique edge connectivity, shape (n_edges, 2).

    Returns
    -------
    torch.Tensor
        Edge midpoint positions using butterfly weights, shape (n_edges, n_spatial_dims).
    """
    n_edges = len(unique_edges)
    device = mesh.points.device
    dtype = mesh.points.dtype

    ### Step 1: Build edge → parent-triangle mapping
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    # Canonical (sorted) hashes for the unique edges
    sorted_unique, _ = torch.sort(unique_edges, dim=1)
    sorted_cands, _ = torch.sort(candidate_edges, dim=1)
    max_v = max(sorted_unique.max().item(), sorted_cands.max().item()) + 1
    unique_hash = sorted_unique[:, 0] * max_v + sorted_unique[:, 1]

    # Pair table: (n_edges, 2), with -1 for missing second triangle
    edge_tri_pairs = _build_edge_to_triangle_pairs(
        candidate_edges=candidate_edges,
        parent_cell_indices=parent_cell_indices,
        n_unique_edges=n_edges,
        unique_edge_hashes=unique_hash,
        max_vertex=max_v,
        device=device,
    )

    ### Step 2: Classify edges
    is_interior = edge_tri_pairs[:, 1] >= 0
    is_boundary = ~is_interior

    ### Step 3: Initialize midpoints
    edge_midpoints = torch.zeros(
        (n_edges, mesh.n_spatial_dims), dtype=dtype, device=device
    )

    ### Step 4: Boundary edges → simple average
    boundary_idx = torch.where(is_boundary)[0]
    if len(boundary_idx) > 0:
        bv0 = mesh.points[unique_edges[boundary_idx, 0]]
        bv1 = mesh.points[unique_edges[boundary_idx, 1]]
        edge_midpoints[boundary_idx] = (bv0 + bv1) / 2

    ### Step 5: Interior edges → 8-point butterfly stencil
    interior_idx = torch.where(is_interior)[0]
    n_interior = len(interior_idx)

    if n_interior == 0:
        return edge_midpoints

    int_edges = unique_edges[interior_idx]  # (n_int, 2)
    int_tris = edge_tri_pairs[interior_idx]  # (n_int, 2)  [T0, T1]

    ### Find opposite vertices a (in T0) and b (in T1)
    v0 = int_edges[:, 0]  # (n_int,)
    v1 = int_edges[:, 1]  # (n_int,)

    T0_verts = mesh.cells[int_tris[:, 0]]  # (n_int, 3)
    T0_opp_mask = ~((T0_verts == v0.unsqueeze(1)) | (T0_verts == v1.unsqueeze(1)))
    a = torch.gather(
        T0_verts, 1, torch.argmax(T0_opp_mask.int(), dim=1, keepdim=True)
    ).squeeze(1)

    T1_verts = mesh.cells[int_tris[:, 1]]  # (n_int, 3)
    T1_opp_mask = ~((T1_verts == v0.unsqueeze(1)) | (T1_verts == v1.unsqueeze(1)))
    b = torch.gather(
        T1_verts, 1, torch.argmax(T1_opp_mask.int(), dim=1, keepdim=True)
    ).squeeze(1)

    ### 4-point base contribution: 1/2·v0 + 1/2·v1 + 1/8·a + 1/8·b
    midpoint = (
        0.5 * mesh.points[v0]
        + 0.5 * mesh.points[v1]
        + 0.125 * mesh.points[a]
        + 0.125 * mesh.points[b]
    )  # (n_int, n_spatial_dims)

    ### Step 6: Wing vertices (−1/16 each)
    #
    # For edge (v0, v1) with opposite vertices a (in T0) and b (in T1):
    #   wing_c: opposite vertex in the OTHER tri sharing edge (v0, a)  [not T0]
    #   wing_d: opposite vertex in the OTHER tri sharing edge (v1, a)  [not T0]
    #   wing_e: opposite vertex in the OTHER tri sharing edge (v0, b)  [not T1]
    #   wing_f: opposite vertex in the OTHER tri sharing edge (v1, b)  [not T1]

    # Pre-compute sorted unique hash lookup for vectorized edge queries
    sorted_uhash, uhash_perm = torch.sort(unique_hash)
    n_uhash = len(sorted_uhash)

    wing_edges_and_known_tris = [
        (torch.stack([v0, a], dim=1), int_tris[:, 0]),  # wing_c
        (torch.stack([v1, a], dim=1), int_tris[:, 0]),  # wing_d
        (torch.stack([v0, b], dim=1), int_tris[:, 1]),  # wing_e
        (torch.stack([v1, b], dim=1), int_tris[:, 1]),  # wing_f
    ]

    for wing_edge, known_tri in wing_edges_and_known_tris:
        # Hash the wing edges and look them up in the unique edge set
        ws, _ = torch.sort(wing_edge, dim=1)
        whash = ws[:, 0] * max_v + ws[:, 1]

        pos = torch.searchsorted(sorted_uhash, whash).clamp(max=n_uhash - 1)
        matched = sorted_uhash[pos] == whash
        eidx = uhash_perm[pos]  # Index into edge_tri_pairs

        # Look up the two parent triangles for each wing edge
        wing_pair = edge_tri_pairs[eidx]  # (n_int, 2)

        # Select the "other" triangle (not known_tri)
        other_tri = torch.where(
            wing_pair[:, 0] == known_tri,
            wing_pair[:, 1],
            wing_pair[:, 0],
        )

        # A wing vertex exists iff the edge was found AND the other tri is valid
        has_wing = matched & (other_tri >= 0)

        if not has_wing.any():
            continue

        valid = torch.where(has_wing)[0]
        other_verts = mesh.cells[other_tri[valid]]  # (n_valid, 3)

        # The wing vertex is the one in other_tri that is NOT in wing_edge
        we0 = wing_edge[valid, 0].unsqueeze(1)
        we1 = wing_edge[valid, 1].unsqueeze(1)
        opp_mask = ~((other_verts == we0) | (other_verts == we1))
        wing_vert = torch.gather(
            other_verts, 1, torch.argmax(opp_mask.int(), dim=1, keepdim=True)
        ).squeeze(1)

        midpoint[valid] -= (1.0 / 16.0) * mesh.points[wing_vert]

    edge_midpoints[interior_idx] = midpoint
    return edge_midpoints


def subdivide_butterfly(mesh: "Mesh") -> "Mesh":
    """Perform one level of butterfly subdivision on the mesh.

    Butterfly subdivision is an interpolating scheme that produces smoother
    results than linear subdivision by using weighted stencils for new vertices.

    Properties:
    - Interpolating: original vertices remain unchanged
    - New edge midpoints use weighted neighbor stencils
    - Designed for 2D manifolds (triangular meshes)
    - For non-2D manifolds: falls back to linear subdivision with warning

    The connectivity pattern is identical to linear subdivision (same topology),
    but the geometric positions of new vertices differ.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to subdivide

    Returns
    -------
    Mesh
        Subdivided mesh with butterfly-weighted vertex positions

    Raises
    ------
    NotImplementedError
        If n_manifold_dims is not 2 (may be relaxed in future)

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Smooth a triangular surface
        >>> mesh = sphere_icosahedral.load(subdivisions=2)
        >>> smooth = subdivide_butterfly(mesh)
        >>> # smooth has same connectivity as linear subdivision
        >>> # but smoother geometry from weighted stencils
    """
    from physicsnemo.mesh.mesh import Mesh

    ### Check manifold dimension
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Butterfly subdivision currently only supports 2D manifolds (triangular meshes). "
            f"Got {mesh.n_manifold_dims=}. "
            f"For other dimensions, use linear subdivision instead."
        )

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return mesh

    ### Extract unique edges
    unique_edges, edge_inverse = extract_unique_edges(mesh)
    n_original_points = mesh.n_points

    ### Compute edge midpoints using butterfly weights
    edge_midpoints = compute_butterfly_weights_2d(mesh, unique_edges)

    ### Create new points: original (unchanged) + butterfly midpoints
    new_points = torch.cat([mesh.points, edge_midpoints], dim=0)

    ### Interpolate point_data to edge midpoints
    # For butterfly, we could use the same weighted stencil for data,
    # but for simplicity, use linear interpolation (average of endpoints)
    from physicsnemo.mesh.subdivision._data import interpolate_point_data_to_edges

    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )

    ### Get subdivision pattern (same as linear)
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device)

    ### Generate child cells (same topology as linear)
    child_cells, parent_indices = generate_child_cells(
        parent_cells=mesh.cells,
        edge_inverse=edge_inverse,
        n_original_points=n_original_points,
        subdivision_pattern=subdivision_pattern,
    )

    ### Propagate cell_data
    new_cell_data = propagate_cell_data_to_children(
        cell_data=mesh.cell_data,
        parent_indices=parent_indices,
        n_total_children=len(child_cells),
    )

    ### Create and return subdivided mesh
    return Mesh(
        points=new_points,
        cells=child_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )
