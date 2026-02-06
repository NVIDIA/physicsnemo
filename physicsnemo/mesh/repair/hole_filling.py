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

"""Fill holes in triangle meshes.

Detects boundary loops (connected components of boundary edges) and closes
each loop independently with fan triangulation from a centroid vertex.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs
from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _trace_boundary_loops(
    boundary_edges: torch.Tensor,
) -> list[torch.Tensor]:
    """Trace disjoint boundary loops from a set of boundary edges.

    Each boundary edge connects two vertices. A boundary loop is a connected
    cycle of boundary edges. This function identifies all such loops by
    building an :class:`Adjacency` structure from the edges, converting it to
    Python lists for efficient scalar access, and walking the graph on CPU.

    Parameters
    ----------
    boundary_edges : torch.Tensor
        Boundary edges, shape (n_boundary_edges, 2). Each row is [v0, v1].

    Returns
    -------
    list[torch.Tensor]
        Each tensor contains the ordered vertex indices forming one boundary
        loop (on the same device as *boundary_edges*). Vertices are in
        traversal order; each consecutive pair shares a boundary edge, and
        the last connects back to the first.
    """
    if len(boundary_edges) == 0:
        return []

    device = boundary_edges.device

    ### Remap sparse vertex indices to a compact 0..n_boundary_verts-1 range
    v0 = boundary_edges[:, 0]
    v1 = boundary_edges[:, 1]
    all_edge_verts = torch.cat([v0, v1])
    unique_verts, inverse = torch.unique(all_edge_verts, return_inverse=True)
    n_boundary_verts = len(unique_verts)

    compact_v0 = inverse[: len(v0)]
    compact_v1 = inverse[len(v0) :]

    ### Build bidirectional Adjacency (each edge contributes both directions)
    compact_sources = torch.cat([compact_v0, compact_v1])
    compact_targets = torch.cat([compact_v1, compact_v0])
    adj = build_adjacency_from_pairs(compact_sources, compact_targets, n_boundary_verts)

    ### Convert adjacency to Python lists for scalar-access walks.
    # This replaces O(N) per-vertex .item() GPU-CPU syncs with 2 bulk transfers.
    offsets = adj.offsets.tolist()
    indices = adj.indices.tolist()
    unique_verts_cpu = unique_verts.cpu()

    ### Walk loops using pure-Python indexing.
    # This walk is sequential by nature: each step depends on (current, prev)
    # to select the next vertex, so it cannot be vectorized. Parallel algorithms
    # (union-find, label propagation) can find components but not vertex ordering,
    # which is required for fan triangulation. Boundary loops are small in
    # practice, so the O(N) CPU walk is not a bottleneck.
    visited = [False] * n_boundary_verts
    loops: list[torch.Tensor] = []

    for start in range(n_boundary_verts):
        if visited[start]:
            continue

        loop_compact = [start]
        visited[start] = True

        # Step to the first neighbor
        current = indices[offsets[start]]
        prev = start

        while current != start:
            visited[current] = True
            loop_compact.append(current)

            # Get neighbors of current vertex
            nb_s = offsets[current]
            nb_e = offsets[current + 1]

            if nb_e - nb_s != 2:
                # Non-manifold boundary vertex - abandon this walk
                break

            # Pick the neighbor that isn't the one we came from
            n0, n1 = indices[nb_s], indices[nb_s + 1]
            prev, current = current, (n1 if n0 == prev else n0)

        # Only keep closed loops with at least 3 vertices
        if current == start and len(loop_compact) >= 3:
            compact_tensor = torch.tensor(loop_compact, dtype=torch.long)
            # Map back to original vertex indices, return on original device
            loops.append(unique_verts_cpu[compact_tensor].to(device))

    return loops


_EMPTY_STATS = {
    "n_holes_detected": 0,
    "n_holes_filled": 0,
    "n_holes_skipped": 0,
    "n_faces_added": 0,
    "n_points_added": 0,
}


def fill_holes(
    mesh: "Mesh",
    max_hole_edges: int = 10,
) -> tuple["Mesh", dict[str, int]]:
    """Fill holes bounded by boundary loops (2D manifolds only).

    Detects boundary loops (connected components of boundary edges that form
    closed cycles) and triangulates each loop independently using fan
    triangulation from a centroid vertex inserted at the loop's center.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must be a 2D manifold, i.e., a triangle mesh).
    max_hole_edges : int
        Maximum number of edges in a hole to fill. Holes larger than this
        are left open. This prevents accidentally filling large openings
        that may be intentional geometry.

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (filled_mesh, stats_dict) where stats_dict contains:

        - ``"n_holes_detected"``: Total number of boundary loops found.
        - ``"n_holes_filled"``: Number of holes actually filled (those with
          <= max_hole_edges edges).
        - ``"n_holes_skipped"``: Number of holes skipped (too large).
        - ``"n_faces_added"``: Total number of new triangular faces added.
        - ``"n_points_added"``: Total number of new centroid points added.

    Raises
    ------
    ValueError
        If mesh is not a 2D manifold.

    Example
    -------
    >>> from physicsnemo.mesh.primitives.surfaces import cylinder_open
    >>> mesh = cylinder_open.load()
    >>> mesh_filled, stats = fill_holes(mesh, max_hole_edges=40)
    >>> assert stats["n_holes_detected"] >= 0
    """
    if mesh.n_manifold_dims != 2:
        raise ValueError(
            f"Hole filling only implemented for 2D manifolds (triangle meshes). "
            f"Got {mesh.n_manifold_dims=}."
        )

    if mesh.n_cells == 0:
        return mesh, dict(_EMPTY_STATS)

    device = mesh.points.device

    ### Step 1: Find boundary edges via canonical detection
    from physicsnemo.mesh.boundaries import get_boundary_edges

    boundary_edges = get_boundary_edges(mesh)

    if len(boundary_edges) == 0:
        return mesh, dict(_EMPTY_STATS)

    ### Step 2: Trace boundary edges into disjoint loops
    loops = _trace_boundary_loops(boundary_edges)

    n_holes_detected = len(loops)
    if n_holes_detected == 0:
        return mesh, dict(_EMPTY_STATS)

    ### Step 3: Fill each loop that is small enough (vectorized fan triangulation)
    new_points_list: list[torch.Tensor] = []
    new_faces_list: list[torch.Tensor] = []
    n_holes_filled = 0
    n_holes_skipped = 0
    next_point_idx = mesh.n_points

    for loop_tensor in loops:
        n_loop_edges = len(loop_tensor)

        if n_loop_edges > max_hole_edges or n_loop_edges < 3:
            n_holes_skipped += 1
            continue

        # Compute centroid of the loop vertices
        loop_points = mesh.points[loop_tensor]
        centroid = loop_points.mean(dim=0)

        # Vectorized fan triangulation: each consecutive pair + centroid
        loop_next = loop_tensor.roll(-1)
        centroid_col = torch.full_like(loop_tensor, next_point_idx)
        fan_triangles = torch.stack([loop_tensor, loop_next, centroid_col], dim=1)

        new_faces_list.append(fan_triangles)
        new_points_list.append(centroid.unsqueeze(0))
        next_point_idx += 1
        n_holes_filled += 1

    ### Step 4: Assemble the filled mesh
    if n_holes_filled == 0:
        return mesh, {
            "n_holes_detected": n_holes_detected,
            "n_holes_filled": 0,
            "n_holes_skipped": n_holes_skipped,
            "n_faces_added": 0,
            "n_points_added": 0,
        }

    all_new_points = torch.cat(new_points_list, dim=0)  # (n_centroids, n_spatial_dims)
    all_new_faces = torch.cat(new_faces_list, dim=0)  # (n_new_faces, 3)
    n_new_points = all_new_points.shape[0]
    n_new_faces = all_new_faces.shape[0]

    new_points = torch.cat([mesh.points, all_new_points], dim=0)
    new_cells = torch.cat([mesh.cells, all_new_faces], dim=0)

    ### Step 5: Extend point_data and cell_data for the new elements
    def extend_point_data(tensor: torch.Tensor) -> torch.Tensor:
        """Extend a point_data tensor with NaN (float) or 0 (int) for new centroids."""
        if tensor.shape[0] != mesh.n_points:
            return tensor
        fill = float("nan") if tensor.dtype.is_floating_point else 0
        pad_shape = (n_new_points, *tensor.shape[1:])
        pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=device)
        return torch.cat([tensor, pad], dim=0)

    def extend_cell_data(tensor: torch.Tensor) -> torch.Tensor:
        """Extend a cell_data tensor with NaN (float) or 0 (int) for new faces."""
        if tensor.shape[0] != mesh.n_cells:
            return tensor
        fill = float("nan") if tensor.dtype.is_floating_point else 0
        pad_shape = (n_new_faces, *tensor.shape[1:])
        pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=device)
        return torch.cat([tensor, pad], dim=0)

    new_point_data = mesh.point_data.exclude(CACHE_KEY).apply(extend_point_data)
    new_cell_data = mesh.cell_data.exclude(CACHE_KEY).apply(extend_cell_data)

    from physicsnemo.mesh.mesh import Mesh

    filled_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data.clone(),
    )

    stats = {
        "n_holes_detected": n_holes_detected,
        "n_holes_filled": n_holes_filled,
        "n_holes_skipped": n_holes_skipped,
        "n_faces_added": n_new_faces,
        "n_points_added": n_new_points,
    }

    return filled_mesh, stats
