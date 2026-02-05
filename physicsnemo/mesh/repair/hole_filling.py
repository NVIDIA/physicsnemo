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

"""Fill holes in triangle meshes.

Detects boundary loops (connected components of boundary edges) and closes
each loop independently with fan triangulation from a centroid vertex.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _trace_boundary_loops(
    boundary_edges: torch.Tensor,
) -> list[list[int]]:
    """Trace disjoint boundary loops from a set of boundary edges.

    Each boundary edge connects two vertices. A boundary loop is a connected
    cycle of boundary edges. This function identifies all such loops by
    walking along the boundary edge graph.

    Parameters
    ----------
    boundary_edges : torch.Tensor
        Boundary edges, shape (n_boundary_edges, 2). Each row is [v0, v1].

    Returns
    -------
    list[list[int]]
        Each inner list is the ordered sequence of vertex indices forming one
        boundary loop. The vertices are in traversal order (each consecutive
        pair shares a boundary edge, and the last connects back to the first).
    """
    if len(boundary_edges) == 0:
        return []

    ### Build adjacency: vertex -> set of neighboring boundary vertices
    edges_cpu = boundary_edges.cpu().tolist()
    adjacency: dict[int, list[int]] = {}
    for v0, v1 in edges_cpu:
        adjacency.setdefault(v0, []).append(v1)
        adjacency.setdefault(v1, []).append(v0)

    ### Walk the boundary graph to extract loops
    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    for start_vertex in adjacency:
        # Try to start a new loop from any vertex with unvisited edges
        for first_neighbor in adjacency[start_vertex]:
            edge_key = (min(start_vertex, first_neighbor), max(start_vertex, first_neighbor))
            if edge_key in visited_edges:
                continue

            # Walk the loop: start_vertex -> first_neighbor -> ...
            loop = [start_vertex]
            prev = start_vertex
            current = first_neighbor

            while current != start_vertex:
                visited_edges.add((min(prev, current), max(prev, current)))
                loop.append(current)

                # Find the next vertex: the neighbor of `current` that isn't `prev`
                neighbors = adjacency[current]
                next_vertex = None
                for nb in neighbors:
                    nb_edge_key = (min(current, nb), max(current, nb))
                    if nb != prev and nb_edge_key not in visited_edges:
                        next_vertex = nb
                        break

                if next_vertex is None:
                    # Dead end (non-manifold boundary) - abandon this walk
                    break
                prev = current
                current = next_vertex

            # Mark the closing edge as visited
            if current == start_vertex and len(loop) >= 3:
                visited_edges.add((min(prev, start_vertex), max(prev, start_vertex)))
                loops.append(loop)

    return loops


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
        return mesh, {
            "n_holes_detected": 0,
            "n_holes_filled": 0,
            "n_holes_skipped": 0,
            "n_faces_added": 0,
            "n_points_added": 0,
        }

    device = mesh.points.device

    ### Step 1: Find boundary edges (edges appearing in exactly 1 face)
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    edges_with_dupes, _parent_faces = extract_candidate_facets(
        mesh.cells, manifold_codimension=1
    )

    # Canonicalize edge ordering
    edges_sorted, _ = torch.sort(edges_with_dupes, dim=1)

    # Count occurrences of each canonical edge
    unique_edges, _inverse_indices, counts = torch.unique(
        edges_sorted, dim=0, return_inverse=True, return_counts=True
    )

    # Boundary edges appear exactly once
    boundary_edges = unique_edges[counts == 1]

    if len(boundary_edges) == 0:
        return mesh, {
            "n_holes_detected": 0,
            "n_holes_filled": 0,
            "n_holes_skipped": 0,
            "n_faces_added": 0,
            "n_points_added": 0,
        }

    ### Step 2: Trace boundary edges into disjoint loops
    loops = _trace_boundary_loops(boundary_edges)

    n_holes_detected = len(loops)
    if n_holes_detected == 0:
        return mesh, {
            "n_holes_detected": 0,
            "n_holes_filled": 0,
            "n_holes_skipped": 0,
            "n_faces_added": 0,
            "n_points_added": 0,
        }

    ### Step 3: Fill each loop that is small enough
    new_points_list: list[torch.Tensor] = []
    new_faces_list: list[torch.Tensor] = []
    n_holes_filled = 0
    n_holes_skipped = 0
    next_point_idx = mesh.n_points  # Index for the next new centroid point

    for loop_vertices in loops:
        n_loop_edges = len(loop_vertices)

        if n_loop_edges > max_hole_edges or n_loop_edges < 3:
            n_holes_skipped += 1
            continue

        # Compute centroid of the loop vertices
        loop_indices = torch.tensor(loop_vertices, dtype=torch.long, device=device)
        loop_points = mesh.points[loop_indices]
        centroid = loop_points.mean(dim=0)

        # Create fan triangles: for each consecutive pair of loop vertices,
        # form a triangle with the centroid
        for i in range(n_loop_edges):
            v0 = loop_vertices[i]
            v1 = loop_vertices[(i + 1) % n_loop_edges]
            new_faces_list.append(
                torch.tensor([[v0, v1, next_point_idx]], dtype=torch.long, device=device)
            )

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
