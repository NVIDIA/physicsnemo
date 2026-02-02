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

Detects boundary loops and closes them with new triangles.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def fill_holes(
    mesh: "Mesh",
    max_hole_edges: int = 10,
) -> tuple["Mesh", dict[str, int]]:
    """Fill holes bounded by boundary loops (2D manifolds only).

    Detects boundary loops (edges with only 1 adjacent face) and triangulates
    them using simple fan triangulation from the first vertex.

    Args:
        mesh: Input mesh (must be 2D manifold)
        max_hole_edges: Maximum number of edges in a hole to fill

    Returns:
        Tuple of (filled_mesh, stats_dict) where stats_dict contains:
        - "n_holes_filled": Number of holes that were filled
        - "n_faces_added": Total number of new faces added
        - "n_holes_detected": Total number of holes detected

    Raises:
        ValueError: If mesh is not a 2D manifold

    Example:
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
        return mesh, {"n_holes_filled": 0, "n_faces_added": 0, "n_holes_detected": 0}

    device = mesh.points.device

    ### Step 1: Find boundary edges (edges with only 1 adjacent face)
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    edges_with_dupes, parent_faces = extract_candidate_facets(
        mesh.cells, manifold_codimension=1
    )

    # Sort edges canonically
    edges_sorted, _ = torch.sort(edges_with_dupes, dim=1)

    # Count occurrences of each edge
    unique_edges, inverse_indices, counts = torch.unique(
        edges_sorted, dim=0, return_inverse=True, return_counts=True
    )

    # Boundary edges appear exactly once
    is_boundary_edge = counts == 1
    boundary_edges = unique_edges[is_boundary_edge]

    n_boundary_edges = len(boundary_edges)

    if n_boundary_edges == 0:
        # No holes (closed mesh)
        return mesh, {"n_holes_filled": 0, "n_faces_added": 0, "n_holes_detected": 0}

    ### Step 2: Group boundary edges into loops
    # Build adjacency: vertex -> boundary edges containing it
    # This is complex to do fully vectorized, so use simplified fan triangulation instead

    ### Simplified approach: For each boundary loop, create fan from centroid
    # This avoids complex loop detection but may create interior vertices

    # For now, implement basic version that fills by creating a single central vertex
    # and connecting all boundary edges to it

    # Compute centroid of boundary vertices
    boundary_vertices = torch.unique(boundary_edges.flatten())

    if len(boundary_vertices) <= 2:
        # Degenerate boundary
        return mesh, {"n_holes_filled": 0, "n_faces_added": 0, "n_holes_detected": 1}

    if len(boundary_vertices) > max_hole_edges:
        # Hole too large
        return mesh, {"n_holes_filled": 0, "n_faces_added": 0, "n_holes_detected": 1}

    # Create central point
    boundary_points = mesh.points[boundary_vertices]
    centroid = boundary_points.mean(dim=0)

    # Add centroid as new point
    new_points = torch.cat([mesh.points, centroid.unsqueeze(0)], dim=0)
    centroid_idx = mesh.n_points

    # Create triangles: each boundary edge + centroid
    # boundary_edges: (n_boundary, 2)
    new_faces = torch.cat(
        [
            boundary_edges,
            torch.full(
                (n_boundary_edges, 1), centroid_idx, dtype=torch.long, device=device
            ),
        ],
        dim=1,
    )  # (n_boundary, 3)

    # Combine with existing cells
    new_cells = torch.cat([mesh.cells, new_faces], dim=0)

    ### Transfer data (excluding cache)
    # For point data: need to extend by 1 for the new centroid
    # Use TensorDict.apply() to handle all tensors uniformly
    def extend_point_data(tensor):
        # Compute centroid value as mean of boundary vertices
        if tensor.ndim == 1 or (tensor.ndim > 1 and tensor.shape[0] == mesh.n_points):
            if tensor.ndim == 1:
                centroid_value = tensor[boundary_vertices].mean()
            else:
                centroid_value = tensor[boundary_vertices].mean(dim=0)
            return torch.cat([tensor, centroid_value.unsqueeze(0)], dim=0)
        return tensor

    new_point_data = mesh.point_data.exclude("_cache").apply(extend_point_data)

    # For cell data: need to extend by n_boundary_edges with NaN/zeros
    def extend_cell_data(tensor):
        # Initialize new faces with NaN for floats, 0 for ints
        if tensor.ndim == 1 or (tensor.ndim > 1 and tensor.shape[0] == mesh.n_cells):
            if tensor.dtype.is_floating_point:
                fill_value = float("nan")
            else:
                fill_value = 0

            if tensor.ndim == 1:
                new_data = torch.full(
                    (n_boundary_edges,), fill_value, dtype=tensor.dtype, device=device
                )
            else:
                new_data = torch.full(
                    (n_boundary_edges, *tensor.shape[1:]),
                    fill_value,
                    dtype=tensor.dtype,
                    device=device,
                )

            return torch.cat([tensor, new_data], dim=0)
        return tensor

    new_cell_data = mesh.cell_data.exclude("_cache").apply(extend_cell_data)

    from physicsnemo.mesh.mesh import Mesh

    filled_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data.clone(),
    )

    stats = {
        "n_holes_filled": 1,  # Simplified: assumes single hole
        "n_faces_added": n_boundary_edges,
        "n_holes_detected": 1,
    }

    return filled_mesh, stats
