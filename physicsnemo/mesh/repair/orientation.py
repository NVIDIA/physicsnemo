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

"""Fix face orientation for consistent normals.

Ensures all faces in a mesh have consistent orientation so normals point
in the same general direction.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def fix_orientation(
    mesh: "Mesh",
) -> tuple["Mesh", dict[str, int]]:
    """Orient all faces consistently (2D manifolds in 3D only).

    Uses graph propagation to ensure adjacent faces have consistent orientation.
    Two faces sharing an edge should have opposite vertex ordering along that edge.

    Args:
        mesh: Input mesh (must be 2D manifold in 3D space)

    Returns:
        Tuple of (oriented_mesh, stats_dict) where stats_dict contains:
        - "n_faces_flipped": Number of faces that were flipped
        - "n_components": Number of connected components found
        - "largest_component_size": Size of largest component

    Raises:
        ValueError: If mesh is not a 2D manifold in 3D

    Example:
        >>> mesh_oriented, stats = fix_orientation(mesh)
        >>> print(f"Flipped {stats['n_faces_flipped']} faces")
    """
    if mesh.n_manifold_dims != 2:
        raise ValueError(
            f"Orientation fixing only implemented for 2D manifolds (triangles). "
            f"Got {mesh.n_manifold_dims=}."
        )

    if mesh.n_cells == 0:
        return mesh, {
            "n_faces_flipped": 0,
            "n_components": 0,
            "largest_component_size": 0,
        }

    device = mesh.points.device
    n_cells = mesh.n_cells

    ### Step 1: Build face adjacency graph via shared edges
    from physicsnemo.mesh.neighbors import get_cell_to_cells_adjacency

    adjacency = get_cell_to_cells_adjacency(mesh, adjacency_codimension=1)

    ### Step 2: Extract edges to determine shared edge orientation
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    edges_with_dupes, parent_faces = extract_candidate_facets(
        mesh.cells, manifold_codimension=1
    )

    # For each edge, determine if adjacent faces have consistent orientation
    # Two faces are consistent if they traverse the shared edge in opposite directions

    ### Step 3: Propagate orientation using iterative flooding (vectorized)
    # Track which faces have been oriented
    is_oriented = torch.zeros(n_cells, dtype=torch.bool, device=device)
    should_flip = torch.zeros(n_cells, dtype=torch.bool, device=device)
    component_id = torch.full((n_cells,), -1, dtype=torch.long, device=device)

    n_components = 0

    # Process each connected component using iterative propagation
    while not torch.all(is_oriented):
        # Find an unoriented face to start from
        unoriented_indices = torch.where(~is_oriented)[0]
        if len(unoriented_indices) == 0:
            break

        start_face = unoriented_indices[0]

        # Initialize component
        is_oriented[start_face] = True
        component_id[start_face] = n_components
        current_front = torch.tensor([start_face], device=device, dtype=torch.long)

        component_size = 1

        # Iteratively expand front until no more neighbors (fully vectorized)
        for iteration in range(n_cells):  # Max iterations = n_cells (diameter of graph)
            if len(current_front) == 0:
                break

            ### Gather all neighbors for entire front at once
            # Compute neighbor counts for each face in front
            offsets_start = adjacency.offsets[current_front]  # (n_front,)
            offsets_end = adjacency.offsets[current_front + 1]  # (n_front,)
            neighbor_counts = offsets_end - offsets_start  # (n_front,)

            # Build gather indices for all neighbors using broadcasting
            # Shape: (n_front, max_neighbors) - padded with -1 for ragged structure
            max_neighbors = (
                neighbor_counts.max().item() if len(neighbor_counts) > 0 else 0
            )

            if max_neighbors == 0:
                break

            # Generate indices using offset + arange pattern
            # Shape: (n_front, max_neighbors)
            neighbor_offsets = torch.arange(
                max_neighbors, device=device, dtype=torch.long
            )
            gather_indices = offsets_start.unsqueeze(1) + neighbor_offsets.unsqueeze(0)

            # Mask for valid neighbors (within each face's neighbor count)
            # Shape: (n_front, max_neighbors)
            valid_mask = neighbor_offsets.unsqueeze(0) < neighbor_counts.unsqueeze(1)

            # Gather all neighbors (use 0 for invalid, will filter out)
            # Shape: (n_front, max_neighbors)
            gather_indices_safe = torch.where(
                valid_mask,
                gather_indices,
                torch.zeros_like(gather_indices),
            )
            all_neighbors_padded = adjacency.indices[gather_indices_safe]

            # Filter to unoriented neighbors only
            # Shape: (n_front, max_neighbors)
            is_unoriented = ~is_oriented[all_neighbors_padded]
            keep_mask = valid_mask & is_unoriented

            # Check if we have any unoriented neighbors
            if not keep_mask.any():
                break

            # Flatten and extract valid neighbors
            next_front = all_neighbors_padded[keep_mask]  # (n_next,)

            # For each neighbor, track which parent face it came from
            # Shape: (n_front, max_neighbors) -> (n_next,)
            parent_faces_expanded = current_front.unsqueeze(1).expand(-1, max_neighbors)
            parent_faces_for_neighbors = parent_faces_expanded[keep_mask]  # (n_next,)

            # Mark as oriented
            is_oriented[next_front] = True
            component_id[next_front] = n_components
            component_size += len(next_front)

            # Determine orientation using normals (vectorized over entire next_front)
            if mesh.n_spatial_dims == 3 and mesh.codimension == 1:
                parent_normals = mesh.cell_normals[parent_faces_for_neighbors]
                neighbor_normals = mesh.cell_normals[next_front]

                # Dot product: negative means opposite orientation
                dots = (neighbor_normals * parent_normals).sum(dim=-1)
                should_flip[next_front] = dots < 0

            current_front = next_front

        if n_components == 0:
            largest_component_size = component_size

        n_components += 1

    ### Step 4: Apply flips
    n_flipped = should_flip.sum().item()

    if n_flipped > 0:
        # Flip faces by reversing vertex order
        new_cells = mesh.cells.clone()

        # For triangles: swap vertices 1 and 2 (keeps vertex 0, reverses orientation)
        new_cells[should_flip, 1], new_cells[should_flip, 2] = (
            mesh.cells[should_flip, 2],
            mesh.cells[should_flip, 1],
        )

        from physicsnemo.mesh.mesh import Mesh

        oriented_mesh = Mesh(
            points=mesh.points,
            cells=new_cells,
            point_data=mesh.point_data.exclude("_cache").clone(),
            cell_data=mesh.cell_data.exclude("_cache").clone(),
            global_data=mesh.global_data.clone(),
        )
    else:
        oriented_mesh = mesh

    stats = {
        "n_faces_flipped": n_flipped,
        "n_components": n_components,
        "largest_component_size": largest_component_size if n_components > 0 else 0,
    }

    return oriented_mesh, stats
