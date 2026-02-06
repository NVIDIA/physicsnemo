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

"""Fix face orientation for consistent normals.

Ensures all faces in a mesh have consistent orientation so normals point
in the same general direction.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.neighbors._adjacency import Adjacency


def _gather_unoriented_neighbors(
    front: torch.Tensor,
    adjacency: "Adjacency",
    is_oriented: torch.Tensor,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a BFS front by one level, returning only unoriented neighbors.

    Given the current front of face indices and a CSR adjacency structure,
    gathers all neighbors, filters to those not yet oriented, and tracks
    which front face discovered each neighbor.

    Parameters
    ----------
    front : torch.Tensor
        Face indices in the current BFS front. Shape (n_front,).
    adjacency : Adjacency
        CSR cell-to-cell adjacency.
    is_oriented : torch.Tensor
        Boolean mask of already-oriented faces. Shape (n_cells,).
    max_neighbors : int
        Upper bound on neighbors per face (n_manifold_dims + 1 for simplices).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (next_front, parent_faces) where next_front contains the unoriented
        neighbor indices and parent_faces[k] is the front face that discovered
        next_front[k]. Both are empty tensors when no unoriented neighbors exist.
    """
    device = front.device

    if max_neighbors == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty

    ### Gather all neighbors for entire front via the CSR structure
    offsets_start = adjacency.offsets[front]  # (n_front,)
    offsets_end = adjacency.offsets[front + 1]  # (n_front,)
    neighbor_counts = offsets_end - offsets_start  # (n_front,)

    # Build padded gather indices using offset + arange pattern.
    # Static upper bound avoids .item() graph break inside the BFS loop.
    # Shape: (n_front, max_neighbors)
    neighbor_offsets = torch.arange(max_neighbors, device=device, dtype=torch.long)
    gather_indices = offsets_start.unsqueeze(1) + neighbor_offsets.unsqueeze(0)

    # Mask for valid neighbors (within each face's actual neighbor count)
    # Shape: (n_front, max_neighbors)
    valid_mask = neighbor_offsets.unsqueeze(0) < neighbor_counts.unsqueeze(1)

    # Gather all neighbors (use 0 for out-of-bounds, filtered out below)
    # Shape: (n_front, max_neighbors)
    gather_indices_safe = torch.where(
        valid_mask, gather_indices, torch.zeros_like(gather_indices)
    )
    all_neighbors = adjacency.indices[gather_indices_safe]

    ### Filter to unoriented neighbors only
    keep_mask = valid_mask & ~is_oriented[all_neighbors]

    if not keep_mask.any():
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty

    next_front = all_neighbors[keep_mask]  # (n_next,)

    # Track which front face discovered each neighbor
    # Shape: (n_front, max_neighbors) -> (n_next,)
    parent_faces = front.unsqueeze(1).expand(-1, max_neighbors)[keep_mask]

    return next_front, parent_faces


def _propagate_flip_from_parents(
    children: torch.Tensor,
    parents: torch.Tensor,
    cell_normals: torch.Tensor,
    should_flip: torch.Tensor,
) -> None:
    """Determine flip flags for children based on normal agreement with parents.

    For each child face, compares its stored normal against the effective normal
    of its parent (accounting for any prior flip). If the dot product is negative,
    the child needs to be flipped for consistent orientation.

    Parameters
    ----------
    children : torch.Tensor
        Face indices of newly discovered BFS neighbors. Shape (n_next,).
    parents : torch.Tensor
        Face indices of the parent that discovered each child. Shape (n_next,).
    cell_normals : torch.Tensor
        Per-face normals from the mesh. Shape (n_cells, n_spatial_dims).
    should_flip : torch.Tensor
        Boolean mask updated in-place. Shape (n_cells,).
    """
    parent_normals = cell_normals[parents]  # (n_next, 3)

    # Account for parents that were themselves flipped in a prior BFS level.
    # Their effective normal is the negation of the stored (original) normal.
    parent_flip_sign = torch.where(should_flip[parents], -1.0, 1.0).unsqueeze(
        -1
    )  # (n_next, 1) for broadcasting over spatial dims
    parent_normals = parent_normals * parent_flip_sign

    child_normals = cell_normals[children]  # (n_next, 3)

    # Negative dot product means opposite orientation -> needs flip
    dots = (child_normals * parent_normals).sum(dim=-1)
    should_flip[children] = dots < 0


def fix_orientation(
    mesh: "Mesh",
) -> tuple["Mesh", dict[str, int]]:
    """Orient all faces consistently (2D manifolds in 3D only).

    Uses graph propagation to ensure adjacent faces have consistent orientation.
    Two faces sharing an edge should have opposite vertex ordering along that edge.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must be 2D manifold in 3D space)

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (oriented_mesh, stats_dict) where stats_dict contains:
        - "n_faces_flipped": Number of faces that were flipped
        - "n_components": Number of connected components found
        - "largest_component_size": Size of largest component

    Raises
    ------
    ValueError
        If mesh is not a 2D manifold in 3D

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=2)
    >>> mesh_oriented, stats = fix_orientation(mesh)
    >>> assert "n_faces_flipped" in stats
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

    ### Step 2: Propagate orientation using iterative BFS (flat state machine)
    is_oriented = torch.zeros(n_cells, dtype=torch.bool, device=device)
    should_flip = torch.zeros(n_cells, dtype=torch.bool, device=device)
    component_id = torch.full((n_cells,), -1, dtype=torch.long, device=device)

    max_neighbors = mesh.n_manifold_dims + 1
    current_front = torch.empty(0, device=device, dtype=torch.long)
    component_sizes: list[int] = []

    while True:
        ### Phase A: If front exhausted, seed next component or terminate
        if len(current_front) == 0:
            unoriented = torch.where(~is_oriented)[0]
            if len(unoriented) == 0:
                break
            seed = unoriented[0]
            is_oriented[seed] = True
            component_id[seed] = len(component_sizes)
            current_front = seed.unsqueeze(0)
            component_sizes.append(1)
            continue

        ### Phase B: Expand BFS front by one level
        next_front, parent_faces = _gather_unoriented_neighbors(
            current_front, adjacency, is_oriented, max_neighbors
        )

        if len(next_front) == 0:
            current_front = next_front  # Triggers re-seeding on next iteration
            continue

        is_oriented[next_front] = True
        component_id[next_front] = len(component_sizes) - 1
        component_sizes[-1] += len(next_front)

        if mesh.n_spatial_dims == 3 and mesh.codimension == 1:
            _propagate_flip_from_parents(
                next_front, parent_faces, mesh.cell_normals, should_flip
            )

        current_front = next_front

    n_components = len(component_sizes)
    largest_component_size = max(component_sizes, default=0)

    ### Step 3: Apply flips
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
            point_data=mesh.point_data.exclude(CACHE_KEY).clone(),
            cell_data=mesh.cell_data.exclude(CACHE_KEY).clone(),
            global_data=mesh.global_data.clone(),
        )
    else:
        oriented_mesh = mesh

    stats = {
        "n_faces_flipped": n_flipped,
        "n_components": n_components,
        "largest_component_size": largest_component_size,
    }

    return oriented_mesh, stats
