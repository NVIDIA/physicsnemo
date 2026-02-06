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

"""Topology generation for mesh subdivision.

This module handles the combinatorial aspects of subdivision: computing
subdivision patterns and generating child cell connectivity.

Edge extraction is provided by
:func:`physicsnemo.mesh.utilities._topology.extract_unique_edges`,
re-exported here for backwards compatibility.
"""

import torch

from physicsnemo.mesh.utilities._topology import extract_unique_edges  # noqa: F401


def get_subdivision_pattern(n_manifold_dims: int) -> torch.Tensor:
    """Get the subdivision pattern for splitting an n-simplex.

    Returns a pattern tensor that encodes how to split an n-simplex into
    2^n child simplices using edge midpoints.

    The pattern uses a specific vertex indexing scheme:
    - Indices 0 to n: original vertices
    - Indices n+1 to n+C(n+1,2): edge midpoints, indexed by edge

    For each n-simplex:
    - n+1 original vertices
    - C(n+1, 2) edges, each gets a midpoint
    - Splits into 2^n child simplices

    Parameters
    ----------
    n_manifold_dims : int
        Manifold dimension of the mesh.

    Returns
    -------
    torch.Tensor
        Pattern tensor of shape (n_children, n_vertices_per_child) where:
        - n_children = 2^n_manifold_dims
        - n_vertices_per_child = n_manifold_dims + 1

        Each row specifies vertex indices for one child simplex.
        Indices reference: [v0, v1, ..., vn, e01, e02, ..., e(n-1,n)]
        where v_i are original vertices and e_ij are edge midpoints.

    Examples
    --------
        For a triangle (n=2):
        - 3 original vertices: v0, v1, v2
        - 3 edge midpoints: e01, e12, e20
        - Indexing: [v0=0, v1=1, v2=2, e01=3, e12=4, e20=5]
        - 4 children: [v0, e01, e20], [v1, e12, e01], [v2, e20, e12], [e01, e12, e20]
    """
    if n_manifold_dims == 1:
        ### 1-simplex (edge) splits into 2 edges
        # Vertices: [v0, v1, e01]
        # Children: [v0, e01], [e01, v1]
        return torch.tensor(
            [
                [0, 2],  # Child 0: v0 to e01
                [2, 1],  # Child 1: e01 to v1
            ],
            dtype=torch.int64,
        )

    elif n_manifold_dims == 2:
        ### 2-simplex (triangle) splits into 4 triangles
        # Vertices: [v0, v1, v2, e01, e12, e20]
        # Edge ordering from _generate_combination_indices(3, 2):
        # (0,1), (0,2), (1,2) -> indices 3, 4, 5
        return torch.tensor(
            [
                [0, 3, 4],  # Corner at v0: v0, e01, e02
                [1, 5, 3],  # Corner at v1: v1, e12, e01
                [2, 4, 5],  # Corner at v2: v2, e02, e12
                [3, 5, 4],  # Center: e01, e12, e02
            ],
            dtype=torch.int64,
        )

    elif n_manifold_dims == 3:
        ### 3-simplex (tetrahedron) splits into 8 tetrahedra
        # Vertices: [v0, v1, v2, v3, e01, e02, e03, e12, e13, e23]
        # Edge ordering from _generate_combination_indices(4, 2):
        # (0,1)=4, (0,2)=5, (0,3)=6, (1,2)=7, (1,3)=8, (2,3)=9
        return torch.tensor(
            [
                [0, 4, 5, 6],  # Corner at v0
                [1, 4, 7, 8],  # Corner at v1
                [2, 5, 7, 9],  # Corner at v2
                [3, 6, 8, 9],  # Corner at v3
                [4, 5, 7, 8],  # Inner tet 1
                [5, 6, 8, 9],  # Inner tet 2
                [4, 5, 6, 8],  # Inner tet 3
                [5, 7, 8, 9],  # Inner tet 4
            ],
            dtype=torch.int64,
        )

    else:
        raise NotImplementedError(
            f"Subdivision pattern not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D (tetrahedra)."
        )


def generate_child_cells(
    parent_cells: torch.Tensor,
    edge_inverse: torch.Tensor,
    n_original_points: int,
    subdivision_pattern: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate child cells from parent cells using subdivision pattern.

    This implementation is fully vectorized using torch operations, avoiding Python loops
    and GPU-CPU transfers for optimal performance on both CPU and GPU.

    Parameters
    ----------
    parent_cells : torch.Tensor
        Parent cell connectivity, shape (n_parent_cells, n_vertices_per_cell)
    edge_inverse : torch.Tensor
        Mapping from candidate edges to unique edge indices,
        shape (n_parent_cells * n_edges_per_cell,). This comes from torch.unique()
        called in extract_unique_edges().
    n_original_points : int
        Number of points in original mesh (before adding edge midpoints)
    subdivision_pattern : torch.Tensor
        Pattern from get_subdivision_pattern(),
        shape (n_children_per_parent, n_vertices_per_child)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (child_cells, parent_indices):
        - child_cells: Child cell connectivity,
          shape (n_parent_cells * n_children_per_parent, n_vertices_per_child)
        - parent_indices: Parent cell index for each child,
          shape (n_parent_cells * n_children_per_parent,)

    Algorithm
    ---------
    1. Reshape edge_inverse to (n_parent_cells, n_edges_per_cell) for per-cell lookup
    2. Build local_to_global mapping for ALL cells at once via concatenation
    3. Apply subdivision pattern using torch.gather to generate all children
    4. No Python loops, no GPU-CPU transfers - fully vectorized
    """
    n_parent_cells, n_vertices_per_cell = parent_cells.shape
    n_children_per_parent = subdivision_pattern.shape[0]
    device = parent_cells.device

    ### Compute number of edges per cell
    # For n-simplex: C(n+1, 2) = (n+1) * n / 2
    n_edges_per_cell = (n_vertices_per_cell * (n_vertices_per_cell - 1)) // 2

    ### Reshape edge_inverse to per-cell mapping
    # Shape: (n_parent_cells, n_edges_per_cell)
    # edge_inverse_per_cell[i, j] = global edge index for j-th edge of cell i
    edge_inverse_per_cell = edge_inverse.reshape(n_parent_cells, n_edges_per_cell)

    ### Build local_to_global mapping for ALL cells at once
    # Shape: (n_parent_cells, n_vertices_per_cell + n_edges_per_cell)
    # First n_vertices_per_cell entries: original vertices of the cell
    # Next n_edges_per_cell entries: global point indices of edge midpoints
    local_to_global = torch.cat(
        [
            parent_cells,  # (n_parent_cells, n_vertices_per_cell)
            n_original_points
            + edge_inverse_per_cell,  # (n_parent_cells, n_edges_per_cell)
        ],
        dim=1,
    )

    ### Apply subdivision pattern using torch.gather
    # Expand pattern to match batch dimension: (1, n_children, n_vertices) → (n_cells, n_children, n_vertices)
    pattern_expanded = subdivision_pattern.unsqueeze(0).expand(n_parent_cells, -1, -1)

    # Gather indices from local_to_global according to pattern
    # local_to_global: (n_cells, local_size)
    # pattern_expanded: (n_cells, n_children, n_vertices)
    # Result: (n_cells, n_children, n_vertices)
    child_cells = torch.gather(
        local_to_global.unsqueeze(1).expand(-1, n_children_per_parent, -1),
        dim=2,
        index=pattern_expanded,
    ).reshape(n_parent_cells * n_children_per_parent, n_vertices_per_cell)

    ### Generate parent indices for each child
    # Shape: (n_parent_cells * n_children_per_parent,)
    parent_indices = torch.arange(
        n_parent_cells,
        dtype=torch.int64,
        device=device,
    ).repeat_interleave(n_children_per_parent)

    return child_cells, parent_indices
