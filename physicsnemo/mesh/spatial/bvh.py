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

"""Bounding Volume Hierarchy (BVH) for efficient spatial queries.

This module implements a GPU-compatible BVH using flat array storage for efficient
traversal on both CPU and GPU. The BVH enables O(log N) query time for finding
which cells contain query points, compared to O(N) for brute-force search.
"""

from typing import TYPE_CHECKING

import torch
from tensordict import tensorclass

from physicsnemo.mesh.neighbors._adjacency import Adjacency, build_adjacency_from_pairs

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


@tensorclass
class BVH:
    """Bounding Volume Hierarchy for efficient spatial queries.

    The BVH is stored as flat tensors for GPU compatibility, avoiding pointer-based
    tree structures. Each internal node has exactly two children (binary tree).

    This is a pure PyTorch implementation that works on both CPU and GPU without
    additional dependencies. For extremely performance-critical applications with
    large meshes, consider NVIDIA Warp's BVH implementation
    (https://nvidia.github.io/warp/api_reference/_generated/warp.Bvh.html) which
    provides GPU-accelerated BVH with SAH, median, and LBVH construction algorithms,
    as well as native support for closest-point and ray-cast queries.

    Attributes
    ----------
    node_aabb_min : torch.Tensor
        Minimum corner of axis-aligned bounding box for each node,
        shape (n_nodes, n_spatial_dims)
    node_aabb_max : torch.Tensor
        Maximum corner of AABB for each node,
        shape (n_nodes, n_spatial_dims)
    node_left_child : torch.Tensor
        Index of left child for each internal node,
        shape (n_nodes,). Value is -1 for leaf nodes.
    node_right_child : torch.Tensor
        Index of right child for each internal node,
        shape (n_nodes,). Value is -1 for leaf nodes.
    node_cell_idx : torch.Tensor
        Cell index for leaf nodes, shape (n_nodes,).
        Value is -1 for internal nodes.

    Examples
    --------
    >>> # Build BVH from mesh
    >>> bvh = BVH.from_mesh(mesh)
    >>>
    >>> # Find candidate cells for query points
    >>> query_points = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    >>> candidates = bvh.find_candidate_cells(query_points)
    """

    node_aabb_min: torch.Tensor  # shape: (n_nodes, n_spatial_dims)
    node_aabb_max: torch.Tensor  # shape: (n_nodes, n_spatial_dims)
    node_left_child: torch.Tensor  # shape: (n_nodes,), dtype: int64
    node_right_child: torch.Tensor  # shape: (n_nodes,), dtype: int64
    node_cell_idx: torch.Tensor  # shape: (n_nodes,), dtype: int64

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the BVH."""
        return self.node_aabb_min.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        """Dimensionality of the spatial space."""
        return self.node_aabb_min.shape[1]

    @property
    def device(self) -> torch.device:
        """Device where BVH tensors are stored."""
        return self.node_aabb_min.device

    @classmethod
    def from_mesh(cls, mesh: "Mesh") -> "BVH":
        """Construct a BVH from a mesh.

        Uses top-down construction with median split along the longest axis
        of each node's bounding box. Cells are sorted by centroid along the
        split axis, then divided at the median into left and right children.

        Parameters
        ----------
        mesh : Mesh
            The mesh to build BVH for

        Returns
        -------
        BVH
            Constructed BVH ready for queries
        """
        ### Compute bounding box for each cell
        cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_vertices, n_spatial_dims)
        cell_aabb_min = cell_vertices.min(dim=1).values  # (n_cells, n_spatial_dims)
        cell_aabb_max = cell_vertices.max(dim=1).values  # (n_cells, n_spatial_dims)

        ### Compute cell centroids for Morton code-based ordering
        cell_centroids = cell_vertices.mean(dim=1)  # (n_cells, n_spatial_dims)

        ### Build BVH using top-down construction
        n_cells = mesh.n_cells

        ### Initialize node storage (worst case: 2*n_cells - 1 nodes for binary tree)
        max_nodes = 2 * n_cells - 1
        node_aabb_min = torch.zeros(
            (max_nodes, mesh.n_spatial_dims),
            dtype=mesh.points.dtype,
            device=mesh.points.device,
        )
        node_aabb_max = torch.zeros_like(node_aabb_min)
        node_left_child = torch.full(
            (max_nodes,), -1, dtype=torch.long, device=mesh.points.device
        )
        node_right_child = torch.full(
            (max_nodes,), -1, dtype=torch.long, device=mesh.points.device
        )
        node_cell_idx = torch.full(
            (max_nodes,), -1, dtype=torch.long, device=mesh.points.device
        )

        ### Build tree recursively (on CPU for now, move to GPU after)
        # Start with all cells
        cell_indices = torch.arange(n_cells, device=mesh.points.device)

        node_counter = [0]  # Use list to make it mutable in nested function

        def build_node(indices: torch.Tensor) -> int:
            """Recursively build BVH node.

            Parameters
            ----------
            indices : torch.Tensor
                Indices of cells to include in this subtree

            Returns
            -------
            int
                Index of the created node
            """
            node_idx = node_counter[0]
            node_counter[0] += 1

            ### Compute bounding box for this node
            node_aabb_min[node_idx] = cell_aabb_min[indices].min(dim=0).values
            node_aabb_max[node_idx] = cell_aabb_max[indices].max(dim=0).values

            ### Base case: single cell (leaf node)
            if len(indices) == 1:
                node_cell_idx[node_idx] = indices[0]
                return node_idx

            ### Recursive case: split and build children
            # Choose split axis as the dimension with largest extent
            extent = node_aabb_max[node_idx] - node_aabb_min[node_idx]
            split_axis = extent.argmax().item()

            # Sort cells by centroid along split axis
            centroids_along_axis = cell_centroids[indices, split_axis]
            sorted_indices_rel = centroids_along_axis.argsort()
            sorted_indices = indices[sorted_indices_rel]

            # Split at median
            mid = len(sorted_indices) // 2
            left_indices = sorted_indices[:mid]
            right_indices = sorted_indices[mid:]

            ### Build children
            left_child_idx = build_node(left_indices)
            right_child_idx = build_node(right_indices)

            node_left_child[node_idx] = left_child_idx
            node_right_child[node_idx] = right_child_idx

            return node_idx

        ### Build the tree starting from root
        build_node(cell_indices)

        ### Trim unused node storage
        n_nodes_used = node_counter[0]
        node_aabb_min = node_aabb_min[:n_nodes_used]
        node_aabb_max = node_aabb_max[:n_nodes_used]
        node_left_child = node_left_child[:n_nodes_used]
        node_right_child = node_right_child[:n_nodes_used]
        node_cell_idx = node_cell_idx[:n_nodes_used]

        return cls(
            node_aabb_min=node_aabb_min,
            node_aabb_max=node_aabb_max,
            node_left_child=node_left_child,
            node_right_child=node_right_child,
            node_cell_idx=node_cell_idx,
            batch_size=torch.Size([n_nodes_used]),
        )

    def point_in_aabb(
        self,
        points: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> torch.Tensor:
        """Test if points are inside axis-aligned bounding boxes.

        Parameters
        ----------
        points : torch.Tensor
            Query points, shape (n_points, n_spatial_dims)
        aabb_min : torch.Tensor
            Minimum corners, shape (n_boxes, n_spatial_dims)
        aabb_max : torch.Tensor
            Maximum corners, shape (n_boxes, n_spatial_dims)

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (n_points, n_boxes) indicating containment
        """
        # Broadcast and compare
        # points: (n_points, 1, n_spatial_dims)
        # aabb_min: (1, n_boxes, n_spatial_dims)
        points_exp = points.unsqueeze(1)
        aabb_min_exp = aabb_min.unsqueeze(0)
        aabb_max_exp = aabb_max.unsqueeze(0)

        # Point is inside if all coordinates are within bounds
        inside = ((points_exp >= aabb_min_exp) & (points_exp <= aabb_max_exp)).all(
            dim=2
        )
        return inside

    def find_candidate_cells(
        self,
        query_points: torch.Tensor,
        max_candidates_per_point: int | None = 32,
        aabb_tolerance: float = 1e-6,
    ) -> Adjacency:
        """Find candidate cells that might contain each query point.

        Uses batched iterative BVH traversal where all queries are processed
        simultaneously in a vectorized manner.

        Parameters
        ----------
        query_points : torch.Tensor
            Points to query, shape (n_queries, n_spatial_dims)
        max_candidates_per_point : int | None, optional
            Maximum number of candidate cells to return
            per query point. Prevents memory explosion for degenerate cases.
            If None, no limit is applied.
        aabb_tolerance : float, optional
            Tolerance for AABB intersection test. Important for
            degenerate cells (e.g., cells with duplicate vertices).

        Returns
        -------
        Adjacency
            Adjacency object where candidates for query i are at
            ``result.indices[result.offsets[i]:result.offsets[i+1]]``.
            Use ``result.to_list()`` for a list-of-tensors representation.

        Notes
        -----
        Performance:
            - Complexity: O(M log N) where M = queries, N = cells
            - All AABB tests and tree operations are fully vectorized across queries
            - No Python-level loops over query points
            - Returns GPU-native Adjacency - no CPU sync required

        BVH traversal could potentially be accelerated with custom CUDA kernels,
        but this adds significant complexity. The current implementation provides
        excellent performance for most use cases.
        """
        ### Batched BVH traversal implementation
        n_queries = query_points.shape[0]

        ### Initialize work queue with (query_idx, node_idx) pairs
        # All queries start at the root node (index 0)
        current_query_indices = torch.arange(n_queries, device=self.device)
        current_node_indices = torch.zeros(
            n_queries, dtype=torch.long, device=self.device
        )

        ### Track how many candidates we've found per query
        candidates_count = torch.zeros(n_queries, dtype=torch.long, device=self.device)

        ### Storage for all (query_idx, cell_idx) pairs found during traversal
        all_query_indices_list = []
        all_cell_indices_list = []

        ### Iterative traversal processing all active (query, node) pairs in parallel
        while len(current_query_indices) > 0:
            ### Vectorized AABB intersection test for all active pairs
            batch_query_points = query_points[
                current_query_indices
            ]  # (n_active, n_spatial_dims)
            batch_aabb_min = self.node_aabb_min[
                current_node_indices
            ]  # (n_active, n_spatial_dims)
            batch_aabb_max = self.node_aabb_max[
                current_node_indices
            ]  # (n_active, n_spatial_dims)

            # Check containment with tolerance for all pairs simultaneously
            inside = (
                (batch_query_points >= batch_aabb_min - aabb_tolerance)
                & (batch_query_points <= batch_aabb_max + aabb_tolerance)
            ).all(dim=1)  # (n_active,)

            ### Filter to only intersecting pairs
            intersecting_query_indices = current_query_indices[inside]
            intersecting_node_indices = current_node_indices[inside]

            if len(intersecting_query_indices) == 0:
                break  # No more intersections, done

            ### Separate leaf nodes from internal nodes
            cell_indices = self.node_cell_idx[intersecting_node_indices]
            is_leaf = cell_indices >= 0

            ### Handle leaf nodes: record candidates
            leaf_query_indices = intersecting_query_indices[is_leaf]
            leaf_cell_indices = cell_indices[is_leaf]

            if len(leaf_query_indices) > 0:
                all_query_indices_list.append(leaf_query_indices)
                all_cell_indices_list.append(leaf_cell_indices)

                # Update candidate counts for these queries
                # Use scatter_add to accumulate counts
                candidates_count.scatter_add_(
                    0,
                    leaf_query_indices,
                    torch.ones_like(leaf_query_indices),
                )

            ### Handle internal nodes: expand to children
            internal_query_indices = intersecting_query_indices[~is_leaf]
            internal_node_indices = intersecting_node_indices[~is_leaf]

            # Filter out queries that have already reached max_candidates
            under_limit = (
                candidates_count[internal_query_indices] < max_candidates_per_point
            )
            internal_query_indices = internal_query_indices[under_limit]
            internal_node_indices = internal_node_indices[under_limit]

            if len(internal_query_indices) == 0:
                break  # All remaining queries have hit their candidate limit

            # Get children for internal nodes
            left_children = self.node_left_child[internal_node_indices]
            right_children = self.node_right_child[internal_node_indices]

            # Create work queue entries for left children (where valid)
            valid_left = left_children >= 0
            left_query_indices = internal_query_indices[valid_left]
            left_node_indices = left_children[valid_left]

            # Create work queue entries for right children (where valid)
            valid_right = right_children >= 0
            right_query_indices = internal_query_indices[valid_right]
            right_node_indices = right_children[valid_right]

            # Combine for next iteration
            if len(left_query_indices) > 0 or len(right_query_indices) > 0:
                current_query_indices = torch.cat(
                    [left_query_indices, right_query_indices]
                )
                current_node_indices = torch.cat(
                    [left_node_indices, right_node_indices]
                )
            else:
                break

        ### Build Adjacency from (query_idx, cell_idx) pairs
        if len(all_query_indices_list) > 0:
            all_query_indices = torch.cat(all_query_indices_list)
            all_cell_indices = torch.cat(all_cell_indices_list)
        else:
            all_query_indices = torch.tensor([], dtype=torch.long, device=self.device)
            all_cell_indices = torch.tensor([], dtype=torch.long, device=self.device)

        adjacency = build_adjacency_from_pairs(
            source_indices=all_query_indices,
            target_indices=all_cell_indices,
            n_sources=n_queries,
        )

        return adjacency.truncate_per_source(max_candidates_per_point)
