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

"""Direct unit tests for the shared morton-LBVH topology builder.

``build_lbvh_topology`` is the single source of truth for the node topology of
both :class:`~physicsnemo.mesh.spatial.bvh.BVH` and
:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree`. These tests pin
its structural invariants (leaf partition, child links, range consistency,
node-count bounds) and the cross-structure contract that both consumers build
the identical tree for the same ``(n_items, leaf_size)``.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.spatial._lbvh import build_lbvh_topology
from physicsnemo.mesh.spatial.bvh import BVH
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

DEVICE = torch.device("cpu")


class TestBuilderInvariants:
    @pytest.mark.parametrize("leaf_size", [1, 3, 8])
    @pytest.mark.parametrize("n_items", [1, 2, 7, 100])
    def test_topology_invariants(self, n_items, leaf_size):
        topo = build_lbvh_topology(n_items, leaf_size, DEVICE)

        assert topo.node_count <= topo.max_nodes
        if leaf_size == 1:
            # With leaf_size=1 every leaf holds exactly one item, so the
            # full-binary-tree bound is tight.
            assert topo.node_count == topo.max_nodes == 2 * n_items - 1

        ### Leaf segments partition [0, n_items) with valid occupancy.
        sizes = topo.leaf_sizes
        assert (sizes >= 1).all()
        assert (sizes <= leaf_size).all()
        assert int(sizes.sum()) == n_items
        order = torch.argsort(topo.leaf_starts)
        starts_sorted = topo.leaf_starts[order]
        sizes_sorted = sizes[order]
        assert starts_sorted[0].item() == 0
        assert torch.equal(starts_sorted[1:], (starts_sorted + sizes_sorted)[:-1]), (
            "leaf segments must be contiguous and non-overlapping"
        )

        ### Compacted leaf arrays agree with the per-node buffers.
        used = slice(0, topo.node_count)
        leaf_start = topo.leaf_start[used]
        leaf_count = topo.leaf_count[used]
        assert torch.equal(leaf_start[topo.leaf_node_ids], topo.leaf_starts)
        assert torch.equal(leaf_count[topo.leaf_node_ids], topo.leaf_sizes)

        ### Leaves have no children; internal nodes have both.
        left = topo.left_child[used]
        right = topo.right_child[used]
        is_leaf = leaf_start >= 0
        assert (left[is_leaf] == -1).all()
        assert (right[is_leaf] == -1).all()
        assert (left[~is_leaf] >= 0).all()
        assert (right[~is_leaf] >= 0).all()
        assert (left < topo.node_count).all()
        assert (right < topo.node_count).all()

        ### Each internal node's range is the midpoint split of its children:
        # left starts at the parent start, right ends at the parent end, and
        # the two child ranges are adjacent.
        range_start = topo.range_start[used]
        range_count = topo.range_count[used]
        assert range_start[0].item() == 0
        assert range_count[0].item() == n_items
        internal = torch.where(~is_leaf)[0]
        lc, rc = left[internal], right[internal]
        assert torch.equal(range_start[lc], range_start[internal])
        assert torch.equal(range_start[rc], range_start[lc] + range_count[lc])
        assert torch.equal(range_count[internal], range_count[lc] + range_count[rc])

    def test_leaf_size_invalid(self):
        with pytest.raises(ValueError, match="leaf_size"):
            build_lbvh_topology(10, 0, DEVICE)


class TestSharedTopologyContract:
    @pytest.mark.parametrize("leaf_size", [1, 4])
    def test_bvh_and_cluster_tree_build_identical_trees(self, leaf_size):
        """BVH and ClusterTree over the same item count and leaf_size must
        produce the identical node topology -- the contract that lets them
        share ``build_lbvh_topology``. The topology depends only on the item
        count, so the mesh geometry need not relate to the points."""
        n = 137
        g = torch.Generator().manual_seed(0)
        pts = torch.rand(n, 3, generator=g)

        tree = ClusterTree.from_points(pts, leaf_size=leaf_size)
        mesh = Mesh(
            points=pts,
            cells=torch.randint(0, n, (n, 3), generator=g),
        )
        bvh = BVH.from_mesh(mesh, leaf_size=leaf_size)

        assert torch.equal(tree.node_left_child, bvh.node_left_child)
        assert torch.equal(tree.node_right_child, bvh.node_right_child)
        assert torch.equal(tree.leaf_start, bvh.leaf_start)
        assert torch.equal(tree.leaf_count, bvh.leaf_count)
