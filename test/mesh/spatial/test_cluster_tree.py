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

"""Direct unit tests for the ClusterTree (Barnes-Hut dual-tree) data structure.

Previously ClusterTree was only exercised indirectly via the experimental GLOBE
Barnes-Hut kernel tests (test/models/globe/). These tests pin its construction
invariants, per-node aggregates, and dual-traversal cover directly, so the
shared-LBVH-build refactor (and future changes) have a safety net.
"""

import pytest
import torch

from physicsnemo.mesh.spatial.cluster_tree import ClusterTree


def _random_points(n: int, d: int = 3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Offset (all-positive-ish) coordinates, the regime the aggregate prefix-sum
    # path is most sensitive to.
    return torch.rand(n, d, generator=g) * 10.0 - 3.0


def _random_areas(n: int, seed: int = 99) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g) + 0.1


class TestClusterTreeConstruction:
    @pytest.mark.parametrize("leaf_size", [1, 4])
    @pytest.mark.parametrize("n", [1, 2, 7, 50])
    def test_construction_invariants(self, n, leaf_size):
        pts = _random_points(n)
        areas = _random_areas(n)
        tree = ClusterTree.from_points(pts, leaf_size=leaf_size, areas=areas)

        # sorted_source_order is a permutation of range(n)
        order = tree.sorted_source_order
        assert order.shape == (n,)
        assert torch.equal(torch.sort(order).values, torch.arange(n))

        # Root node covers all sources, with the full area.
        assert tree.node_range_start[0].item() == 0
        assert tree.node_range_count[0].item() == n
        assert torch.isclose(tree.node_total_area[0], areas.sum())

        internal = tree.leaf_start < 0
        leaves = ~internal
        # Internal nodes carry no leaf payload; leaves carry their range count.
        assert torch.equal(
            tree.leaf_count[internal],
            torch.zeros(int(internal.sum()), dtype=tree.leaf_count.dtype),
        )
        assert torch.equal(tree.node_range_count[leaves], tree.leaf_count[leaves])

        # Child indices are either -1 or valid node indices.
        for child in (tree.node_left_child, tree.node_right_child):
            present = child[child >= 0]
            assert (present < tree.n_nodes).all()

        # Each node's AABB bounds the points in its (contiguous, sorted) subtree,
        # and an internal node's range count equals the sum of its children's.
        sorted_pts = pts[order]
        for node in range(tree.n_nodes):
            s = tree.node_range_start[node].item()
            c = tree.node_range_count[node].item()
            if c > 0:
                sub = sorted_pts[s : s + c]
                assert (sub >= tree.node_aabb_min[node] - 1e-5).all()
                assert (sub <= tree.node_aabb_max[node] + 1e-5).all()
            if tree.leaf_start[node].item() < 0:  # internal
                child_sum = 0
                for child in (
                    tree.node_left_child[node].item(),
                    tree.node_right_child[node].item(),
                ):
                    if child >= 0:
                        child_sum += tree.node_range_count[child].item()
                assert c == child_sum

    def test_empty_point_set(self):
        tree = ClusterTree.from_points(torch.empty(0, 3))
        assert tree.n_sources == 0
        assert tree.n_nodes == 0

    def test_single_point(self):
        tree = ClusterTree.from_points(torch.tensor([[1.0, 2.0, 3.0]]))
        assert tree.n_sources == 1
        assert tree.node_range_count[0].item() == 1
        assert tree.sorted_source_order.tolist() == [0]

    def test_leaf_size_invalid(self):
        with pytest.raises(ValueError, match="leaf_size"):
            ClusterTree.from_points(_random_points(5), leaf_size=0)


class TestSourceAggregates:
    def test_centroids_match_direct_area_weighted_mean(self):
        n = 40
        pts = _random_points(n, seed=2)
        areas = _random_areas(n, seed=3)
        tree = ClusterTree.from_points(pts, leaf_size=1, areas=areas)

        agg = tree.compute_source_aggregates(pts, areas)
        order = tree.sorted_source_order
        sorted_pts, sorted_areas = pts[order], areas[order]
        for node in range(tree.n_nodes):
            s = tree.node_range_start[node].item()
            c = tree.node_range_count[node].item()
            if c == 0:
                continue
            w = sorted_areas[s : s + c]
            expected = (sorted_pts[s : s + c] * w.unsqueeze(-1)).sum(0) / w.sum()
            assert torch.allclose(agg.node_centroid[node], expected, atol=1e-5), (
                f"node {node} centroid mismatch"
            )


class TestDualTraversal:
    def test_theta_zero_is_exact_full_cover(self):
        """theta=0 forces all interactions exact: no node opens as far, so the
        (near, near) stream is exactly the full target x source product and all
        far streams are empty."""
        n = 12
        pts = _random_points(n, seed=4)
        tree = ClusterTree.from_points(pts, leaf_size=1)

        plan = tree.find_dual_interaction_pairs(tree, theta=0.0)
        plan.validate()

        assert plan.n_far_nodes == 0
        assert plan.n_nf == 0
        assert plan.n_fn == 0

        pairs = set(zip(plan.near_target_ids.tolist(), plan.near_source_ids.tolist()))
        assert pairs == {(t, s) for t in range(n) for s in range(n)}

    def test_plan_validates_at_moderate_theta(self):
        pts = _random_points(60, seed=5)
        tree = ClusterTree.from_points(pts, leaf_size=2)
        plan = tree.find_dual_interaction_pairs(tree, theta=1.0)
        plan.validate()  # raises on any internal inconsistency
        # Some far-field approximation should kick in at theta=1 for 60 points.
        assert plan.n_far_nodes + plan.n_nf + plan.n_fn > 0
