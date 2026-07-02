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

"""Direct unit tests for :class:`physicsnemo.mesh.spatial.cluster_tree.ClusterTree`.

ClusterTree was historically exercised only indirectly. These tests pin down its
own contracts, so the shared-LBVH-build refactor (and future changes) have a
safety net:

- **Coverage / no-double-count**: a dual-tree plan's four interaction streams
  ((near,near), (near,far), (far,near), (far,far)) together cover every
  (target, source) pair *exactly once*, for any theta, leaf size, self- or
  cross-interaction, and with ``expand_far_targets``. This is the invariant
  every downstream kernel/attention evaluation relies on.
- **Tree structure**: leaves partition the morton-sorted order, subtree ranges
  nest correctly, AABBs contain their points, and per-node total areas are
  exact sums - down to degenerate trees (n = 1, 2).
- **Aggregates**: per-node area-weighted means match a brute-force reference,
  including in fp32 on offset (all-positive) coordinates - the catastrophic-
  cancellation regime the internal fp64 prefix-sum path exists to handle.
- **Edge cases**: empty and single-point trees.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.spatial import ClusterTree
from physicsnemo.mesh.spatial._ragged import _ragged_arange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _points(n, n_dims, device, seed=0, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, n_dims, generator=g, dtype=dtype).to(device)


def _offset_points(n, n_dims, device, seed=0, dtype=torch.float32):
    """Offset (all-positive-ish) coordinates.

    This is the regime the aggregate prefix-sum path is most sensitive to:
    range sums of a long same-sign cumsum suffer catastrophic cancellation
    unless accumulated in fp64 internally.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(n, n_dims, generator=g, dtype=dtype) * 10.0 - 3.0).to(device)


def _areas(n, device, seed=1, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(n, generator=g, dtype=dtype) + 0.5).to(device)


def _coverage_counts(plan, target_tree, source_tree, n_targets, n_sources):
    """Expand all four plan streams into a dense (n_targets, n_sources) count.

    Every (target, source) pair must be covered exactly once for the plan to
    be a valid decomposition of the dense interaction.
    """
    device = source_tree.source_points.device
    count = torch.zeros(n_targets, n_sources, dtype=torch.long, device=device)

    def _acc(t_ids, s_ids):
        if t_ids.numel() > 0:
            count.index_put_((t_ids, s_ids), torch.ones_like(t_ids), accumulate=True)

    # (near, near): individual pairs.
    _acc(plan.near_target_ids, plan.near_source_ids)

    # (near, far): target point x every source in the source node's subtree.
    s_starts = source_tree.node_range_start[plan.nf_source_node_ids]
    s_counts = source_tree.node_range_count[plan.nf_source_node_ids]
    pos, seg = _ragged_arange(s_starts, s_counts)
    _acc(plan.nf_target_ids[seg], source_tree.sorted_source_order[pos])

    # (far, near): the broadcast targets of entry i x source point i.
    bpos, bseg = _ragged_arange(plan.fn_broadcast_starts, plan.fn_broadcast_counts)
    _acc(plan.fn_broadcast_targets[bpos], plan.fn_source_ids[bseg])

    # (far, far): every target in the target node x every source in the
    # source node (nested ragged expansion).
    t_starts = target_tree.node_range_start[plan.far_target_node_ids]
    t_counts = target_tree.node_range_count[plan.far_target_node_ids]
    s_starts = source_tree.node_range_start[plan.far_source_node_ids]
    s_counts = source_tree.node_range_count[plan.far_source_node_ids]
    tpos, tseg = _ragged_arange(t_starts, t_counts)
    expanded_tgts = target_tree.sorted_source_order[tpos]
    spos, sseg = _ragged_arange(s_starts[tseg], s_counts[tseg])
    _acc(expanded_tgts[sseg], source_tree.sorted_source_order[spos])

    return count


def _subtree_point_ids(tree, node_id):
    """Original point ids in a node's subtree (via the sorted-order range)."""
    start = int(tree.node_range_start[node_id])
    n = int(tree.node_range_count[node_id])
    return tree.sorted_source_order[start : start + n]


# ---------------------------------------------------------------------------
# Coverage / no-double-count: the core dual-tree contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.5])
@pytest.mark.parametrize("leaf_size", [1, 4])
def test_plan_covers_every_pair_exactly_once_cross(device, theta, leaf_size):
    """Cross-interaction plans cover all (target, source) pairs exactly once."""
    n_t, n_s = 57, 43
    tgt_pts = _points(n_t, 3, device, seed=0)
    src_pts = _points(n_s, 3, device, seed=1)
    src_tree = ClusterTree.from_points(src_pts, leaf_size=leaf_size)
    tgt_tree = ClusterTree.from_points(tgt_pts, leaf_size=leaf_size)
    plan = src_tree.find_dual_interaction_pairs(target_tree=tgt_tree, theta=theta)
    count = _coverage_counts(plan, tgt_tree, src_tree, n_t, n_s)
    assert (count == 1).all(), (
        f"coverage violated: min={count.min()}, max={count.max()}"
    )


@pytest.mark.parametrize("theta", [0.0, 1.0])
def test_plan_covers_every_pair_exactly_once_self(device, theta):
    """Self-interaction plans (target_tree is source_tree) are also exact."""
    n = 64
    pts = _points(n, 3, device, seed=2)
    tree = ClusterTree.from_points(pts, areas=_areas(n, device))
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=theta)
    count = _coverage_counts(plan, tree, tree, n, n)
    assert (count == 1).all()


def test_plan_coverage_with_expand_far_targets(device):
    """expand_far_targets converts (far,far) to (near,far) without gaps/overlap."""
    n = 64
    pts = _points(n, 3, device, seed=3)
    tree = ClusterTree.from_points(pts)
    plan = tree.find_dual_interaction_pairs(
        target_tree=tree, theta=1.0, expand_far_targets=True
    )
    assert plan.n_far_nodes == 0
    count = _coverage_counts(plan, tree, tree, n, n)
    assert (count == 1).all()


def test_theta_zero_is_all_near(device):
    """At theta=0 every interaction is an exact (near, near) pair."""
    n_t, n_s = 30, 20
    src_tree = ClusterTree.from_points(_points(n_s, 3, device, seed=4))
    tgt_tree = ClusterTree.from_points(_points(n_t, 3, device, seed=5))
    plan = src_tree.find_dual_interaction_pairs(target_tree=tgt_tree, theta=0.0)
    assert plan.n_near == n_t * n_s
    assert plan.n_far_nodes == 0 and plan.n_nf == 0 and plan.n_fn == 0


def test_plan_validates_and_far_field_engages(device):
    """plan.validate() passes, and theta=1 actually produces far-field work.

    The second assertion guards against a regression where everything is
    classified near (which would make the far-field machinery dead code while
    all exactness tests still pass).
    """
    tree = ClusterTree.from_points(_points(80, 3, device, seed=6))
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=1.0)
    plan.validate()  # raises on inconsistency
    assert plan.n_far_nodes + plan.n_nf + plan.n_fn > 0


# ---------------------------------------------------------------------------
# Tree structure invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 100])
@pytest.mark.parametrize("leaf_size", [1, 4])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_tree_structure_invariants(device, n, leaf_size, n_dims):
    """Leaves partition the sorted order; ranges nest; AABBs contain points."""
    pts = _points(n, n_dims, device, seed=7)
    areas = _areas(n, device)
    tree = ClusterTree.from_points(pts, leaf_size=leaf_size, areas=areas)

    ### Root covers everything, with the full area.
    assert int(tree.node_range_start[0]) == 0
    assert int(tree.node_range_count[0]) == n
    assert torch.isclose(tree.node_total_area[0], areas.sum(), rtol=1e-5)

    ### Leaves: occupancy <= leaf_size, and they partition [0, n).
    is_leaf = tree.leaf_count > 0
    assert (tree.leaf_count[is_leaf] <= leaf_size).all()
    starts = tree.leaf_start[is_leaf]
    counts = tree.leaf_count[is_leaf]
    order = starts.argsort()
    starts, counts = starts[order], counts[order]
    assert int(starts[0]) == 0
    assert (starts[1:] == (starts[:-1] + counts[:-1])).all()
    assert int(starts[-1] + counts[-1]) == n

    ### Leaf/internal bookkeeping is mutually consistent.
    is_internal = tree.node_left_child >= 0
    assert (is_internal == (tree.node_right_child >= 0)).all()
    assert not (is_leaf & is_internal).any()
    assert (tree.leaf_count[is_internal] == 0).all()
    assert torch.equal(tree.node_range_count[is_leaf], tree.leaf_count[is_leaf])

    ### Internal nodes: child ids are valid, and children partition the
    ### parent's range (left first, right immediately after).
    left = tree.node_left_child[is_internal]
    right = tree.node_right_child[is_internal]
    assert (left < tree.n_nodes).all() and (right < tree.n_nodes).all()
    assert (
        tree.node_range_count[is_internal]
        == tree.node_range_count[left] + tree.node_range_count[right]
    ).all()
    assert (tree.node_range_start[is_internal] == tree.node_range_start[left]).all()
    assert (
        tree.node_range_start[right]
        == tree.node_range_start[left] + tree.node_range_count[left]
    ).all()

    ### Per-node AABB containment, total area, and diameter.
    sorted_pts = pts[tree.sorted_source_order]
    sorted_areas = areas[tree.sorted_source_order]
    for node in range(tree.n_nodes):
        s = int(tree.node_range_start[node])
        c = int(tree.node_range_count[node])
        sub = sorted_pts[s : s + c]
        assert (sub >= tree.node_aabb_min[node] - 1e-6).all()
        assert (sub <= tree.node_aabb_max[node] + 1e-6).all()
        assert torch.isclose(
            tree.node_total_area[node],
            sorted_areas[s : s + c].sum(),
            rtol=1e-5,
        )
    diag_sq = (tree.node_aabb_max - tree.node_aabb_min).pow(2).sum(-1)
    assert torch.allclose(tree.node_diameter_sq, diag_sq, rtol=1e-6)


def test_sorted_source_order_is_permutation(device):
    n = 77
    tree = ClusterTree.from_points(_points(n, 3, device, seed=8))
    assert torch.equal(
        tree.sorted_source_order.sort().values,
        torch.arange(n, device=device),
    )


# ---------------------------------------------------------------------------
# Aggregates vs brute force
# ---------------------------------------------------------------------------


def test_source_aggregates_match_bruteforce(device):
    """Per-node centroids and data means equal explicit area-weighted means."""
    n = 90
    pts = _points(n, 3, device, seed=9, dtype=torch.float64)
    areas = _areas(n, device, dtype=torch.float64)
    tree = ClusterTree.from_points(pts, leaf_size=2, areas=areas)
    data = TensorDict(
        {
            "vec": _points(n, 3, device, seed=10, dtype=torch.float64),
            "mat": _points(n, 6, device, seed=11, dtype=torch.float64).reshape(n, 2, 3),
        },
        batch_size=[n],
        device=device,
    )
    agg = tree.compute_source_aggregates(
        source_points=pts, areas=areas, source_data=data
    )

    for node in range(tree.n_nodes):
        ids = _subtree_point_ids(tree, node)
        w = areas[ids]
        ref_centroid = (pts[ids] * w[:, None]).sum(0) / w.sum()
        assert (agg.node_centroid[node] - ref_centroid).abs().max() < 1e-10
        for key in ("vec", "mat"):
            flat = data[key][ids].reshape(len(ids), -1)
            ref = (flat * w[:, None]).sum(0) / w.sum()
            got = agg.node_source_data[key][node].reshape(-1)
            assert (got - ref).abs().max() < 1e-10


def test_source_aggregates_use_call_time_weights(device):
    """Runtime weights normalize aggregates, including zero-weight subtrees."""
    points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], device=device)
    construction_areas = torch.ones(3, device=device)
    runtime_areas = torch.tensor([0.0, 2.0, 7.0], device=device)
    data = TensorDict(
        {"value": torch.tensor([[1.0], [4.0], [9.0]], device=device)},
        batch_size=[3],
        device=device,
    )
    tree = ClusterTree.from_points(points, leaf_size=1, areas=construction_areas)

    aggregates = tree.compute_source_aggregates(points, runtime_areas, data)

    for node in range(tree.n_nodes):
        ids = _subtree_point_ids(tree, node)
        weights = runtime_areas[ids]
        total_weight = weights.sum()
        if total_weight == 0:
            assert aggregates.node_centroid[node].eq(0).all()
            assert aggregates.node_source_data["value"][node].eq(0).all()
            continue

        expected_centroid = (points[ids] * weights[:, None]).sum(0) / total_weight
        expected_value = (data["value"][ids] * weights[:, None]).sum(0) / total_weight
        torch.testing.assert_close(aggregates.node_centroid[node], expected_centroid)
        torch.testing.assert_close(
            aggregates.node_source_data["value"][node], expected_value
        )


def test_fp32_aggregates_accurate_on_offset_coordinates(device):
    """fp32 centroids stay accurate on offset (all-positive) coordinates.

    Range sums extracted from a long same-sign cumsum suffer catastrophic
    cancellation in fp32; the implementation accumulates in fp64 internally to
    avoid this. This test pins that behavior at the fp32 public interface, in
    the coordinate regime (offset coordinates, small leaves) where plain fp32
    prefix sums were measurably wrong.
    """
    n = 200
    pts = _offset_points(n, 3, device, seed=12)
    areas = _areas(n, device)
    tree = ClusterTree.from_points(pts, leaf_size=1, areas=areas)
    agg = tree.compute_source_aggregates(source_points=pts, areas=areas)

    pts64, areas64 = pts.double(), areas.double()
    for node in range(tree.n_nodes):
        ids = _subtree_point_ids(tree, node)
        w = areas64[ids]
        ref = (pts64[ids] * w[:, None]).sum(0) / w.sum()
        assert (agg.node_centroid[node].double() - ref).abs().max() < 1e-5, (
            f"node {node} centroid mismatch"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_tree_and_plan(device):
    pts = torch.empty(0, 3, device=device)
    tree = ClusterTree.from_points(pts)
    assert tree.n_nodes == 0 and tree.n_sources == 0
    other = ClusterTree.from_points(_points(10, 3, device, seed=12))
    plan = tree.find_dual_interaction_pairs(target_tree=other, theta=1.0)
    assert plan.n_near == 0 and plan.n_far_nodes == 0
    assert plan.n_nf == 0 and plan.n_fn == 0


def test_single_point_self_plan(device):
    pts = _points(1, 3, device, seed=13)
    tree = ClusterTree.from_points(pts)
    assert tree.n_sources == 1
    assert tree.sorted_source_order.tolist() == [0]
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=1.0)
    count = _coverage_counts(plan, tree, tree, 1, 1)
    assert (count == 1).all()


def test_invalid_leaf_size_raises(device):
    with pytest.raises(ValueError, match="leaf_size"):
        ClusterTree.from_points(_points(10, 3, device), leaf_size=0)
