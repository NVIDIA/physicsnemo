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

"""Tests for :class:`physicsnemo.mesh.spatial.cluster_tree.ClusterTree`.

ClusterTree was historically exercised only indirectly. These tests pin down its
own contracts, so the shared-LBVH-build refactor (and future changes) have a
safety net, and additionally attack the dual-tree interaction plan with the
classic tree-code failure modes.

- **Coverage / no-double-count**: a dual-tree plan's four interaction streams
  ((near,near), (near,far), (far,near), (far,far)) together cover every
  (target, source) pair *exactly once*, for any theta, leaf size, self- or
  cross-interaction, and with ``expand_far_targets``. This is the invariant
  every downstream kernel/attention evaluation relies on. It is verified on
  random clouds and on an adversarial geometry battery (coincident points,
  morton-tie grids, collinear/coplanar degeneracies, large offsets, extreme
  size asymmetry, leaf-size boundaries, and the zero-survivor broadcast edge),
  plus a randomized fuzz over the survivor/no-survivor broadcast regime.
- **MAC soundness**: every admitted far-type entry satisfies its multipole
  acceptance criterion when recomputed in float64.
- **Tree structure**: leaves partition the morton-sorted order, subtree ranges
  nest correctly, AABBs contain their points, and per-node total areas are
  exact sums - down to degenerate trees (n = 1, 2).
- **Aggregates**: per-node area-weighted means and totals match a brute-force
  reference, including in fp32 on offset (all-positive) coordinates - the
  catastrophic-cancellation regime the internal fp64 prefix-sum path exists to
  handle - and with zero-weight subtrees.
- **Edge cases & validation**: empty and single-point trees, and that
  ``plan.validate()`` catches corrupted plans.
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


def _assert_exact_cover(
    target_points,
    source_points=None,
    *,
    theta=1.0,
    leaf_size=4,
    expand_far_targets=False,
    context="",
):
    """Build a plan and assert that it covers every pair exactly once."""
    target_tree = ClusterTree.from_points(target_points, leaf_size=leaf_size)
    if source_points is None:
        source_points = target_points
        source_tree = target_tree
    else:
        source_tree = ClusterTree.from_points(source_points, leaf_size=leaf_size)
    plan = source_tree.find_dual_interaction_pairs(
        target_tree,
        theta=theta,
        expand_far_targets=expand_far_targets,
    )
    plan.validate()
    count = _coverage_counts(
        plan,
        target_tree,
        source_tree,
        target_points.shape[0],
        source_points.shape[0],
    )
    bad = (count != 1).nonzero(as_tuple=False)
    assert bad.numel() == 0, (
        f"{context}: {bad.shape[0]} pairs were not covered exactly once; "
        f"first pairs={bad[:5].tolist()}"
    )
    return plan


def _aabb_distance_squared_float64(points, aabb_min, aabb_max):
    """Compute point-to-AABB squared distance independently in float64."""
    points = points.double()
    clamped = torch.clamp(points, min=aabb_min.double(), max=aabb_max.double())
    return (points - clamped).pow(2).sum(dim=-1)


def _adversarial_clouds(device, n_dims):
    """Point sets covering common tree-code geometric degeneracies."""
    generator = torch.Generator(device="cpu").manual_seed(1234)

    def random_points(n):
        return torch.randn(n, n_dims, generator=generator)

    first_cluster = torch.zeros(20, n_dims)
    second_cluster = torch.zeros(20, n_dims)
    second_cluster[:, 0] = 100.0

    line = torch.zeros(41, n_dims)
    line[:, 0] = torch.linspace(0, 1, 41)

    axes = [torch.arange(5, dtype=torch.float32)] * n_dims
    grid = torch.cartesian_prod(*axes).reshape(-1, n_dims)

    outlier = random_points(33) * 0.01
    outlier[0] = 1.0e4

    base = random_points(11)
    clouds = {
        "all coincident": torch.zeros(37, n_dims),
        "two tight clusters": torch.cat([first_cluster, second_cluster]),
        "collinear": line,
        "integer grid": grid,
        # At 1e4, a float32 cloud still retains the 0.1-scale perturbations.
        "large offset": random_points(64) * 0.1 + 1.0e4,
        "single outlier": outlier,
        "mixed duplicates": torch.cat([base, base[:7], base[:3]]),
    }
    if n_dims == 3:
        plane = random_points(29)
        plane[:, 2] = 0.0
        clouds["coplanar"] = plane
    return {name: points.to(device) for name, points in clouds.items()}


def _zero_survivor_clouds(device):
    """Leaf-pair geometry with far targets but no broadcast survivors."""
    generator = torch.Generator(device="cpu").manual_seed(17)
    targets = torch.rand(8, 3, generator=generator) - 0.5
    sources = targets.clone()
    sources[:, 0] += 2.4
    return targets.to(device), sources.to(device)


# Relative slack for re-verifying float32 comparisons in float64: admits
# only genuine rounding at the comparison boundary, not logic errors.
_MAC_RTOL = 1e-5

# ---------------------------------------------------------------------------
# Coverage / no-double-count: the core dual-tree contract (random clouds)
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


def _plan_entries(plan):
    """Each stream of a plan as a sorted list of its entries (order-free)."""
    far = list(
        zip(plan.far_target_node_ids.tolist(), plan.far_source_node_ids.tolist())
    )
    fn = list(zip(plan.fn_target_node_ids.tolist(), plan.fn_source_ids.tolist()))
    return {
        "near": sorted(
            zip(plan.near_target_ids.tolist(), plan.near_source_ids.tolist())
        ),
        "nf": sorted(
            zip(plan.nf_target_ids.tolist(), plan.nf_source_node_ids.tolist())
        ),
        "far": sorted(far),
        "fn": sorted(fn),
        "far_broadcast": sorted(
            (t, *far[p])
            for t, p in zip(
                plan.far_broadcast_target_ids.tolist(),
                plan.far_broadcast_pair_ids.tolist(),
            )
        ),
        "fn_expanded": sorted(
            (t, *fn[p])
            for t, p in zip(
                plan.fn_expanded_target_ids.tolist(), plan.fn_expanded_pair_ids.tolist()
            )
        ),
    }


### What each id of a stream's entries indexes: "t" or "s" for the target
### or source side, then "p" or "n" for a point or node id.
_ENTRY_SIDES = {
    "near": ("tp", "sp"),
    "nf": ("tp", "sn"),
    "far": ("tn", "sn"),
    "fn": ("tn", "sp"),
    "far_broadcast": ("tp", "tn", "sn"),
    "fn_expanded": ("tp", "tn", "sp"),
}


def _joined_entries(parts):
    """Join per-tree plan entries into the entries of one forest plan.

    ``parts`` holds ``(entries, offsets)`` per (target tree, source tree)
    pair, with ``offsets`` mapping ``"tp"``, ``"tn"``, ``"sp"`` and ``"sn"``
    to that pair's point and node offsets in the forests.
    """
    joined = {name: [] for name in _ENTRY_SIDES}
    for entries, offsets in parts:
        for name, sides in _ENTRY_SIDES.items():
            for entry in entries[name]:
                joined[name].append(
                    tuple(i + offsets[side] for i, side in zip(entry, sides))
                )
    return {name: sorted(v) for name, v in joined.items()}


def _forest_offsets(trees):
    """Point and node offset of each tree in the forest of ``trees``."""
    offsets, n_points, n_nodes = [], 0, 0
    for tree in trees:
        offsets.append((n_points, n_nodes))
        n_points += tree.n_sources
        n_nodes += tree.n_nodes
    return offsets


def _forest_trees(device, seed, sizes, leaf_size=2):
    """Trees over separate random clouds of the given sizes."""
    return [
        ClusterTree.from_points(
            _points(n, 3, device, seed=seed + i) * (0.1 + i) + 1.5 * i,
            leaf_size=leaf_size,
        )
        for i, n in enumerate(sizes)
    ]


def _assert_forest_plan(tgt_trees, src_trees, target, source, kwargs):
    """The forest plan is the per-tree plans joined and covers every pair once."""
    plan = source.find_dual_interaction_pairs(target, **kwargs)
    plan.validate()
    parts = []
    for tgt, (tp, tn) in zip(tgt_trees, _forest_offsets(tgt_trees)):
        for src, (sp, sn) in zip(src_trees, _forest_offsets(src_trees)):
            part = src.find_dual_interaction_pairs(tgt, **kwargs)
            parts.append((_plan_entries(part), dict(tp=tp, tn=tn, sp=sp, sn=sn)))
    assert _plan_entries(plan) == _joined_entries(parts)
    count = _coverage_counts(plan, target, source, target.n_sources, source.n_sources)
    assert (count == 1).all()


@pytest.mark.parametrize("theta", [0.0, 1.0, 1.5])
@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_forest_plan_is_the_joined_per_tree_plans(device, theta, expand_far_targets):
    """One traversal against a merged forest plans what one per tree would.

    The target trees (one of them empty) are merged; a traversal from the
    forest's roots must give, stream by stream, the per-tree plans with
    target ids and target-node ids shifted to the concatenated points.
    """
    src_tree = ClusterTree.from_points(_points(40, 3, device, seed=4), leaf_size=2)
    target_sets = [
        _points(35, 3, device, seed=5),
        _points(0, 3, device, seed=6),
        _points(22, 3, device, seed=7) + 1.5,
        _points(9, 3, device, seed=8) * 0.1,
    ]
    trees = [ClusterTree.from_points(p, leaf_size=2) for p in target_sets]
    forest = ClusterTree.merge(trees)
    assert forest.n_sources == sum(p.shape[0] for p in target_sets)
    assert forest.root_ids.tolist() == [
        0,
        trees[0].n_nodes,
        trees[0].n_nodes + trees[2].n_nodes,
    ]
    kwargs = dict(theta=theta, expand_far_targets=expand_far_targets)
    _assert_forest_plan(trees, [src_tree], forest, src_tree, kwargs)


@pytest.mark.parametrize("theta", [0.0, 1.0, 1.5])
@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_forest_source_plan_is_the_joined_per_tree_plans(
    device, theta, expand_far_targets
):
    """A forest as the source plans what one traversal per source tree would."""
    tgt_tree = ClusterTree.from_points(_points(25, 3, device, seed=9), leaf_size=2)
    trees = _forest_trees(device, 10, [30, 0, 20, 7])
    forest = ClusterTree.merge(trees)
    kwargs = dict(theta=theta, expand_far_targets=expand_far_targets)
    _assert_forest_plan([tgt_tree], trees, tgt_tree, forest, kwargs)


@pytest.mark.parametrize("theta", [0.0, 1.0, 1.5])
@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_forest_to_forest_plan_is_the_joined_per_tree_plans(
    device, theta, expand_far_targets
):
    """Forest to forest, and a forest to itself, plans every pair of trees."""
    src_trees = _forest_trees(device, 20, [30, 20, 7])
    tgt_trees = _forest_trees(device, 30, [12, 0, 26])
    src_forest = ClusterTree.merge(src_trees)
    tgt_forest = ClusterTree.merge(tgt_trees)
    kwargs = dict(theta=theta, expand_far_targets=expand_far_targets)
    _assert_forest_plan(tgt_trees, src_trees, tgt_forest, src_forest, kwargs)
    _assert_forest_plan(src_trees, src_trees, src_forest, src_forest, kwargs)


def test_nested_merge_keeps_every_root(device):
    """Merging a forest keeps its roots and gives the flat merge."""
    a, empty, b, c = _forest_trees(device, 40, [30, 0, 20, 15])
    assert a.root_ids.tolist() == [0] and empty.root_ids.tolist() == []
    nested = ClusterTree.merge([ClusterTree.merge([a, empty, b]), c])
    flat = ClusterTree.merge([a, b, c])
    assert nested.root_ids.tolist() == [0, a.n_nodes, a.n_nodes + b.n_nodes]
    assert (nested.to_tensordict() == flat.to_tensordict()).all()
    s = ClusterTree.from_points(_points(25, 3, device, seed=44), leaf_size=2)
    for target, source in [(s, nested), (nested, s), (nested, nested)]:
        flat_target = flat if target is nested else s
        flat_source = flat if source is nested else s
        assert _plan_entries(
            source.find_dual_interaction_pairs(target)
        ) == _plan_entries(flat_source.find_dual_interaction_pairs(flat_target))


def test_merge_rejects_mismatched_trees(device):
    """merge checks dtype, device and spatial dimension, and needs a tree."""
    tree = ClusterTree.from_points(_points(10, 3, device, seed=50))
    other_dtype = ClusterTree.from_points(
        _points(10, 3, device, seed=51, dtype=torch.float64)
    )
    with pytest.raises(TypeError, match=r"dtype: trees \[1\]"):
        ClusterTree.merge([tree, other_dtype])
    with pytest.raises(ValueError, match=r"device: trees \[2\]"):
        ClusterTree.merge([tree, tree, tree.to("meta")])
    with pytest.raises(ValueError, match=r"n_spatial_dims: trees \[1\]"):
        ClusterTree.merge([tree, ClusterTree.from_points(_points(10, 2, device))])
    with pytest.raises(ValueError, match="at least one tree"):
        ClusterTree.merge([])


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


@pytest.mark.parametrize("n_dims", [2, 3])
def test_adversarial_self_plan_coverage(device, n_dims):
    """Hostile geometries still produce an exact self-interaction cover."""
    for name, points in _adversarial_clouds(device, n_dims).items():
        _assert_exact_cover(points, context=f"{n_dims}D {name}")


@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_adversarial_cross_plan_coverage(device, expand_far_targets):
    """Mismatched hostile trees and extreme size ratios remain exact."""
    clouds = _adversarial_clouds(device, 3)
    generator = torch.Generator(device="cpu").manual_seed(19)
    singleton = torch.randn(1, 3, generator=generator).to(device)
    many_points = torch.randn(200, 3, generator=generator).to(device)
    pairs = [
        ("clusters/coincident", clouds["two tight clusters"], clouds["all coincident"]),
        ("grid/line", clouds["integer grid"], clouds["collinear"]),
        ("offset/outlier", clouds["large offset"], clouds["single outlier"]),
        ("singleton/many", singleton, many_points),
        ("many/singleton", many_points, singleton),
    ]
    for name, target_points, source_points in pairs:
        _assert_exact_cover(
            target_points,
            source_points,
            expand_far_targets=expand_far_targets,
            context=name,
        )


@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_nan_point_keeps_exact_cover(device, expand_far_targets):
    """A NaN coordinate must not drop interactions from the plan.

    Every box containing the NaN point has NaN bounds and diameter, so no
    opening test passes; the split rule must still descend such pairs to
    their leaves (where they become exact near-field pairs and the NaN
    reaches the output) rather than discard them.
    """
    points = _points(60, 3, device, seed=21)
    points[17, 1] = float("nan")
    _assert_exact_cover(points, theta=1.0, expand_far_targets=expand_far_targets)
    _assert_exact_cover(
        points,
        _points(45, 3, device, seed=22),
        theta=1.0,
        expand_far_targets=expand_far_targets,
    )


def test_float64_plan_coverage(device):
    """Float64 trees preserve the exactly-once plan contract."""
    points = _points(97, 3, device, seed=20, dtype=torch.float64)
    _assert_exact_cover(points)


def test_zero_survivor_leaf_pair_is_valid(device):
    """Zero-count far/near broadcasts remain in bounds and cover exactly once."""
    target_points, source_points = _zero_survivor_clouds(device)
    plan = _assert_exact_cover(
        target_points,
        source_points,
        leaf_size=target_points.shape[0],
        context="zero-survivor leaf pair",
    )

    assert plan.n_fn > 0
    assert plan.fn_broadcast_targets.numel() == 0
    assert plan.fn_broadcast_counts.eq(0).all()


def test_far_admissions_satisfy_mac_in_float64(device):
    """Independently recompute every far-stream acceptance criterion."""
    theta = 0.7
    target_points = (_points(80, 3, device, seed=21) * 2.0).add(0.5)
    source_points = (_points(70, 3, device, seed=22) * 2.0).add(1.5)
    target_tree = ClusterTree.from_points(target_points, leaf_size=4)
    source_tree = ClusterTree.from_points(source_points, leaf_size=4)
    plan = source_tree.find_dual_interaction_pairs(target_tree, theta=theta)
    plan.validate()

    assert plan.n_far_nodes > 0 and plan.n_nf > 0 and plan.n_fn > 0
    comparison_slack = 1.0 + 1.0e-5

    target_min = target_tree.node_aabb_min[plan.far_target_node_ids].double()
    target_max = target_tree.node_aabb_max[plan.far_target_node_ids].double()
    source_min = source_tree.node_aabb_min[plan.far_source_node_ids].double()
    source_max = source_tree.node_aabb_max[plan.far_source_node_ids].double()
    gap = torch.clamp(
        torch.maximum(target_min - source_max, source_min - target_max), min=0
    )
    distance_squared = gap.pow(2).sum(dim=-1)
    target_diameter = (
        target_tree.node_diameter_sq[plan.far_target_node_ids].double().sqrt()
    )
    source_diameter = (
        source_tree.node_diameter_sq[plan.far_source_node_ids].double().sqrt()
    )
    assert (
        distance_squared * theta**2 * comparison_slack
        > (target_diameter + source_diameter).pow(2)
    ).all()

    distance_squared = _aabb_distance_squared_float64(
        target_points[plan.nf_target_ids],
        source_tree.node_aabb_min[plan.nf_source_node_ids],
        source_tree.node_aabb_max[plan.nf_source_node_ids],
    )
    assert (
        distance_squared * theta**2 * comparison_slack
        > source_tree.node_diameter_sq[plan.nf_source_node_ids].double()
    ).all()

    distance_squared = _aabb_distance_squared_float64(
        source_points[plan.fn_source_ids],
        target_tree.node_aabb_min[plan.fn_target_node_ids],
        target_tree.node_aabb_max[plan.fn_target_node_ids],
    )
    assert (
        distance_squared * theta**2 * comparison_slack
        > target_tree.node_diameter_sq[plan.fn_target_node_ids].double()
    ).all()


# ---------------------------------------------------------------------------
# Coverage on the adversarial geometry battery + fuzz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("theta", [0.0, 0.7, 1.5])
@pytest.mark.parametrize("leaf_size", [1, 7])
def test_adversarial_self_coverage(device, n_dims, theta, leaf_size):
    """Self-interaction plans cover every pair exactly once on hostile geometry."""
    for name, pts in _adversarial_clouds(device, n_dims).items():
        tree = ClusterTree.from_points(pts, leaf_size=leaf_size)
        plan = tree.find_dual_interaction_pairs(tree, theta=theta)
        plan.validate()
        count = _coverage_counts(plan, tree, tree, pts.shape[0], pts.shape[0])
        assert (count == 1).all(), (
            f"cloud={name!r} n_dims={n_dims} theta={theta} leaf_size={leaf_size}: "
            f"{(count != 1).sum().item()} mis-covered pairs"
        )


@pytest.mark.parametrize("theta", [0.7, 1.5])
@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_adversarial_cross_coverage(device, theta, expand_far_targets):
    """Cross-tree plans (mismatched hostile geometries) cover exactly once."""
    clouds3 = _adversarial_clouds(device, 3)
    pairs = [
        ("two tight clusters", "all coincident"),
        ("integer grid", "collinear"),
        ("large offset", "single outlier"),
        ("mixed duplicates", "integer grid"),
    ]
    for tname, sname in pairs:
        _assert_exact_cover(
            clouds3[tname],
            clouds3[sname],
            theta=theta,
            leaf_size=4,
            expand_far_targets=expand_far_targets,
        )


@pytest.mark.parametrize("n_t,n_s", [(1, 200), (200, 1), (1, 1), (2, 3)])
def test_extreme_size_asymmetry(device, n_t, n_s):
    """Tiny-vs-large trees: single-point trees exercise root==leaf paths."""
    g = torch.Generator(device="cpu").manual_seed(7)
    pts_t = torch.randn(n_t, 3, generator=g).to(device)
    pts_s = torch.randn(n_s, 3, generator=g).to(device)
    for theta in (0.0, 1.0):
        _assert_exact_cover(pts_t, pts_s, theta=theta, leaf_size=4)


@pytest.mark.parametrize("leaf_size", [1, 2, 7, 16, 64])
def test_leaf_size_boundaries(device, leaf_size):
    """n around leaf_size (n = ls-1, ls, ls+1, 2*ls+1) covers exactly once."""
    g = torch.Generator(device="cpu").manual_seed(11)
    for n in {max(1, leaf_size - 1), leaf_size, leaf_size + 1, 2 * leaf_size + 1}:
        pts = torch.randn(n, 3, generator=g).to(device)
        tree = ClusterTree.from_points(pts, leaf_size=leaf_size)
        plan = tree.find_dual_interaction_pairs(tree, theta=1.0)
        count = _coverage_counts(plan, tree, tree, n, n)
        assert (count == 1).all(), f"n={n}, leaf_size={leaf_size}"


def test_float64_tree_coverage(device):
    """float64 point trees traverse and cover exactly once."""
    g = torch.Generator(device="cpu").manual_seed(13)
    pts = (torch.randn(97, 3, generator=g, dtype=torch.float64)).to(device)
    tree = ClusterTree.from_points(pts, leaf_size=4)
    plan = tree.find_dual_interaction_pairs(tree, theta=1.0)
    count = _coverage_counts(plan, tree, tree, 97, 97)
    assert (count == 1).all()


def test_zero_survivor_leaf_pair_fn_entries(device):
    """Leaf pairs where EVERY target is stage-1 far (zero fn-broadcast
    survivors) must not corrupt the plan or the broadcast index remap.

    Geometry: two clusters of diameter ~1 separated by gap ~1.5 with
    theta=1.0.  The node-pair criterion fails (1.5 < D_T + D_S ~ 2) so the
    pair descends to leaf-leaf; every per-point test passes (1.5 > 1), so
    all targets go (near, far), all sources test far with ZERO survivors --
    the fn stream then carries count-0 entries whose starts must never be
    dereferenced out of bounds.
    """
    g = torch.Generator(device="cpu").manual_seed(17)
    n = 8
    a = (torch.rand(n, 3, generator=g) - 0.5).to(device)  # diameter ~<= sqrt(3)*1
    b = a.clone()
    b[:, 0] += 2.4  # gap ~1.4-1.9 between AABBs along x
    pts_t = a
    pts_s = b
    # leaf_size >= n so each cluster is a single leaf; the root pair is
    # (leaf, leaf) immediately.
    plan = _assert_exact_cover(pts_t, pts_s, theta=1.0, leaf_size=n)
    # The interesting regime: nf covers everything, near is empty.
    assert plan.n_near == 0, "expected the all-far-targets regime"
    assert plan.n_nf == n
    # fn entries (if any survived compaction) must all have zero counts and
    # in-bounds starts even at the very end of the broadcast buffer.
    if plan.n_fn > 0:
        assert (plan.fn_broadcast_counts == 0).all()
    plan.validate()


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_cluster_geometries_mixed_survivors(device, seed):
    """Fuzz cluster-pair geometries around the survivor/no-survivor boundary.

    Random cluster diameters, gaps, thetas, and leaf sizes chosen to land
    leaf pairs on every side of the stage-1/stage-2 tests, including
    same-traversal mixtures of zero-survivor and some-survivor leaf pairs --
    the regime where the sentinel-padded broadcast remap could index out of
    bounds if the start bookkeeping were wrong.
    """
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)

    def _u(lo, hi):
        return float(torch.empty(1).uniform_(lo, hi, generator=g))

    n_clusters = int(torch.randint(2, 6, (1,), generator=g))
    pts_list = []
    for _ in range(n_clusters):
        n_i = int(torch.randint(2, 12, (1,), generator=g))
        center = torch.randn(3, generator=g) * _u(0.5, 4.0)
        diam = _u(0.05, 2.0)
        pts_list.append(torch.randn(n_i, 3, generator=g) * diam * 0.5 + center)
    pts = torch.cat(pts_list).to(device)
    n = pts.shape[0]

    theta = _u(0.3, 2.5)
    leaf_size = int(torch.randint(1, 9, (1,), generator=g))
    tree = ClusterTree.from_points(pts, leaf_size=leaf_size)
    plan = tree.find_dual_interaction_pairs(tree, theta=theta)
    plan.validate()
    count = _coverage_counts(plan, tree, tree, n, n)
    assert (count == 1).all(), (
        f"seed={seed}: {(count != 1).sum().item()} mis-covered pairs "
        f"(theta={theta:.3f}, leaf_size={leaf_size}, n={n})"
    )
    ### Broadcast starts must index within the compacted buffer for every
    ### entry that has a nonzero count (zero-count starts are never read).
    nz = plan.fn_broadcast_counts > 0
    if nz.any():
        ends = plan.fn_broadcast_starts[nz] + plan.fn_broadcast_counts[nz]
        assert int(ends.max()) <= plan.fn_broadcast_targets.shape[0]


# ---------------------------------------------------------------------------
# theta = 0 collapses to a fully exact near field
# ---------------------------------------------------------------------------


def test_theta_zero_is_fully_exact(device):
    """theta=0 yields ONLY exact (near, near) pairs, on random and hostile geometry.

    At theta=0 the multipole acceptance can never fire, so the entire dense
    interaction must be exact near pairs regardless of geometry.
    """
    ### Random cross pair.
    n_t, n_s = 30, 20
    src_tree = ClusterTree.from_points(_points(n_s, 3, device, seed=4))
    tgt_tree = ClusterTree.from_points(_points(n_t, 3, device, seed=5))
    plan = src_tree.find_dual_interaction_pairs(target_tree=tgt_tree, theta=0.0)
    assert plan.n_near == n_t * n_s
    assert plan.n_far_nodes == 0 and plan.n_nf == 0 and plan.n_fn == 0

    ### Every adversarial self cloud.
    for name, pts in _adversarial_clouds(device, 3).items():
        tree = ClusterTree.from_points(pts, leaf_size=4)
        plan = tree.find_dual_interaction_pairs(tree, theta=0.0)
        n = pts.shape[0]
        assert plan.n_far_nodes == 0 and plan.n_nf == 0 and plan.n_fn == 0, name
        assert plan.n_near == n * n, name


# ---------------------------------------------------------------------------
# MAC soundness: every far-type admission satisfies its criterion (fp64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.5, 1.0, 2.0])
def test_far_admissions_satisfy_mac_fp64(device, theta):
    """Recompute every admitted far/nf/fn criterion in float64.

    Also verifies node AABBs actually bound their subtree points and that
    ``node_diameter_sq`` matches the AABB diagonal -- the quantities the MAC
    depends on.
    """
    g = torch.Generator(device="cpu").manual_seed(23)
    pts_t = (torch.rand(300, 3, generator=g) * 4.0).to(device)
    pts_s = (torch.rand(260, 3, generator=g) * 4.0 + 1.0).to(device)
    tgt_tree = ClusterTree.from_points(pts_t, leaf_size=4)
    src_tree = ClusterTree.from_points(pts_s, leaf_size=4)
    plan = src_tree.find_dual_interaction_pairs(tgt_tree, theta=theta)

    ### AABB containment + diameter consistency, both trees.
    for tree, pts in ((tgt_tree, pts_t), (src_tree, pts_s)):
        starts = tree.node_range_start
        counts = tree.node_range_count
        pos, seg = _ragged_arange(starts, counts)
        subtree_pts = pts[tree.sorted_source_order[pos]]
        lo = tree.node_aabb_min[seg]
        hi = tree.node_aabb_max[seg]
        assert (subtree_pts >= lo).all() and (subtree_pts <= hi).all(), (
            "node AABB does not bound its subtree points"
        )
        diag_sq = (
            (tree.node_aabb_max.double() - tree.node_aabb_min.double()).pow(2).sum(-1)
        )
        assert torch.allclose(
            tree.node_diameter_sq.double(), diag_sq, rtol=1e-5, atol=1e-30
        ), "node_diameter_sq inconsistent with AABB diagonal"

    theta64 = float(theta)
    slack = 1.0 + _MAC_RTOL

    ### (far, far): min AABB gap * theta > D_T + D_S.
    if plan.n_far_nodes > 0:
        tmin = tgt_tree.node_aabb_min[plan.far_target_node_ids].double()
        tmax = tgt_tree.node_aabb_max[plan.far_target_node_ids].double()
        smin = src_tree.node_aabb_min[plan.far_source_node_ids].double()
        smax = src_tree.node_aabb_max[plan.far_source_node_ids].double()
        gap = torch.clamp(torch.maximum(tmin - smax, smin - tmax), min=0)
        min_dist = gap.pow(2).sum(-1).sqrt()
        d_t = tgt_tree.node_diameter_sq[plan.far_target_node_ids].double().sqrt()
        d_s = src_tree.node_diameter_sq[plan.far_source_node_ids].double().sqrt()
        assert (min_dist * theta64 * slack > d_t + d_s).all(), (
            "far-far admission violates the combined MAC"
        )

    ### (near, far): dist(target point, source node AABB) * theta > D_S.
    if plan.n_nf > 0:
        d = _aabb_distance_squared_float64(
            pts_t[plan.nf_target_ids],
            src_tree.node_aabb_min[plan.nf_source_node_ids],
            src_tree.node_aabb_max[plan.nf_source_node_ids],
        ).sqrt()
        d_s = src_tree.node_diameter_sq[plan.nf_source_node_ids].double().sqrt()
        assert (d * theta64 * slack > d_s).all(), (
            "nf admission violates the source-side MAC"
        )

    ### (far, near): dist(source point, target node AABB) * theta > D_T.
    if plan.n_fn > 0:
        d = _aabb_distance_squared_float64(
            pts_s[plan.fn_source_ids],
            tgt_tree.node_aabb_min[plan.fn_target_node_ids],
            tgt_tree.node_aabb_max[plan.fn_target_node_ids],
        ).sqrt()
        d_t = tgt_tree.node_diameter_sq[plan.fn_target_node_ids].double().sqrt()
        assert (d * theta64 * slack > d_t).all(), (
            "fn admission violates the target-side MAC"
        )


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


@pytest.mark.parametrize("with_zero_areas", [False, True])
@pytest.mark.parametrize("offset", [0.0, 100.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_source_aggregates_match_bruteforce(device, dtype, offset, with_zero_areas):
    """Per-node centroids, feature means, and total areas match area-weighted
    brute-force references.

    Parametrized over precision, an ``offset`` (all-positive) coordinate regime
    that stresses fp32 prefix-sum cancellation (range sums extracted from a long
    same-sign cumsum lose precision unless accumulated in fp64 internally), and
    zero-area (zero-weight) leaves, which must yield zero aggregates (a
    zero-weight *internal* node is constructed explicitly in
    :func:`test_zero_weight_internal_subtree_aggregates`).  ``leaf_size=1``
    maximizes the number of internal-node range reductions.

    Errors are bounded absolutely against a float64 reference.  The fp32 bound
    (1e-5 in coordinate/feature units) is a few output ulps at magnitude ~100
    and far below the milliscale errors of fp32 prefix sums in small leaves, so
    it pins the internal fp64 accumulation at the fp32 public interface.
    """
    n = 173
    pts = _points(n, 3, device, seed=9, dtype=dtype) + offset
    areas = _areas(n, device, dtype=dtype)
    if with_zero_areas:
        areas = areas.clone()
        areas[::5] = 0.0
    data = TensorDict(
        {
            "vec": _points(n, 3, device, seed=10, dtype=dtype),
            "mat": _points(n, 6, device, seed=11, dtype=dtype).reshape(n, 2, 3),
        },
        batch_size=[n],
        device=device,
    )
    tree = ClusterTree.from_points(pts, leaf_size=1, areas=areas)
    agg = tree.compute_source_aggregates(
        source_points=pts, areas=areas, source_data=data
    )

    # fp64: the reference itself carries ~ulp(cumsum total) ~ 4e-12 at offset 100.
    max_abs_err = 1e-5 if dtype == torch.float32 else 1e-10

    for node in range(tree.n_nodes):
        ids = _subtree_point_ids(tree, node)
        w = areas[ids].double()
        total = w.sum()
        if total == 0:
            assert agg.node_centroid[node].eq(0).all()
            for key in ("vec", "mat"):
                assert agg.node_source_data[key][node].eq(0).all()
        else:
            ref_c = (pts[ids].double() * w[:, None]).sum(0) / total
            err_c = (agg.node_centroid[node].double() - ref_c).abs().max()
            assert err_c < max_abs_err, (
                f"centroid mismatch at node {node}: max abs err {err_c:.3e}"
            )
            for key in ("vec", "mat"):
                flat = data[key][ids].reshape(len(ids), -1).double()
                ref = (flat * w[:, None]).sum(0) / total
                got = agg.node_source_data[key][node].reshape(-1).double()
                err = (got - ref).abs().max()
                assert err < max_abs_err, (
                    f"{key} aggregate mismatch at node {node}: max abs err {err:.3e}"
                )
        ### node_total_area is a construction-time reduction; brute-force it too.
        assert torch.allclose(
            tree.node_total_area[node].double(),
            areas[ids].double().sum(),
            rtol=1e-5,
            atol=1e-6,
        ), f"node_total_area mismatch at node {node}"


@pytest.mark.parametrize("split", ["morton", "midpoint"])
def test_zero_weight_internal_subtree_aggregates(device, split):
    """A zero-weight *internal* node reduces to exactly zero aggregates.

    Zeroing scattered areas only guarantees zero-weight single-point leaves.
    Here a spatially separated 8-point cluster carries zero weight: it takes the
    highest morton codes, so it is the final contiguous block in sorted order.
    With ``n = 64`` the index-midpoint splits make ``[56, 64)`` (and its 4- and
    2-point descendants) tree nodes, and the Morton splits separate the block
    at the coarse cell boundary in front of it. With ``leaf_size=1`` these are
    internal nodes whose whole range has zero total weight, so the zero-total
    guard is exercised on internal range reductions, not only on leaves.
    """
    n_main, n_cluster = 56, 8
    main = _points(n_main, 3, device, seed=14)
    cluster = _points(n_cluster, 3, device, seed=15) + 50.0
    pts = torch.cat([main, cluster])
    areas = torch.cat([_areas(n_main, device), torch.zeros(n_cluster, device=device)])
    data = TensorDict(
        {"vec": _points(n_main + n_cluster, 3, device, seed=16)},
        batch_size=[n_main + n_cluster],
        device=device,
    )
    tree = ClusterTree.from_points(pts, leaf_size=1, areas=areas, split=split)
    agg = tree.compute_source_aggregates(
        source_points=pts, areas=areas, source_data=data
    )

    zero_internal_nodes = []
    for node in range(tree.n_nodes):
        ids = _subtree_point_ids(tree, node)
        if not (ids >= n_main).all():
            continue
        ### Every node inside the cluster block has zero weight, exactly.
        assert tree.node_total_area[node] == 0
        assert agg.node_centroid[node].eq(0).all()
        assert agg.node_source_data["vec"][node].eq(0).all()
        if int(tree.node_left_child[node]) >= 0:
            zero_internal_nodes.append(node)

    ### The construction must actually have produced internal zero-weight
    ### nodes (the full 8-point block plus its 4- and 2-point descendants).
    counts = sorted(int(tree.node_range_count[node]) for node in zero_internal_nodes)
    if split == "midpoint":
        assert counts == [2, 2, 2, 2, 4, 4, 8], counts
    else:
        assert counts[-1] == n_cluster and len(counts) == n_cluster - 1, counts


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


def test_source_aggregates_keep_nested_keys_dtype_and_contiguity(device):
    """Nested leaves come back under their keys, in their dtype, contiguous."""
    n = 40
    points = _points(n, 3, device, seed=3)
    areas = _areas(n, device)
    data = TensorDict(
        {
            "normals": _points(n, 3, device, seed=4),
            "latent": {"s": _points(n, 1, device, seed=5)[:, 0]},
        },
        batch_size=[n],
        device=device,
    )
    tree = ClusterTree.from_points(points, leaf_size=4, areas=areas)
    aggregates = tree.compute_source_aggregates(points, areas, data)

    node_data = aggregates.node_source_data
    assert set(node_data.keys(include_nested=True, leaves_only=True)) == {
        "normals",
        ("latent", "s"),
    }
    for key in ("normals", ("latent", "s")):
        value = node_data[key]
        assert value.shape == (tree.n_nodes, *data[key].shape[1:])
        assert value.dtype == data[key].dtype
        assert value.is_contiguous()
    ### The root aggregates every source.
    root = int(torch.argmax(tree.node_range_count))
    expected = (data["normals"] * areas[:, None]).sum(0) / areas.sum()
    torch.testing.assert_close(node_data["normals"][root], expected)


def test_source_aggregates_reject_mixed_dtypes(device):
    """Source data is packed into one tensor, so its leaves must share a dtype."""
    n = 8
    points = _points(n, 3, device)
    areas = _areas(n, device)
    data = TensorDict(
        {
            "a": torch.zeros(n, device=device),
            "b": torch.zeros(n, dtype=torch.float64, device=device),
        },
        batch_size=[n],
        device=device,
    )
    tree = ClusterTree.from_points(points, leaf_size=2, areas=areas)
    with pytest.raises(TypeError, match="different dtypes"):
        tree.compute_source_aggregates(points, areas, data)


# ---------------------------------------------------------------------------
# Edge cases and validation
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


def test_validate_rejects_corrupted_plan(device):
    """Plan validation fails loudly for shape and broadcast corruption."""
    tree = ClusterTree.from_points(_points(10, 3, device, seed=23))
    plan = tree.find_dual_interaction_pairs(tree, theta=0.0)
    plan.near_source_ids = plan.near_source_ids[:-1]
    with pytest.raises(ValueError, match="Shape mismatch"):
        plan.validate()

    target_points, source_points = _zero_survivor_clouds(device)
    target_tree = ClusterTree.from_points(target_points, leaf_size=8)
    source_tree = ClusterTree.from_points(source_points, leaf_size=8)
    plan = source_tree.find_dual_interaction_pairs(target_tree, theta=1.0)
    assert plan.n_fn > 0 and plan.fn_broadcast_targets.numel() == 0
    plan.fn_broadcast_counts[-1] = 1
    with pytest.raises(ValueError, match="out of bounds"):
        plan.validate()


# ---------------------------------------------------------------------------
# Split rules, admissibility masks, and expanded targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["morton", "midpoint"])
@pytest.mark.parametrize("leaf_size", [1, 3])
def test_split_rules_keep_structure_and_cover(device, split, leaf_size):
    """Both split rules build valid partition trees with exact plan covers."""
    clouds = _adversarial_clouds(device, 3)
    clouds["anisotropic"] = _points(150, 3, device, seed=23) * torch.tensor(
        [10.0, 1.0, 0.1], device=device
    )
    for name, points in clouds.items():
        tree = ClusterTree.from_points(points, leaf_size=leaf_size, split=split)
        is_leaf = tree.leaf_count > 0
        assert (tree.leaf_count[is_leaf] <= leaf_size).all(), name
        assert int(tree.leaf_count.sum()) == points.shape[0], name
        is_internal = tree.node_left_child >= 0
        left = tree.node_left_child[is_internal]
        right = tree.node_right_child[is_internal]
        assert (
            tree.node_range_count[is_internal]
            == tree.node_range_count[left] + tree.node_range_count[right]
        ).all(), name
        assert (tree.node_range_count[left] > 0).all(), name
        assert (tree.node_range_count[right] > 0).all(), name
        for theta in (0.7, 2.0):
            for expand in (False, True):
                plan = tree.find_dual_interaction_pairs(
                    tree, theta=theta, expand_far_targets=expand
                )
                n = points.shape[0]
                count = _coverage_counts(plan, tree, tree, n, n)
                assert (count == 1).all(), (name, theta, expand)


def test_morton_split_tightens_boxes(device):
    """Splitting at Morton-cell boundaries opens fewer pairs on a surface."""
    points = _points(4000, 3, device, seed=24)
    points = points / points.norm(dim=-1, keepdim=True)  # a sphere
    sizes = {}
    for split in ("morton", "midpoint"):
        tree = ClusterTree.from_points(points, split=split)
        plan = tree.find_dual_interaction_pairs(
            tree, theta=1.0, expand_far_targets=True
        )
        sizes[split] = plan.n_near + plan.n_nf
    assert sizes["morton"] < 0.8 * sizes["midpoint"]


def test_invalid_split_raises(device):
    with pytest.raises(ValueError, match="split"):
        ClusterTree.from_points(_points(10, 3, device), split="radix")  # type: ignore[arg-type]


@pytest.mark.parametrize("expand_far_targets", [False, True])
def test_source_admissible_mask(device, expand_far_targets):
    """Inadmissible source nodes are never admitted; the cover stays exact."""
    points = _points(300, 3, device, seed=25)
    generator = torch.Generator(device="cpu").manual_seed(26)
    for leaf_size in (1, 4):
        tree = ClusterTree.from_points(points, leaf_size=leaf_size)
        admissible = (torch.rand(tree.n_nodes, generator=generator) > 0.5).to(device)
        plan = tree.find_dual_interaction_pairs(
            tree,
            theta=1.5,
            expand_far_targets=expand_far_targets,
            source_admissible=admissible,
        )
        admitted = torch.cat([plan.nf_source_node_ids, plan.far_source_node_ids])
        assert admissible[admitted].all()
        n = points.shape[0]
        assert (_coverage_counts(plan, tree, tree, n, n) == 1).all()


@pytest.mark.parametrize("leaf_size", [1, 4, 16])
def test_expand_far_targets_has_no_target_side_approximation(device, leaf_size):
    """With expand_far_targets, no entry evaluates the kernel at a target centroid."""
    targets = _points(200, 3, device, seed=28)
    sources = _points(250, 3, device, seed=29) + 0.5
    target_tree = ClusterTree.from_points(targets, leaf_size=leaf_size)
    source_tree = ClusterTree.from_points(sources, leaf_size=leaf_size)
    plan = source_tree.find_dual_interaction_pairs(
        target_tree, theta=1.5, expand_far_targets=True
    )
    assert plan.n_far_nodes == 0
    assert plan.n_fn == 0
    count = _coverage_counts(plan, target_tree, source_tree, 200, 250)
    assert (count == 1).all()


# ---------------------------------------------------------------------------
# Point plans: the single-tree walk from individual targets
# ---------------------------------------------------------------------------


def _point_plan_counts(plan, tree, n_targets, n_sources):
    """Expand a point plan's two streams into a dense (n_targets, n_sources) count."""
    count = torch.zeros(
        n_targets, n_sources, dtype=torch.long, device=tree.source_points.device
    )
    if plan.n_near:
        count.index_put_(
            (plan.near_target_ids, plan.near_source_ids),
            torch.ones_like(plan.near_target_ids),
            accumulate=True,
        )
    if plan.n_far:
        positions, pair_ids = _ragged_arange(
            tree.node_range_start[plan.far_source_node_ids],
            tree.node_range_count[plan.far_source_node_ids],
        )
        count.index_put_(
            (plan.far_target_ids[pair_ids], tree.sorted_source_order[positions]),
            torch.ones_like(positions),
            accumulate=True,
        )
    return count


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.5])
@pytest.mark.parametrize("leaf_size", [1, 4])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_point_plan_covers_every_pair_exactly_once(device, theta, leaf_size, n_dims):
    """Near pairs plus expanded far nodes cover every (target, source) pair once,
    for targets inside the cloud and far outside it."""
    n_t, n_s = 61, 47
    sources = _points(n_s, n_dims, device, seed=31)
    targets = _points(n_t, n_dims, device, seed=32)
    targets[::7] += 30.0  # distant targets admit coarse nodes
    tree = ClusterTree.from_points(sources, leaf_size=leaf_size)
    plan = tree.find_point_interaction_pairs(targets, theta=theta)
    count = _point_plan_counts(plan, tree, n_t, n_s)
    assert (count == 1).all(), f"min={count.min()}, max={count.max()}"
    if theta == 0.0:
        assert plan.n_far == 0 and plan.n_near == n_t * n_s
    else:
        assert plan.n_far > 0


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.5])
def test_point_plan_with_a_forest_source(device, theta):
    """A forest source plans what one walk per source tree would, joined."""
    trees = _forest_trees(device, 60, [30, 0, 20, 7])
    forest = ClusterTree.merge(trees)
    targets = _points(41, 3, device, seed=33)
    targets[::5] += 30.0
    plan = forest.find_point_interaction_pairs(targets, theta=theta)
    count = _point_plan_counts(plan, forest, targets.shape[0], forest.n_sources)
    assert (count == 1).all(), f"min={count.min()}, max={count.max()}"

    def entries(plan, point_offset=0, node_offset=0):
        near = zip(plan.near_target_ids.tolist(), plan.near_source_ids.tolist())
        far = zip(plan.far_target_ids.tolist(), plan.far_source_node_ids.tolist())
        return (
            sorted((t, s + point_offset) for t, s in near),
            sorted((t, n + node_offset) for t, n in far),
        )

    near, far = [], []
    for tree, (sp, sn) in zip(trees, _forest_offsets(trees)):
        part = entries(tree.find_point_interaction_pairs(targets, theta=theta), sp, sn)
        near += part[0]
        far += part[1]
    assert entries(plan) == (sorted(near), sorted(far))


def test_point_plan_far_admissions_satisfy_mac_in_float64(device):
    """Every far entry satisfies D_S / r < theta, recomputed in float64."""
    theta = 0.7
    sources = (_points(90, 3, device, seed=33) * 2.0).add(1.5)
    targets = (_points(70, 3, device, seed=34) * 2.0).add(0.5)
    tree = ClusterTree.from_points(sources, leaf_size=4)
    plan = tree.find_point_interaction_pairs(targets, theta=theta)
    assert plan.n_far > 0 and plan.n_near > 0
    distance_squared = _aabb_distance_squared_float64(
        targets[plan.far_target_ids],
        tree.node_aabb_min[plan.far_source_node_ids],
        tree.node_aabb_max[plan.far_source_node_ids],
    )
    assert (
        distance_squared * theta**2 * (1.0 + _MAC_RTOL)
        > tree.node_diameter_sq[plan.far_source_node_ids].double()
    ).all()


def test_point_plan_of_a_subset_is_the_subset_of_the_plan(device):
    """A target's entries depend on the tree and the target alone."""
    sources = _points(80, 3, device, seed=35)
    targets = _points(50, 3, device, seed=36)
    tree = ClusterTree.from_points(sources, leaf_size=2)
    full = tree.find_point_interaction_pairs(targets, theta=0.8)
    pick = torch.tensor([3, 17, 41], device=device)
    part = tree.find_point_interaction_pairs(targets[pick], theta=0.8)

    def entries(plan, t_ids, remap):
        near = {
            (int(remap[t]), int(s))
            for t, s in zip(
                plan.near_target_ids.tolist(), plan.near_source_ids.tolist()
            )
            if int(t) in t_ids
        }
        far = {
            (int(remap[t]), int(s))
            for t, s in zip(
                plan.far_target_ids.tolist(), plan.far_source_node_ids.tolist()
            )
            if int(t) in t_ids
        }
        return near, far

    remap_full = {int(t): i for i, t in enumerate(pick.tolist())}
    remap_part = {i: i for i in range(len(pick))}
    assert entries(full, set(remap_full), remap_full) == entries(
        part, set(remap_part), remap_part
    )


def test_point_plan_source_admissible_mask(device):
    """Masked source nodes never appear in the far stream, and the cover holds."""
    sources = _points(64, 3, device, seed=37)
    targets = _points(40, 3, device, seed=38) + 4.0
    tree = ClusterTree.from_points(sources, leaf_size=2)
    g = torch.Generator(device="cpu").manual_seed(39)
    admissible = (torch.rand(tree.n_nodes, generator=g) > 0.5).to(device)
    plan = tree.find_point_interaction_pairs(
        targets, theta=1.0, source_admissible=admissible
    )
    assert admissible[plan.far_source_node_ids].all()
    assert (_point_plan_counts(plan, tree, 40, 64) == 1).all()
    plain = tree.find_point_interaction_pairs(targets, theta=1.0)
    assert plan.n_near + plan.n_far >= plain.n_near + plain.n_far


def test_point_plan_nan_target_and_source_keep_exact_cover(device):
    """A NaN target coordinate or a NaN source coordinate opens its pairs down to
    the exact stream instead of dropping them."""
    sources = _points(50, 3, device, seed=40)
    targets = _points(30, 3, device, seed=41)
    targets[5, 1] = float("nan")
    tree = ClusterTree.from_points(sources, leaf_size=2)
    plan = tree.find_point_interaction_pairs(targets, theta=1.0)
    count = _point_plan_counts(plan, tree, 30, 50)
    assert (count == 1).all()
    assert (plan.far_target_ids != 5).all()
    sources_nan = sources.clone()
    sources_nan[7, 0] = float("nan")
    tree_nan = ClusterTree.from_points(sources_nan, leaf_size=2)
    plan = tree_nan.find_point_interaction_pairs(
        _points(30, 3, device, seed=41), theta=1.0
    )
    assert (_point_plan_counts(plan, tree_nan, 30, 50) == 1).all()


def test_point_plan_empty_cases_and_validation(device):
    """Empty trees or target sets give empty plans; validate rejects a corrupted one."""
    tree = ClusterTree.from_points(_points(10, 3, device, seed=42))
    empty = tree.find_point_interaction_pairs(
        torch.empty(0, 3, device=device), theta=1.0
    )
    assert empty.n_near == 0 and empty.n_far == 0
    none = ClusterTree.from_points(torch.empty(0, 3, device=device))
    plan = none.find_point_interaction_pairs(_points(4, 3, device, seed=43), theta=1.0)
    assert plan.n_near == 0 and plan.n_far == 0
    plan = tree.find_point_interaction_pairs(_points(4, 3, device, seed=43), theta=0.0)
    plan.validate()
    assert plan.n_near == 40
    plan.near_source_ids = plan.near_source_ids[:-1]
    with pytest.raises(ValueError, match="Shape mismatch"):
        plan.validate()


def test_point_plan_is_cheaper_than_the_dual_plan_for_sparse_targets(device):
    """Scattered targets around a surface-like cloud: the point plan needs fewer
    kernel evaluations than the dual plan with expanded far targets, which charges
    each target leaf's own extent."""
    g = torch.Generator(device="cpu").manual_seed(44)
    angles = torch.rand(2000, generator=g) * 6.283185307179586
    surface = torch.stack(
        [angles.cos(), angles.sin(), 0.2 * torch.randn(2000, generator=g)], -1
    )
    targets = torch.randn(1500, 3, generator=g) * 6.0
    tree = ClusterTree.from_points(surface.to(device), leaf_size=4)
    point_plan = tree.find_point_interaction_pairs(targets.to(device), theta=1.0)
    target_tree = ClusterTree.from_points(targets.to(device), leaf_size=4)
    dual = tree.find_dual_interaction_pairs(
        target_tree, theta=1.0, expand_far_targets=True
    )
    assert point_plan.n_near + point_plan.n_far < dual.n_near + dual.n_nf + dual.n_fn
    assert (_point_plan_counts(point_plan, tree, 1500, 2000) == 1).all()


# ---------------------------------------------------------------------------
# Warp build and grouped build
# ---------------------------------------------------------------------------

_TREE_FIELDS = (
    "node_aabb_min",
    "node_aabb_max",
    "node_diameter_sq",
    "node_left_child",
    "node_right_child",
    "leaf_start",
    "leaf_count",
    "node_range_start",
    "node_range_count",
    "node_total_area",
    "sorted_source_order",
    "source_points",
    "max_depth",
    "root_ids",
)


def _bitwise(a, b):
    """Same dtype, shape and bits (so NaNs and signed zeros compare too)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        ints = {8: torch.int64, 4: torch.int32, 2: torch.int16}[a.element_size()]
        a, b = a.contiguous().view(ints), b.contiguous().view(ints)
    return torch.equal(a, b)


def _tree_differences(a, b):
    """The fields in which two trees differ in any bit."""
    return [k for k in _TREE_FIELDS if not _bitwise(getattr(a, k), getattr(b, k))]


def _build_cases(n_dims, device, dtype):
    """A random cloud, the same with half its points on one point (equal Morton
    codes, midpoint splits), and tiny ones (one point, no larger than a leaf)."""
    points = _points(1500, n_dims, device, seed=21, dtype=dtype)
    areas = _areas(1500, device, seed=22, dtype=dtype)
    duplicates = points.clone()
    duplicates[:700] = duplicates[0]
    yield points, areas
    yield duplicates, None
    for n in (1, 3, 9, 40):
        yield points[:n], areas[:n]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("n_dims", [2, 3])
def test_warp_build_is_the_torch_build(device, n_dims, dtype):
    """The Warp build (on Warp's CPU backend for CPU points) gives the torch
    build's tree in every bit and every field, at three leaf sizes."""
    for points, areas in _build_cases(n_dims, device, dtype):
        for leaf_size in (1, 4, 8):
            trees = [
                ClusterTree.from_points(
                    points, leaf_size=leaf_size, areas=areas, implementation=impl
                )
                for impl in ("torch", "warp")
            ]
            assert not _tree_differences(*trees), (points.shape[0], leaf_size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_warp_build_is_the_torch_build_under_autocast(device, n_dims, dtype):
    """Under bfloat16 autocast the Warp build follows the precision the torch
    build computes in (CUDA's autocast runs ``pow`` and ``sum`` in float32)."""
    for points, areas in _build_cases(n_dims, device, dtype):
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            trees = [
                ClusterTree.from_points(
                    points, leaf_size=4, areas=areas, implementation=impl
                )
                for impl in ("torch", "warp")
            ]
        assert not _tree_differences(*trees), points.shape[0]


def _group_case(device, n_dims=3, seed=23):
    """Points in groups 0..6: group 3 empty, group 5 one point, group 6 half on
    one point."""
    n = 800
    points = _points(n, n_dims, device, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    groups = torch.randint(0, 5, (n,), generator=g)
    groups[groups == 3] = 4
    groups[7] = 5
    groups[100:180] = 6
    points[100:140] = points[100]
    groups = groups.to(device)
    return points, groups, _areas(n, device, seed=seed + 1), 7


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("leaf_size", [1, 4])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_point_groups_are_the_per_group_trees_and_their_merge(
    device, implementation, leaf_size, n_dims
):
    """``trees[g]`` is ``from_points`` of group ``g`` in every bit (an empty tree
    for an empty group), and the forest is their merge in every bit, except
    that its order indexes the caller's points."""
    points, groups, areas, n_groups = _group_case(device, n_dims)
    for a in (areas, None):
        forest, trees = ClusterTree.from_point_groups(
            points,
            groups,
            n_groups=n_groups,
            leaf_size=leaf_size,
            areas=a,
            implementation=implementation,
        )
        assert len(trees) == n_groups
        assert trees[-1] is trees[n_groups - 1]
        assert all(a is b for a, b in zip(trees[1:3], (trees[1], trees[2])))
        for k, tree in enumerate(trees):
            mask = groups == k
            ref = ClusterTree.from_points(
                points[mask],
                leaf_size=leaf_size,
                areas=None if a is None else a[mask],
                implementation="torch",
            )
            assert not _tree_differences(tree, ref), k
        assert trees[3].n_nodes == 0
        merged = ClusterTree.merge(trees)
        assert set(_tree_differences(forest, merged)) == {
            "sorted_source_order",
            "source_points",
        }
        assert _bitwise(forest.source_points, points)
        by_group = torch.argsort(groups, stable=True)
        assert torch.equal(
            forest.sorted_source_order, by_group[merged.sorted_source_order]
        )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_point_groups_of_one_group_and_of_no_points(device, implementation):
    """One group is one tree: ``from_points``' own; no points give empty trees."""
    points = _points(300, 3, device, seed=25)
    forest, trees = ClusterTree.from_point_groups(
        points,
        torch.zeros(300, dtype=torch.long, device=device),
        n_groups=1,
        leaf_size=4,
        implementation=implementation,
    )
    ref = ClusterTree.from_points(points, leaf_size=4, implementation="torch")
    assert not _tree_differences(forest, ref)
    assert len(trees) == 1 and not _tree_differences(trees[0], ref)
    forest, trees = ClusterTree.from_point_groups(
        points[:0],
        torch.zeros(0, dtype=torch.long, device=device),
        n_groups=2,
        implementation=implementation,
    )
    assert forest.n_nodes == 0 and forest.root_ids.numel() == 0
    assert [t.n_nodes for t in trees] == [0, 0]


def test_builds_refuse_points_and_areas_of_other_shapes(device):
    """Points that are not ``(n_points, n_dims)`` and areas that are not
    ``(n_points,)`` are refused before either build runs."""
    points = _points(50, 3, device, seed=66)
    groups = torch.zeros(50, dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="points must have shape"):
        ClusterTree.from_points(points[:, 0])
    with pytest.raises(ValueError, match="points must have shape"):
        ClusterTree.from_point_groups(points[:, 0], groups, n_groups=1)
    with pytest.raises(ValueError, match="areas must have shape"):
        ClusterTree.from_points(points, areas=torch.ones(50, 1, device=device))
    with pytest.raises(ValueError, match="areas must have shape"):
        ClusterTree.from_point_groups(
            points, groups, n_groups=1, areas=torch.ones(49, device=device)
        )


def test_build_implementation_resolution(monkeypatch):
    """``None`` builds with torch on CPU and with Warp on CUDA when Warp is
    available and the points fit; a forced name is kept and an unknown one
    refused."""
    from types import SimpleNamespace

    import physicsnemo.mesh.spatial.cluster_tree as ct

    cpu = torch.zeros(10, 3)
    cuda = SimpleNamespace(is_cuda=True, shape=(10, 3), device=torch.device("cuda"))
    assert ct._resolve_build(None, cpu) == "torch"
    assert ct._resolve_build("warp", cpu) == "warp"
    assert ct._resolve_build(None, cuda) == "warp"
    with pytest.raises(ValueError, match="implementation must be"):
        ct._resolve_build("gpu", cpu)
    monkeypatch.setattr(ct, "_warp_available", lambda: False)
    assert ct._resolve_build(None, cuda) == "torch"
    monkeypatch.setattr(ct, "_warp_available", lambda: True)
    ### 10 points fit, but their node boxes (up to 2 * 10 * 3 entries) do not.
    monkeypatch.setattr(ct, "_WARP_MAX_ELEMENTS", 50)
    assert ct._resolve_build(None, cuda) == "torch"
    monkeypatch.setattr(ct, "_WARP_MAX_ELEMENTS", 10)
    assert ct._resolve_build(None, cuda) == "torch"
    with pytest.raises(ValueError, match="fewer than 2\\*\\*31"):
        ct._resolve_build("warp", cpu)

    ### On CPU, ``None`` never reaches the Warp build.
    def no_warp(*args, **kwargs):
        raise AssertionError("the Warp build ran")

    monkeypatch.setattr(ct.ClusterTree, "_from_point_groups_warp", no_warp)
    ClusterTree.from_points(_points(50, 3, "cpu"))
    ClusterTree.from_point_groups(
        _points(50, 3, "cpu"), torch.zeros(50, dtype=torch.long), n_groups=1
    )


def test_warp_build_refuses_or_falls_back_where_it_cannot_build(monkeypatch):
    """A forced Warp build refuses the midpoint split and a forest too deep
    for its node keys; under ``None`` such a forest is built by torch."""
    import physicsnemo.mesh.spatial._lbvh_warp as lbvh_warp
    import physicsnemo.mesh.spatial.cluster_tree as ct

    points = _points(300, 3, "cpu", seed=26)
    with pytest.raises(ValueError, match="midpoint"):
        ClusterTree.from_points(points, split="midpoint", implementation="warp")
    with pytest.raises(TypeError, match="integer"):
        ClusterTree.from_point_groups(points, torch.zeros(300), n_groups=1)
    ref = ClusterTree.from_points(points, leaf_size=4, implementation="torch")
    groups = (points[:, 0] > 0).long()
    ref_forest, _ = ClusterTree.from_point_groups(
        points, groups, n_groups=2, leaf_size=4, implementation="torch"
    )
    monkeypatch.setattr(lbvh_warp, "_DEPTH_BINS", 4)
    with pytest.raises(ValueError, match="too deep"):
        ClusterTree.from_points(points, leaf_size=4, implementation="warp")
    monkeypatch.setattr(ct, "_resolve_build", lambda *args: "warp")
    tree = ClusterTree.from_points(points, leaf_size=4)
    assert not _tree_differences(tree, ref)
    forest, _ = ClusterTree.from_point_groups(points, groups, n_groups=2, leaf_size=4)
    assert not _tree_differences(forest, ref_forest)


@pytest.mark.cuda
def test_warp_build_on_cuda_is_the_default_and_the_torch_build():
    """On CUDA, ``None`` builds with Warp, which gives the torch build's tree."""
    import physicsnemo.mesh.spatial.cluster_tree as ct

    points = _points(5000, 3, "cuda", seed=27)
    assert ct._resolve_build(None, points) == "warp"
    trees = [
        ClusterTree.from_points(points, leaf_size=8, implementation=impl)
        for impl in (None, "torch")
    ]
    assert not _tree_differences(*trees)


@pytest.mark.cuda
def test_warp_build_non_default_cuda_stream():
    """The Warp build runs on torch's current, non-default CUDA stream."""
    points, groups, areas, n_groups = _group_case("cuda")
    baseline, _ = ClusterTree.from_point_groups(
        points, groups, n_groups=n_groups, areas=areas, implementation="torch"
    )
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(int(1e8))  # ~100 ms hazard window on the stream
        points, groups, areas, n_groups = _group_case("cuda")
        forest, _ = ClusterTree.from_point_groups(
            points, groups, n_groups=n_groups, areas=areas, implementation="warp"
        )
        forest = forest.clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert not _tree_differences(forest, baseline)


def test_importing_the_package_registers_no_warp_tree_kernels():
    """The Warp modules are imported on first use, not with the package."""
    import subprocess
    import sys

    code = (
        "import sys, physicsnemo.mesh.spatial; "
        "print(any(m.startswith('physicsnemo.mesh.spatial._') and m.endswith('_warp') "
        "for m in sys.modules))"
    )
    out = subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# Warp point walk and the point plan's order
# ---------------------------------------------------------------------------

_POINT_PLAN_FIELDS = (
    "near_target_ids",
    "near_source_ids",
    "far_target_ids",
    "far_source_node_ids",
)


def _same_point_plans(a, b):
    return all(torch.equal(getattr(a, k), getattr(b, k)) for k in _POINT_PLAN_FIELDS)


def _level_major_point_walk(tree, targets, theta):
    """The point walk of one tree in its pre-sort order: level by level, each
    level the opened nodes' left children then their right children, the
    targets interleaved (the reference for the order contract)."""
    box_min, box_max = tree.node_aabb_min, tree.node_aabb_max
    n_dims = tree.n_spatial_dims
    points = targets.to(box_min.dtype)
    active_t = torch.arange(targets.shape[0], device=targets.device)
    active_s = torch.zeros_like(active_t)
    near = [torch.empty((2, 0), dtype=torch.long, device=targets.device)]
    far = list(near)
    while active_t.numel():
        p = points[active_t]
        gap = torch.clamp(
            torch.maximum(p - box_max[active_s], box_min[active_s] - p), min=0
        )
        is_far = gap.pow(2).sum(dim=-1) * theta**2 > tree.node_diameter_sq[active_s]
        left = tree.node_left_child[active_s]
        right = tree.node_right_child[active_s]
        far.append(torch.stack([active_t, active_s])[:, is_far])
        hit = ~is_far & (left < 0)
        for t, s in zip(active_t[hit].tolist(), active_s[hit].tolist()):
            start, count = int(tree.leaf_start[s]), int(tree.leaf_count[s])
            sources = tree.sorted_source_order[start : start + count]
            near.append(torch.stack([torch.full_like(sources, t), sources]))
        split = ~is_far & (left >= 0)
        active_t = torch.cat([active_t[split], active_t[split]])
        active_s = torch.cat([left[split], right[split]])
    assert n_dims == targets.shape[1]
    return torch.cat(near, dim=1), torch.cat(far, dim=1)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_point_plan_order_is_the_per_tree_walks_joined(device, implementation, n_dims):
    """The forest walk's near and far streams are the near and far entries of
    one level-major walk per tree, listed tree by tree (tree 0 near, tree 0 far,
    tree 1 near, ...) and stably sorted by target: ISLA's per-tree lists."""
    points, groups, _, n_groups = _group_case(device, n_dims)
    forest, trees = ClusterTree.from_point_groups(
        points, groups, n_groups=n_groups, leaf_size=4
    )
    targets = _points(90, n_dims, device, seed=51) * 2.0
    theta = 0.8
    by_group = torch.argsort(groups, stable=True)
    entries, point_offset, node_offset = [], 0, 0
    for tree in trees:
        if tree.n_nodes:
            near, far = _level_major_point_walk(tree, targets, theta)
            near = torch.stack([near[0], by_group[near[1] + point_offset]])
            for kind, part in (
                (0, near),
                (1, far + torch.tensor([[0], [node_offset]], device=device)),
            ):
                entries.append(torch.cat([part, torch.full_like(part[:1], kind)]))
        point_offset += tree.n_sources
        node_offset += tree.n_nodes
    joined = torch.cat(entries, dim=1)
    joined = joined[:, torch.argsort(joined[0], stable=True)]
    plan = forest.find_point_interaction_pairs(
        targets, theta, implementation=implementation
    )
    assert plan.n_near > 0 and plan.n_far > 0
    near, far = joined[:2, joined[2] == 0], joined[:2, joined[2] == 1]
    assert torch.equal(plan.near_target_ids, near[0])
    assert torch.equal(plan.near_source_ids, near[1])
    assert torch.equal(plan.far_target_ids, far[0])
    assert torch.equal(plan.far_source_node_ids, far[1])


@pytest.mark.parametrize("theta", [0.5, 1.0])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, "bfloat16 autocast"]
)
@pytest.mark.parametrize("n_dims", [2, 3])
def test_warp_walk_is_the_torch_walk(device, n_dims, dtype, theta):
    """The Warp walk (on Warp's CPU backend for CPU targets) gives the torch
    walk's plan entry for entry, for a forest with an admissibility mask, for
    targets near and far, a NaN target (opened down to every source) and an
    infinite one; under bfloat16 autocast too, in whatever precision the tree
    is built. ``count_only`` gives each target's numbers of entries."""
    autocast = dtype == "bfloat16 autocast"
    dtype = torch.bfloat16 if autocast else dtype
    points, groups, areas, n_groups = _group_case(device, n_dims, seed=52)
    targets = _points(150, n_dims, device, seed=53) * 3.0
    targets[5] = float("nan")
    targets[6, 0] = float("inf")
    points, targets = points.to(dtype), targets.to(dtype)
    with torch.autocast(
        torch.device(device).type, dtype=torch.bfloat16, enabled=autocast
    ):
        forest, _ = ClusterTree.from_point_groups(
            points, groups, n_groups=n_groups, leaf_size=4
        )
        g = torch.Generator(device="cpu").manual_seed(54)
        admissible = (torch.rand(forest.n_nodes, generator=g) > 0.2).to(device)
        for mask in (None, admissible):
            kwargs = dict(theta=theta, source_admissible=mask)
            torch_plan = forest.find_point_interaction_pairs(
                targets, implementation="torch", **kwargs
            )
            warp_plan = forest.find_point_interaction_pairs(
                targets, implementation="warp", **kwargs
            )
            assert _same_point_plans(warp_plan, torch_plan)
            assert torch_plan.n_far > 0
            for impl in ("torch", "warp"):
                n_near, n_far = forest.find_point_interaction_pairs(
                    targets, implementation=impl, count_only=True, **kwargs
                )
                assert torch.equal(
                    n_near, torch_plan.near_target_ids.bincount(minlength=150)
                )
                assert torch.equal(
                    n_far, torch_plan.far_target_ids.bincount(minlength=150)
                )
    nan_sources = torch_plan.near_source_ids[torch_plan.near_target_ids == 5]
    assert torch.equal(nan_sources.sort().values, torch.arange(800, device=device))


@pytest.mark.parametrize(
    "boxes,diagonals,autocast",
    [
        (torch.float32, torch.float32, False),
        (torch.float64, torch.float64, False),
        (torch.bfloat16, torch.bfloat16, False),
        (torch.float16, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
    ],
    ids=["float32", "float64", "bfloat16", "float16", "bfloat16_boxes_under_autocast"],
)
def test_warp_walk_opening_test_is_the_torch_walks_at_the_threshold(
    device, boxes, diagonals, autocast
):
    """Every node a one-leaf tree whose squared diagonal is the torch walk's own
    ``min_dist_sq * theta^2`` for one point (not far) or one ulp below it (far):
    each (point, node) decision of the Warp walk is the torch walk's, so it
    rounds the test as torch does (its sum order, and the steps autocast runs
    in float32 for a tree built under it)."""
    n, d, theta, dev = 256, 3, 0.7, torch.device(device)
    g = torch.Generator().manual_seed(0)
    f64 = torch.float64
    scale = torch.logspace(-3, 3, n, dtype=f64)[torch.randperm(n, generator=g)][:, None]
    p = torch.randn(n, d, generator=g, dtype=f64) * scale
    c = torch.randn(n, d, generator=g, dtype=f64) * 3 * scale
    half = torch.rand(n, d, generator=g, dtype=f64) * scale
    p, lo, hi = (x.to(boxes).to(dev) for x in (p, c - half, c + half))
    ar, one = torch.arange(n, device=dev), torch.ones(n, dtype=torch.long, device=dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
        pt = p.t().contiguous()
        gap = torch.clamp(
            torch.maximum(pt - hi.t().contiguous(), lo.t().contiguous() - pt), min=0
        )
        at = (gap.pow(2).sum(dim=0) * theta**2).to(diagonals)
        bits = {
            torch.bfloat16: torch.int16,
            torch.float16: torch.int16,
            torch.float32: torch.int32,
            f64: torch.int64,
        }
        below = torch.where(
            at > 0, (at.view(bits[diagonals]) - 1).view(diagonals), at
        )  # one ulp toward zero
        diam_sq = torch.where(ar % 2 == 0, at, below)
        leaves = torch.full((n,), -1, device=dev)
        tree = ClusterTree(
            node_aabb_min=lo,
            node_aabb_max=hi,
            node_diameter_sq=diam_sq,
            node_left_child=leaves,
            node_right_child=leaves,
            leaf_start=ar,
            leaf_count=one,
            node_range_start=ar,
            node_range_count=one,
            node_total_area=one.float(),
            sorted_source_order=ar,
            source_points=p.float(),
            max_depth=torch.zeros((), dtype=torch.long, device=dev),
            root_ids=ar,
            batch_size=torch.Size([]),
        )
        plans = [
            tree.find_point_interaction_pairs(p, theta=theta, implementation=impl)
            for impl in ("torch", "warp")
        ]
    far = [torch.zeros(n, n, dtype=torch.bool, device=dev) for _ in plans]
    for f, plan in zip(far, plans):
        f[plan.far_target_ids, plan.far_source_node_ids] = True
    assert 0 < int(far[0][ar, ar].sum()) < n
    assert torch.equal(far[1], far[0])
    assert _same_point_plans(*plans)


def _left_deep_tree(depth, device):
    """A tree of ``depth`` internal nodes on a line, each the left child of the
    one before, with a one-point leaf on the right: a depth-first walk keeps
    every right leaf on its stack."""
    n = depth + 1
    points = torch.zeros(n, 3, device=device)
    points[:, 0] = torch.arange(n, device=device, dtype=torch.float32)
    n_nodes = 2 * depth + 1
    left = torch.full((n_nodes,), -1, dtype=torch.long)
    right = left.clone()
    leaf_start, leaf_count = left.clone(), torch.zeros(n_nodes, dtype=torch.long)
    start, count = torch.zeros(n_nodes, dtype=torch.long), torch.zeros_like(left)
    for j in range(depth):  # internal node j covers points [0, n - j)
        left[j] = j + 1 if j + 1 < depth else 2 * depth
        right[j] = depth + j
        count[j] = n - j
        start[depth + j] = leaf_start[depth + j] = n - j - 1
        count[depth + j] = leaf_count[depth + j] = 1
    start[2 * depth] = leaf_start[2 * depth] = 0
    count[2 * depth] = leaf_count[2 * depth] = 1
    lo = points[start]
    hi = points[start + count - 1]
    return ClusterTree(
        node_aabb_min=lo,
        node_aabb_max=hi,
        node_diameter_sq=(hi - lo).pow(2).sum(-1),
        node_left_child=left.to(device),
        node_right_child=right.to(device),
        leaf_start=leaf_start.to(device),
        leaf_count=leaf_count.to(device),
        node_range_start=start.to(device),
        node_range_count=count.to(device),
        node_total_area=count.float().to(device),
        sorted_source_order=torch.arange(n, device=device),
        source_points=points,
        max_depth=torch.tensor(depth, device=device),
        root_ids=torch.zeros(1, dtype=torch.long, device=device),
        batch_size=torch.Size([]),
    )


def test_warp_walk_refuses_or_falls_back_past_its_limits(monkeypatch):
    """A forced Warp walk refuses trees it cannot take (4-D), a walk too deep
    for its stack and streams past Warp's array limit; under ``None`` such
    walks run in torch."""
    import physicsnemo.mesh.spatial.cluster_tree as ct

    ### The torch walk takes trees up to about 64 levels deeper than a Morton
    ### tree of their size can be, the Warp walk's stack up to about 95.
    targets = torch.zeros(3, 3)
    tree = _left_deep_tree(60, "cpu")
    ref = tree.find_point_interaction_pairs(targets, implementation="torch")
    counts = ref.near_target_ids.bincount(minlength=3)
    assert (counts + ref.far_target_ids.bincount(minlength=3) == 61).all()
    plan = tree.find_point_interaction_pairs(targets, implementation="warp")
    assert _same_point_plans(plan, ref)
    tree = _left_deep_tree(100, "cpu")
    with pytest.raises(ValueError, match="too deep"):
        tree.find_point_interaction_pairs(targets, implementation="warp")
    with pytest.raises(ValueError, match="too deep"):
        tree.find_point_interaction_pairs(
            targets, implementation="warp", count_only=True
        )
    with pytest.raises(ValueError, match="2-D or 3-D"):
        ClusterTree.from_points(torch.rand(20, 4)).find_point_interaction_pairs(
            torch.rand(5, 4), implementation="warp"
        )
    sources = _points(300, 3, "cpu", seed=55)
    tree4 = ClusterTree.from_points(sources, leaf_size=4)
    targets4 = _points(40, 3, "cpu", seed=56) * 3.0
    ref4 = tree4.find_point_interaction_pairs(targets4, implementation="torch")
    monkeypatch.setattr(ct, "_WARP_MAX_ELEMENTS", ref4.n_near)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        tree4.find_point_interaction_pairs(targets4, implementation="warp")
    monkeypatch.setattr(ct, "_resolve_walk", lambda *args: "warp")
    assert _same_point_plans(tree4.find_point_interaction_pairs(targets4), ref4)


def test_walk_implementation_resolution(monkeypatch):
    """``None`` walks with torch on CPU and for trees Warp cannot take, and
    with Warp for other CUDA targets; unknown names are refused."""
    from types import SimpleNamespace

    import physicsnemo.mesh.spatial.cluster_tree as ct

    tree = ClusterTree.from_points(_points(30, 3, "cpu"))
    cpu = torch.zeros(10, 3)
    cuda = SimpleNamespace(is_cuda=True, shape=(10, 3), device=torch.device("cuda"))
    assert ct._resolve_walk(None, tree, cpu) == "torch"
    assert ct._resolve_walk("warp", tree, cpu) == "warp"
    assert ct._resolve_walk(None, tree, cuda) == "warp"
    assert ct._resolve_walk(None, tree.double(), cuda) == "warp"
    tree_4d = ClusterTree.from_points(torch.rand(30, 4))
    assert ct._resolve_walk(None, tree_4d, cuda) == "torch"
    with pytest.raises(ValueError, match="implementation must be"):
        ct._resolve_walk("gpu", tree, cpu)
    monkeypatch.setattr(ct, "_warp_available", lambda: False)
    assert ct._resolve_walk(None, tree, cuda) == "torch"


def test_point_plan_count_only_at_theta_zero_and_when_empty(device):
    """``count_only`` without a walk: every source near at ``theta = 0``, no
    entries for an empty tree."""
    tree = ClusterTree.from_points(_points(10, 3, device, seed=57))
    targets = _points(4, 3, device, seed=58)
    n_near, n_far = tree.find_point_interaction_pairs(targets, 0.0, count_only=True)
    assert n_near.tolist() == [10] * 4 and n_far.tolist() == [0] * 4
    empty = ClusterTree.from_points(torch.empty(0, 3, device=device))
    n_near, n_far = empty.find_point_interaction_pairs(targets, count_only=True)
    assert n_near.tolist() == [0] * 4 and n_far.tolist() == [0] * 4


@pytest.mark.parametrize("implementation", [None, "torch", "warp"])
def test_point_walk_refuses_targets_and_masks_that_do_not_fit_the_tree(
    device, implementation
):
    """Targets of another dimension and masks of another length or dtype are
    refused before either walk runs (the Warp kernels would read past their
    arrays)."""
    tree = ClusterTree.from_points(_points(200, 3, device, seed=62), leaf_size=4)
    targets = _points(20, 3, device, seed=63)
    for bad in (targets[:, :2], torch.cat([targets, targets[:, :1]], 1), targets[:, 0]):
        with pytest.raises(ValueError, match="targets must have shape"):
            tree.find_point_interaction_pairs(bad, implementation=implementation)
    for mask in (
        torch.ones(tree.n_nodes - 1, dtype=torch.bool, device=device),
        torch.ones(tree.n_nodes, dtype=torch.uint8, device=device),
    ):
        with pytest.raises(ValueError, match="source_admissible must be"):
            tree.find_point_interaction_pairs(
                targets, source_admissible=mask, implementation=implementation
            )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_dims", [2, 3])
def test_warp_walk_on_cuda_is_the_default_and_the_torch_walk(n_dims, dtype):
    """On CUDA, ``None`` walks with Warp, and the Warp walk gives the torch
    walk's plan entry for entry."""
    import physicsnemo.mesh.spatial.cluster_tree as ct

    points = _points(3000, n_dims, "cuda", seed=59, dtype=dtype)
    targets = _points(2000, n_dims, "cuda", seed=60, dtype=dtype) * 3.0
    tree = ClusterTree.from_points(points, leaf_size=4)
    assert ct._resolve_walk(None, tree, targets) == "warp"
    plans = [
        tree.find_point_interaction_pairs(targets, implementation=impl)
        for impl in ("torch", "warp", None)
    ]
    assert plans[0].n_far > 0
    assert _same_point_plans(plans[1], plans[0])
    assert _same_point_plans(plans[2], plans[0])


@pytest.mark.cuda
def test_warp_walk_non_default_cuda_stream():
    """The Warp walk runs on torch's current, non-default CUDA stream."""
    points, groups, _, n_groups = _group_case("cuda")
    forest, _ = ClusterTree.from_point_groups(points, groups, n_groups=n_groups)
    targets = _points(500, 3, "cuda", seed=61) * 2.0
    baseline = forest.find_point_interaction_pairs(targets, implementation="torch")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(int(1e8))  # ~100 ms hazard window on the stream
        targets = _points(500, 3, "cuda", seed=61) * 2.0
        plan = forest.find_point_interaction_pairs(
            targets, implementation="warp", validate=False
        ).clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert _same_point_plans(plan, baseline)


@pytest.mark.cuda
def test_inputs_on_another_device_than_the_tree_are_refused():
    """Targets or a mask on another device than the tree, and groups on
    another device than the points, are refused instead of reaching a Warp
    kernel (CUDA targets on a CPU tree used to pick the Warp walk)."""
    tree = ClusterTree.from_points(_points(200, 3, "cpu", seed=64), leaf_size=4)
    targets = _points(20, 3, "cpu", seed=65)
    with pytest.raises(ValueError, match="targets must be on cpu"):
        tree.find_point_interaction_pairs(targets.cuda())
    mask = torch.ones(tree.n_nodes, dtype=torch.bool)
    with pytest.raises(ValueError, match="source_admissible must be"):
        tree.to("cuda").find_point_interaction_pairs(
            targets.cuda(), source_admissible=mask
        )
    with pytest.raises(ValueError, match="groups must have shape"):
        ClusterTree.from_point_groups(
            targets.cuda(), torch.zeros(20, dtype=torch.long), n_groups=1
        )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("split", ["morton", "midpoint"])
def test_warp_node_boxes_are_the_torch_node_boxes(device, split, dtype):
    """The one-kernel bottom-up boxes are ``_node_boxes``' in every bit for
    finite points without signed-zero ties, on both topologies and at three
    leaf sizes; a NaN point makes the same boxes NaN."""
    from physicsnemo.mesh.spatial._lbvh import (
        build_lbvh_topology,
        build_morton_topology,
    )
    from physicsnemo.mesh.spatial._lbvh_warp import node_boxes
    from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
    from physicsnemo.mesh.spatial.cluster_tree import _node_boxes

    for n in (1, 2, 9, 700):
        points = _points(n, 3, device, seed=62, dtype=dtype)
        codes, order = _compute_morton_codes(points, isotropic=True).sort(stable=True)
        sorted_points = points[order]
        for leaf_size in (1, 4, 8):
            if split == "morton":
                topo = build_morton_topology(codes, leaf_size)
            else:
                topo = build_lbvh_topology(n, leaf_size, torch.device(device))
            torch_boxes = _node_boxes(topo, sorted_points, leaf_size)
            warp_boxes = node_boxes(topo, sorted_points)
            assert all(map(_bitwise, torch_boxes, warp_boxes)), (n, leaf_size)
            if n > 2:
                with_nan = sorted_points.clone()
                with_nan[n // 2, 1] = float("nan")
                torch_boxes = _node_boxes(topo, with_nan, leaf_size)
                warp_boxes = node_boxes(topo, with_nan)
                for a, b in zip(torch_boxes, warp_boxes):
                    assert torch.equal(a.isnan(), b.isnan())
                    assert a.isnan().any()
                    assert _bitwise(a.nan_to_num(), b.nan_to_num())
