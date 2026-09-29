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


def test_zero_weight_internal_subtree_aggregates(device):
    """A zero-weight *internal* node reduces to exactly zero aggregates.

    Zeroing scattered areas only guarantees zero-weight single-point leaves.
    Here a spatially separated 8-point cluster carries zero weight: it takes the
    highest morton codes, so it is the final contiguous block in sorted order,
    and with ``n = 64`` the index-midpoint splits make ``[56, 64)`` (and its
    4- and 2-point descendants) tree nodes.  With ``leaf_size=1`` these are
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
    tree = ClusterTree.from_points(pts, leaf_size=1, areas=areas)
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
    assert counts == [2, 2, 2, 2, 4, 4, 8], counts


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
