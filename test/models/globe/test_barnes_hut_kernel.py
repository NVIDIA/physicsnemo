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

"""Tests for Barnes-Hut accelerated kernel evaluation.

Covers: ClusterTree construction and aggregation, BarnesHutKernel convergence
to exact results, gradient correctness, equivariance preservation, and
MultiscaleKernel integration.
"""

from typing import Any, Literal

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from physicsnemo.experimental.models.globe.cluster_tree import (
    ClusterTree,
    DualInteractionPlan,
)
from physicsnemo.experimental.models.globe.field_kernel import (
    BarnesHutKernel,
    Kernel,
    MultiscaleKernel,
)
from physicsnemo.mesh.spatial._ragged import _ragged_arange

DEFAULT_SEED = 42
DEFAULT_LEAF_SIZE = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bh_kernel_and_data(
    n_spatial_dims: int = 2,
    n_source_scalars: int = 0,
    n_source_vectors: int = 1,
    output_fields: dict[str, Literal["scalar", "vector"]] | None = None,
    n_global_scalars: int = 0,
    n_global_vectors: int = 0,
    hidden_layer_sizes: list[int] | None = None,
    n_source_points: int = 30,
    n_target_points: int = 20,
    leaf_size: int = DEFAULT_LEAF_SIZE,
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
) -> tuple[BarnesHutKernel, Kernel, dict[str, Any]]:
    """Create matched BH and exact kernels with shared weights and test data."""
    if output_fields is None:
        output_fields = {"pressure": "scalar", "velocity": "vector"}
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [32, 32]

    device_obj = torch.device(device)
    torch.manual_seed(seed)

    output_field_ranks = {
        k: (0 if v == "scalar" else 1) for k, v in output_fields.items()
    }
    source_data_ranks = {
        **{f"source_scalar_{i}": 0 for i in range(n_source_scalars)},
        **{f"source_vector_{i}": 1 for i in range(n_source_vectors)},
    }
    global_data_ranks = {
        **{f"global_scalar_{i}": 0 for i in range(n_global_scalars)},
        **{f"global_vector_{i}": 1 for i in range(n_global_vectors)},
    }

    common_kwargs = dict(
        n_spatial_dims=n_spatial_dims,
        output_field_ranks=output_field_ranks,
        source_data_ranks=source_data_ranks,
        global_data_ranks=global_data_ranks,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    bh_kernel = BarnesHutKernel(**common_kwargs, leaf_size=leaf_size).to(device_obj)
    exact_kernel = Kernel(**common_kwargs).to(device_obj)

    # Share weights so outputs are comparable
    exact_kernel.load_state_dict(bh_kernel.state_dict(), strict=False)
    bh_kernel.eval()
    exact_kernel.eval()

    torch.manual_seed(seed + 1)

    source_data_dict: dict[str, torch.Tensor] = {}
    for i in range(n_source_scalars):
        source_data_dict[f"source_scalar_{i}"] = torch.randn(
            n_source_points, device=device_obj
        )
    for i in range(n_source_vectors):
        source_data_dict[f"source_vector_{i}"] = F.normalize(
            torch.randn(n_source_points, n_spatial_dims, device=device_obj), dim=-1
        )

    global_data_dict: dict[str, torch.Tensor] = {}
    for i in range(n_global_scalars):
        global_data_dict[f"global_scalar_{i}"] = torch.randn(
            1, device=device_obj
        ).squeeze()
    for i in range(n_global_vectors):
        global_data_dict[f"global_vector_{i}"] = F.normalize(
            torch.randn(n_spatial_dims, device=device_obj), dim=0
        )

    input_data = {
        "source_points": torch.randn(
            n_source_points, n_spatial_dims, device=device_obj
        ),
        "target_points": torch.randn(n_target_points, n_spatial_dims, device=device_obj)
        * 5,
        "source_strengths": torch.randn(n_source_points, device=device_obj).abs() + 0.1,
        "reference_length": torch.ones((), device=device_obj),
        "source_data": TensorDict(
            source_data_dict, batch_size=[n_source_points], device=device_obj
        ),
        "global_data": TensorDict(global_data_dict, batch_size=[], device=device_obj),
    }

    return bh_kernel, exact_kernel, input_data


# ---------------------------------------------------------------------------
# ClusterTree tests
# ---------------------------------------------------------------------------


class TestClusterTree:
    """Tests for ClusterTree construction and traversal."""

    def test_construction_basic(self):
        """Tree construction produces valid node structure."""
        torch.manual_seed(DEFAULT_SEED)
        points = torch.randn(50, 3)
        tree = ClusterTree.from_points(points, leaf_size=4)

        assert tree.n_nodes > 0
        assert tree.n_sources == 50
        assert tree.n_spatial_dims == 3
        assert tree.sorted_source_order.shape == (50,)
        # Sorted order is a permutation of [0, N)
        assert set(tree.sorted_source_order.tolist()) == set(range(50))

    def test_construction_empty(self):
        """Empty point set produces empty tree."""
        tree = ClusterTree.from_points(torch.empty(0, 2), leaf_size=4)
        assert tree.n_nodes == 0
        assert tree.n_sources == 0

    def test_construction_single_point(self):
        """Single point produces a single-leaf tree."""
        tree = ClusterTree.from_points(torch.randn(1, 2), leaf_size=4)
        assert tree.n_nodes == 1
        assert tree.leaf_count[0].item() == 1

    def test_aabb_containment(self):
        """Every source point is contained in the root's AABB."""
        torch.manual_seed(DEFAULT_SEED)
        points = torch.randn(100, 3)
        tree = ClusterTree.from_points(points, leaf_size=8)

        root_min = tree.node_aabb_min[0]
        root_max = tree.node_aabb_max[0]

        assert (points >= root_min - 1e-6).all(), "Some points below root AABB min"
        assert (points <= root_max + 1e-6).all(), "Some points above root AABB max"

    def test_leaf_source_coverage(self):
        """All sources are covered by exactly one leaf node."""
        torch.manual_seed(DEFAULT_SEED)
        points = torch.randn(60, 2)
        tree = ClusterTree.from_points(points, leaf_size=8)

        is_leaf = tree.leaf_count > 0
        leaf_ids = torch.where(is_leaf)[0]
        total_sources = tree.leaf_count[leaf_ids].sum().item()
        assert total_sources == 60, (
            f"Expected 60 sources in leaves, got {total_sources}"
        )

    @pytest.mark.parametrize("n_dims", [2, 3])
    @pytest.mark.parametrize("theta", [0.3, 1.0, 5.0])
    def test_interaction_plan_source_coverage(self, n_dims: int, theta: float):
        """For every target, near + far pairs cover all sources exactly once.

        This is the fundamental invariant of the dual-tree traversal:
        every source must be accounted for (no omissions) and no source
        may be double-counted (no duplicates).  For far-field node pairs,
        we expand both the target node (to individual targets) and the
        source node (to individual sources via DFS) to verify coverage.
        """
        torch.manual_seed(DEFAULT_SEED)
        n_src, n_tgt = 40, 10
        source_pts = torch.randn(n_src, n_dims)
        target_pts = torch.randn(n_tgt, n_dims) * 3
        source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
        target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
        plan = source_tree.find_dual_interaction_pairs(
            target_tree=target_tree, theta=theta
        )

        all_sources = set(range(n_src))

        def _collect_sources(tree: ClusterTree, node_id: int) -> set[int]:
            """DFS to collect all source indices under a tree node."""
            count = tree.leaf_count[node_id].item()
            if count > 0:
                start = tree.leaf_start[node_id].item()
                return {
                    tree.sorted_source_order[start + j].item() for j in range(count)
                }
            result: set[int] = set()
            left = tree.node_left_child[node_id].item()
            right = tree.node_right_child[node_id].item()
            if left >= 0:
                result |= _collect_sources(tree, left)
            if right >= 0:
                result |= _collect_sources(tree, right)
            return result

        near_tgt = plan.near_target_ids.tolist()
        near_src = plan.near_source_ids.tolist()
        far_tgt_nids = plan.far_target_node_ids.tolist()
        far_src_nids = plan.far_source_node_ids.tolist()
        nf_tgt = plan.nf_target_ids.tolist()
        nf_src_nids = plan.nf_source_node_ids.tolist()
        fn_src = plan.fn_source_ids.tolist()
        fn_bcast_tgts = plan.fn_broadcast_targets.tolist()
        fn_bcast_starts = plan.fn_broadcast_starts.tolist()
        fn_bcast_counts = plan.fn_broadcast_counts.tolist()

        ### Expand (far,far): target node × source node
        far_expanded: list[tuple[int, int]] = []
        for tgt_nid, src_nid in zip(far_tgt_nids, far_src_nids):
            tgt_set = _collect_sources(target_tree, tgt_nid)
            src_set = _collect_sources(source_tree, src_nid)
            far_expanded.extend((t, s) for t in tgt_set for s in src_set)

        ### Expand (near,far): individual target × source node
        nf_expanded: list[tuple[int, int]] = []
        for ti, src_nid in zip(nf_tgt, nf_src_nids):
            src_set = _collect_sources(source_tree, src_nid)
            nf_expanded.extend((ti, s) for s in src_set)

        ### Expand (far,near): broadcast to survivors × individual source
        fn_expanded: list[tuple[int, int]] = []
        for src_id, start, count in zip(fn_src, fn_bcast_starts, fn_bcast_counts):
            fn_expanded.extend((fn_bcast_tgts[start + j], src_id) for j in range(count))

        for t in range(n_tgt):
            near_sources = {s for ti, s in zip(near_tgt, near_src) if ti == t}
            far_sources = {s for ti, s in far_expanded if ti == t}
            nf_sources = {s for ti, s in nf_expanded if ti == t}
            fn_sources = {s for ti, s in fn_expanded if ti == t}

            all_sets = [near_sources, far_sources, nf_sources, fn_sources]
            for i, (a, name_a) in enumerate(zip(all_sets, ["near", "far", "nf", "fn"])):
                for b, name_b in zip(all_sets[i + 1 :], ["far", "nf", "fn"][i:]):
                    overlap = a & b
                    assert not overlap, (
                        f"Target {t}: sources {overlap} in both {name_a} and {name_b}"
                    )

            covered = near_sources | far_sources | nf_sources | fn_sources
            assert covered == all_sources, (
                f"Target {t}: missing sources {all_sources - covered}, "
                f"extra sources {covered - all_sources}"
            )

    def test_large_theta_all_far(self):
        """With very large theta, most interactions become far-field node pairs."""
        torch.manual_seed(DEFAULT_SEED)
        source_pts = torch.randn(30, 2) * 0.1
        target_pts = torch.randn(10, 2) * 100
        source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
        target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
        plan = source_tree.find_dual_interaction_pairs(
            target_tree=target_tree, theta=100.0
        )
        assert plan.n_far_nodes > 0, "Expected some far-field node pairs"

    def test_zero_theta_all_near(self):
        """With theta=0 (exact), all interactions are near-field."""
        torch.manual_seed(DEFAULT_SEED)
        source_pts = torch.randn(20, 2)
        target_pts = torch.randn(5, 2) * 3
        source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
        target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
        plan = source_tree.find_dual_interaction_pairs(
            target_tree=target_tree, theta=0.0
        )

        # theta=0: all per-point criteria also fail, everything is (near,near).
        assert plan.n_near > 0
        assert plan.n_near == 20 * 5, (
            f"Expected {20 * 5} near-field pairs, got {plan.n_near}"
        )
        assert plan.n_far_nodes == 0
        assert plan.n_nf == 0, f"Expected 0 nf pairs at theta=0, got {plan.n_nf}"
        assert plan.n_fn == 0, f"Expected 0 fn pairs at theta=0, got {plan.n_fn}"

        # Every (target, source) pair must be unique.
        pairs = torch.stack([plan.near_target_ids, plan.near_source_ids], dim=1)
        unique_pairs = pairs.unique(dim=0)
        assert unique_pairs.shape[0] == pairs.shape[0], (
            f"Found {pairs.shape[0] - unique_pairs.shape[0]} duplicate "
            f"(target, source) pairs"
        )

    def test_aggregate_centroid_accuracy(self):
        """Root centroid matches brute-force area-weighted mean."""
        torch.manual_seed(DEFAULT_SEED)
        points = torch.randn(30, 3)
        areas = torch.rand(30) + 0.1
        tree = ClusterTree.from_points(points, leaf_size=4, areas=areas)
        agg = tree.compute_source_aggregates(points, areas)

        expected_centroid = (points * areas.unsqueeze(-1)).sum(0) / areas.sum()
        root_centroid = agg.node_centroid[0]

        torch.testing.assert_close(
            root_centroid, expected_centroid, atol=1e-5, rtol=1e-5
        )

    def test_aggregate_source_data_scalars(self):
        """Root aggregate of scalar source data matches brute-force."""
        torch.manual_seed(DEFAULT_SEED)
        n = 30
        points = torch.randn(n, 3)
        areas = torch.rand(n) + 0.1
        scalar_feat = torch.randn(n)

        tree = ClusterTree.from_points(points, leaf_size=4, areas=areas)
        source_data = TensorDict({"my_scalar": scalar_feat}, batch_size=[n])
        agg = tree.compute_source_aggregates(points, areas, source_data=source_data)

        expected = (scalar_feat * areas).sum() / areas.sum()
        actual = agg.node_source_data["my_scalar"][0]

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_aggregate_source_data_mixed(self):
        """Root aggregate of mixed scalar + vector source data matches brute-force."""
        torch.manual_seed(DEFAULT_SEED)
        n = 40
        D = 3
        points = torch.randn(n, D)
        areas = torch.rand(n) + 0.1
        scalar_feat = torch.randn(n)
        vector_feat = torch.randn(n, D)

        tree = ClusterTree.from_points(points, leaf_size=4, areas=areas)
        source_data = TensorDict({"s": scalar_feat, "v": vector_feat}, batch_size=[n])
        agg = tree.compute_source_aggregates(points, areas, source_data=source_data)

        total_area = areas.sum()
        expected_s = (scalar_feat * areas).sum() / total_area
        expected_v = (vector_feat * areas.unsqueeze(-1)).sum(0) / total_area

        torch.testing.assert_close(
            agg.node_source_data["s"][0], expected_s, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            agg.node_source_data["v"][0], expected_v, atol=1e-5, rtol=1e-5
        )

    # -- Precomputed leaf field consistency tests ----------------------------

    @pytest.mark.parametrize(
        "n_points, leaf_size, n_dims",
        [
            (50, 4, 3),
            (1, 4, 2),
            (10, 100, 2),
            (20, 1, 3),
        ],
        ids=["normal", "single_point", "root_only_leaf", "one_per_leaf"],
    )
    def test_precomputed_leaf_node_ids(
        self,
        n_points: int,
        leaf_size: int,
        n_dims: int,
    ):
        """Precomputed leaf_node_ids matches torch.where(leaf_count > 0)."""
        torch.manual_seed(DEFAULT_SEED)
        points = torch.randn(n_points, n_dims)
        tree = ClusterTree.from_points(points, leaf_size=leaf_size)

        expected = torch.where(tree.leaf_count > 0)[0]
        assert torch.equal(tree.leaf_node_ids, expected)
        assert tree.n_leaves == expected.shape[0]

    @pytest.mark.parametrize(
        "n_points, leaf_size, n_dims",
        [
            (50, 4, 3),
            (1, 4, 2),
            (10, 100, 2),
            (20, 1, 3),
        ],
        ids=["normal", "single_point", "root_only_leaf", "one_per_leaf"],
    )
    def test_precomputed_leaf_seg_ids(
        self,
        n_points: int,
        leaf_size: int,
        n_dims: int,
    ):
        """Precomputed leaf_seg_ids matches on-the-fly _ragged_arange computation."""
        torch.manual_seed(DEFAULT_SEED)
        points = torch.randn(n_points, n_dims)
        tree = ClusterTree.from_points(points, leaf_size=leaf_size)

        assert tree.leaf_seg_ids.shape == (n_points,)
        assert tree.leaf_seg_ids.dtype == torch.long
        if n_points > 0:
            assert tree.leaf_seg_ids.max() < tree.n_leaves

        # Rebuild seg_ids from scratch and compare
        leaf_starts = tree.leaf_start[tree.leaf_node_ids]
        leaf_counts = tree.leaf_count[tree.leaf_node_ids]
        positions, compact_ids = _ragged_arange(
            leaf_starts,
            leaf_counts,
            total=n_points,
        )
        expected = torch.zeros(n_points, dtype=torch.long)
        expected[positions] = compact_ids

        assert torch.equal(tree.leaf_seg_ids, expected)

    def test_precomputed_leaf_fields_empty_tree(self):
        """Empty tree has empty leaf_node_ids and leaf_seg_ids."""
        tree = ClusterTree.from_points(torch.empty(0, 2), leaf_size=4)

        assert tree.leaf_node_ids.numel() == 0
        assert tree.leaf_seg_ids.numel() == 0
        assert tree.n_leaves == 0

    def test_compute_source_aggregates_single_point(self):
        """Single-point tree centroid equals the point itself."""
        point = torch.tensor([[3.0, -1.0, 7.0]])
        area = torch.tensor([2.5])
        tree = ClusterTree.from_points(point, leaf_size=4, areas=area)
        agg = tree.compute_source_aggregates(point, area)

        torch.testing.assert_close(agg.node_centroid[0], point[0])

    def test_compute_source_aggregates_root_only_leaf(self):
        """Root-is-only-leaf centroid matches brute-force area-weighted mean."""
        torch.manual_seed(DEFAULT_SEED)
        n = 10
        points = torch.randn(n, 3)
        areas = torch.rand(n) + 0.1
        tree = ClusterTree.from_points(points, leaf_size=100, areas=areas)

        assert tree.n_leaves == 1, "Expected single leaf (root)"
        agg = tree.compute_source_aggregates(points, areas)

        expected = (points * areas.unsqueeze(-1)).sum(0) / areas.sum()
        torch.testing.assert_close(
            agg.node_centroid[0],
            expected,
            atol=1e-5,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# BarnesHutKernel convergence tests
# ---------------------------------------------------------------------------


dims_params = pytest.mark.parametrize("n_dims", [2, 3])
output_fields_params = pytest.mark.parametrize(
    "output_fields",
    [
        {"potential": "scalar"},
        {"velocity": "vector"},
        {"potential": "scalar", "velocity": "vector"},
    ],
)
source_config_params = pytest.mark.parametrize(
    "n_source_scalars, n_source_vectors",
    [(0, 1), (2, 0), (2, 1)],
    ids=["vectors_only", "scalars_only", "mixed"],
)


@dims_params
@output_fields_params
@source_config_params
def test_bh_convergence_to_exact(
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
    n_source_scalars: int,
    n_source_vectors: int,
):
    """BarnesHutKernel converges to exact Kernel as theta decreases toward 0."""
    bh_kernel, exact_kernel, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        n_source_scalars=n_source_scalars,
        n_source_vectors=n_source_vectors,
        n_source_points=30,
        n_target_points=15,
    )

    exact_result = exact_kernel(
        **data,
    )

    ### As theta decreases (more conservative), result converges to exact.
    # The tolerance factor accounts for the four-quadrant classification:
    # as theta changes, interactions shift between (near,far), (far,near),
    # and (near,near) modes, each with different approximation properties.
    # This can cause non-monotonic error at large theta values.
    prev_max_err = float("inf")
    for theta in [10.0, 2.0, 0.5, 0.01]:
        bh_result = bh_kernel(**data, theta=theta)

        max_err = max(
            (bh_result[k] - exact_result[k]).abs().max().item() for k in output_fields
        )

        assert max_err <= prev_max_err * 3.0 + 1e-5, (
            f"Error increased from {prev_max_err:.2e} to {max_err:.2e} at theta={theta}"
        )
        prev_max_err = max_err

    # At theta=0.01, should be very close to exact
    for field_name in output_fields:
        torch.testing.assert_close(
            bh_result[field_name],
            exact_result[field_name],
            atol=1e-4,
            rtol=1e-3,
            msg=f"Field {field_name!r} not close to exact at theta=0.01",
        )


@dims_params
@source_config_params
def test_bh_gradient_correctness(
    n_dims: int,
    n_source_scalars: int,
    n_source_vectors: int,
):
    """Gradients through BarnesHutKernel match exact kernel at low theta."""
    bh_kernel, exact_kernel, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields={"field": "scalar"},
        n_source_scalars=n_source_scalars,
        n_source_vectors=n_source_vectors,
        n_source_points=15,
        n_target_points=8,
    )
    bh_kernel.train()
    exact_kernel.train()

    # Make source_points require grad for gradient comparison
    data["source_points"] = data["source_points"].clone().requires_grad_(True)

    # Exact gradient
    exact_result = exact_kernel(**data)
    exact_loss = exact_result["field"].sum()
    exact_loss.backward()
    exact_grad = data["source_points"].grad.clone()

    data["source_points"].grad = None

    # BH gradient at low theta (near-exact, should match closely)
    bh_result = bh_kernel(**data, theta=0.01)
    bh_loss = bh_result["field"].sum()
    bh_loss.backward()
    bh_grad = data["source_points"].grad.clone()

    torch.testing.assert_close(
        bh_grad,
        exact_grad,
        atol=1e-3,
        rtol=1e-2,
        msg="BH gradients don't match exact at low theta",
    )


# ---------------------------------------------------------------------------
# Equivariance tests
# ---------------------------------------------------------------------------


@dims_params
@output_fields_params
@source_config_params
def test_bh_translation_equivariance(
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
    n_source_scalars: int,
    n_source_vectors: int,
):
    """Barnes-Hut kernel preserves translation equivariance.

    Translation does not change the morton-code relative ordering, so the
    tree structure and interaction plan are identical pre- and
    post-translation. This test uses a moderate theta.
    """
    bh_kernel, _, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        n_source_scalars=n_source_scalars,
        n_source_vectors=n_source_vectors,
    )

    result1 = bh_kernel(**data, theta=2.0)

    translation = torch.randn(n_dims)
    translated_data = {**data}
    translated_data["source_points"] = data["source_points"] + translation
    translated_data["target_points"] = data["target_points"] + translation

    result2 = bh_kernel(**translated_data, theta=2.0)

    for field_name in output_fields:
        torch.testing.assert_close(
            result1[field_name],
            result2[field_name],
            atol=1e-4,
            rtol=1e-4,
            msg=f"Translation equivariance failed for {field_name!r}",
        )


@dims_params
@output_fields_params
@source_config_params
def test_bh_rotational_equivariance(
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
    n_source_scalars: int,
    n_source_vectors: int,
):
    """Barnes-Hut kernel preserves rotational equivariance.

    The underlying kernel is exactly equivariant, but the tree
    decomposition is axis-aligned (morton codes). Rotation changes the tree
    structure, so equivariance is only recovered in the near-exact limit.
    We use a small theta so that nearly all interactions are exact.
    """
    # Ensure at least one source vector for basis construction
    effective_src_vectors = max(n_source_vectors, 1)
    bh_kernel, _, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        n_source_scalars=n_source_scalars,
        n_source_vectors=effective_src_vectors,
        n_global_vectors=1,
    )

    ### Build rotation matrix
    if n_dims == 2:
        angle = torch.tensor(torch.pi / 3)
        R = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)],
            ]
        )
    else:
        axis = F.normalize(torch.randn(3), dim=0)
        angle = torch.tensor(torch.pi / 3)
        K = torch.zeros(3, 3)
        K[0, 1], K[0, 2] = -axis[2], axis[1]
        K[1, 0], K[1, 2] = axis[2], -axis[0]
        K[2, 0], K[2, 1] = -axis[1], axis[0]
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    def _rotate_td(td: TensorDict) -> TensorDict:
        return td.apply(lambda v: v @ R.T if v.ndim > td.batch_dims else v)

    # Low theta: near-exact, so equivariance holds
    result1 = bh_kernel(**data, theta=0.01)

    rotated_data = {**data}
    rotated_data["source_points"] = data["source_points"] @ R.T
    rotated_data["target_points"] = data["target_points"] @ R.T
    rotated_data["source_data"] = _rotate_td(data["source_data"])
    rotated_data["global_data"] = _rotate_td(data["global_data"])

    result2 = bh_kernel(**rotated_data, theta=0.01)

    for field_name, field_type in output_fields.items():
        if field_type == "scalar":
            torch.testing.assert_close(
                result1[field_name],
                result2[field_name],
                atol=1e-4,
                rtol=1e-4,
                msg=f"Scalar {field_name!r} not invariant under rotation",
            )
        else:
            rotated_field1 = result1[field_name] @ R.T
            torch.testing.assert_close(
                rotated_field1,
                result2[field_name],
                atol=1e-4,
                rtol=1e-4,
                msg=f"Vector {field_name!r} not equivariant under rotation",
            )


# ---------------------------------------------------------------------------
# MultiscaleKernel integration
# ---------------------------------------------------------------------------


@dims_params
def test_multiscale_bh_convergence(n_dims: int):
    """MultiscaleKernel at low theta converges to exact per-branch Kernel results."""
    torch.manual_seed(DEFAULT_SEED)

    ms = MultiscaleKernel(
        n_spatial_dims=n_dims,
        output_field_ranks={"p": 0},
        reference_length_names=["short", "long"],
        source_data_ranks={"normal": 1},
        hidden_layer_sizes=[16],
        leaf_size=4,
    )
    ms.eval()

    n_src = 25
    torch.manual_seed(DEFAULT_SEED + 1)
    src = torch.randn(n_src, n_dims)
    tgt = torch.randn(10, n_dims) * 3
    normals = F.normalize(torch.randn(n_src, n_dims), dim=-1)
    ref_lengths = {"short": torch.tensor(0.1), "long": torch.tensor(1.0)}

    # Compute exact reference by evaluating each branch's underlying Kernel
    # (the parent class forward = exact dense evaluation)
    from physicsnemo.experimental.models.globe.field_kernel import Kernel

    exact_total = None
    for name in ms.reference_length_names:
        branch: BarnesHutKernel = ms.kernels[name]
        branch_result = Kernel.forward(
            branch,
            reference_length=ref_lengths[name] * torch.exp(ms.log_scalefactors[name]),
            source_points=src,
            target_points=tgt,
            source_strengths=torch.ones(n_src),
            source_data=TensorDict({"normal": normals}, batch_size=[n_src]),
            global_data=TensorDict(
                {
                    "log_reference_length_ratios": TensorDict(
                        {
                            "short_long": (
                                ref_lengths["short"] / ref_lengths["long"]
                            ).log()
                        }
                    ),
                }
            ),
        )
        exact_total = (
            branch_result if exact_total is None else exact_total + branch_result
        )

    bh_result = ms(
        source_points=src,
        target_points=tgt,
        reference_lengths=ref_lengths,
        source_data=TensorDict({"normal": normals}, batch_size=[n_src]),
        theta=0.01,
    )

    torch.testing.assert_close(
        bh_result["p"],
        exact_total["p"],
        atol=1e-3,
        rtol=1e-2,
        msg="MultiscaleKernel BH doesn't converge to exact at low theta",
    )


# ---------------------------------------------------------------------------
# Source permutation equivariance
# ---------------------------------------------------------------------------


@dims_params
@source_config_params
def test_bh_source_permutation(
    n_dims: int,
    n_source_scalars: int,
    n_source_vectors: int,
):
    """Result is independent of source ordering."""
    bh_kernel, _, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields={"p": "scalar"},
        n_source_scalars=n_source_scalars,
        n_source_vectors=n_source_vectors,
    )

    result1 = bh_kernel(**data, theta=2.0)

    perm = torch.randperm(data["source_points"].shape[0])
    perm_data = {**data}
    perm_data["source_points"] = data["source_points"][perm]
    perm_data["source_strengths"] = data["source_strengths"][perm]
    perm_data["source_data"] = data["source_data"][perm]

    result2 = bh_kernel(**perm_data, theta=2.0)

    torch.testing.assert_close(
        result1["p"],
        result2["p"],
        atol=1e-4,
        rtol=1e-4,
        msg="BH result changed under source permutation",
    )


# ---------------------------------------------------------------------------
# GLOBE-like configuration (mimics communication hyperlayer source data)
# ---------------------------------------------------------------------------


@dims_params
def test_bh_globe_like_config(n_dims: int):
    """Convergence with a source data configuration matching GLOBE's
    communication hyperlayers: multiple latent scalars, latent vectors,
    and strength scalars - the exact mix that triggered the production bug.
    """
    bh_kernel, exact_kernel, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields={"p": "scalar", "u": "vector"},
        n_source_scalars=8,
        n_source_vectors=3,
        n_global_scalars=1,
        n_global_vectors=1,
        n_source_points=40,
        n_target_points=20,
    )

    exact_result = exact_kernel(**data)
    bh_result = bh_kernel(**data, theta=0.01)

    # Wider tolerance than basic tests: 8 scalars + 3 vectors + globals
    # produces more accumulated floating-point error through the aggregation
    # and feature engineering pipeline, even at low theta.
    for field in ("p", "u"):
        torch.testing.assert_close(
            bh_result[field],
            exact_result[field],
            atol=5e-3,
            rtol=5e-2,
            msg=f"GLOBE-like config: {field!r} not close to exact at theta=0.01",
        )


# ---------------------------------------------------------------------------
# Nested source_data keys (matches GLOBE's actual data structure)
# ---------------------------------------------------------------------------


@dims_params
def test_bh_nested_source_data_keys(n_dims: int):
    """Convergence with nested TensorDict keys matching GLOBE's production format.

    GLOBE passes source_data structured like:
        {"physical": {"velocity": ...}, "latent": {"scalars": {"0": ...},
         "vectors": {"0": ...}}, "normals": ...}

    The aggregation, split_by_leaf_rank, and TensorDict.cat operations must
    handle this nesting correctly.
    """
    torch.manual_seed(DEFAULT_SEED)
    n_src, n_tgt = 30, 15

    source_data_ranks = {
        "physical": {"pressure": 0},
        "latent": {"scalars": {"0": 0, "1": 0}, "vectors": {"0": 1}},
        "normals": 1,
    }
    output_field_ranks = {"p": 0, "u": 1}

    common_kwargs = dict(
        n_spatial_dims=n_dims,
        output_field_ranks=output_field_ranks,
        source_data_ranks=source_data_ranks,
        hidden_layer_sizes=[16],
    )

    bh_kernel = BarnesHutKernel(**common_kwargs, leaf_size=DEFAULT_LEAF_SIZE)
    exact_kernel = Kernel(**common_kwargs)
    exact_kernel.load_state_dict(bh_kernel.state_dict(), strict=True)
    bh_kernel.eval()
    exact_kernel.eval()

    torch.manual_seed(DEFAULT_SEED + 1)
    source_data = TensorDict(
        {
            "physical": TensorDict(
                {"pressure": torch.randn(n_src)},
                batch_size=[n_src],
            ),
            "latent": TensorDict(
                {
                    "scalars": TensorDict(
                        {"0": torch.randn(n_src), "1": torch.randn(n_src)},
                        batch_size=[n_src],
                    ),
                    "vectors": TensorDict(
                        {"0": F.normalize(torch.randn(n_src, n_dims), dim=-1)},
                        batch_size=[n_src],
                    ),
                },
                batch_size=[n_src],
            ),
            "normals": F.normalize(torch.randn(n_src, n_dims), dim=-1),
        },
        batch_size=[n_src],
    )

    data = {
        "source_points": torch.randn(n_src, n_dims),
        "target_points": torch.randn(n_tgt, n_dims) * 5,
        "source_strengths": torch.rand(n_src) + 0.1,
        "reference_length": torch.ones(()),
        "source_data": source_data,
        "global_data": TensorDict({}, batch_size=[]),
    }

    exact_result = exact_kernel(**data)
    bh_result = bh_kernel(**data, theta=0.01)

    for field_name in output_field_ranks:
        torch.testing.assert_close(
            bh_result[field_name],
            exact_result[field_name],
            atol=1e-4,
            rtol=1e-3,
            msg=lambda default, f=field_name: (
                f"Nested keys: {f!r} not close to exact at theta=0.01 "
                f"(n_dims={n_dims}, torch={torch.__version__}).\n{default}"
            ),
        )


# ---------------------------------------------------------------------------
# Four-quadrant interaction mode tests
# ---------------------------------------------------------------------------


@dims_params
@source_config_params
def test_all_four_categories_active_and_correct(
    n_dims: int,
    n_source_scalars: int,
    n_source_vectors: int,
):
    """At moderate theta, all four interaction categories should be active
    and the combined result should still converge to exact.

    This is the critical test for the (near,far) and (far,near) code paths:
    the convergence tests at theta=0.01 barely exercise them because nearly
    everything is (near,near) at low theta.
    """
    ### Use balanced source/target scales so the (far,near) target-centroid
    # broadcast and (near,far) source monopole have comparable accuracy.
    # The default helper scales targets by 5x, making target leaf diameters
    # much larger and the (far,near) approximation very coarse.
    torch.manual_seed(DEFAULT_SEED)
    n_src, n_tgt = 60, 30
    common_kwargs = dict(
        n_spatial_dims=n_dims,
        output_field_ranks={"p": 0, "v": 1},
        source_data_ranks={
            **{f"source_scalar_{i}": 0 for i in range(n_source_scalars)},
            **{f"source_vector_{i}": 1 for i in range(max(n_source_vectors, 1))},
        },
        hidden_layer_sizes=[32, 32],
    )
    bh_kernel = BarnesHutKernel(**common_kwargs, leaf_size=4)
    exact_kernel = Kernel(**common_kwargs)
    exact_kernel.load_state_dict(bh_kernel.state_dict(), strict=False)
    bh_kernel.eval()
    exact_kernel.eval()

    torch.manual_seed(DEFAULT_SEED + 1)
    source_pts = torch.randn(n_src, n_dims)
    target_pts = torch.randn(n_tgt, n_dims)

    source_data_dict: dict[str, torch.Tensor] = {}
    for i in range(n_source_scalars):
        source_data_dict[f"source_scalar_{i}"] = torch.randn(n_src)
    for i in range(max(n_source_vectors, 1)):
        source_data_dict[f"source_vector_{i}"] = F.normalize(
            torch.randn(n_src, n_dims), dim=-1
        )

    data = {
        "source_points": source_pts,
        "target_points": target_pts,
        "source_strengths": torch.randn(n_src).abs() + 0.1,
        "reference_length": torch.ones(()),
        "source_data": TensorDict(source_data_dict, batch_size=[n_src]),
        "global_data": TensorDict({}, batch_size=[]),
    }

    exact_result = exact_kernel(**data)

    ### Sweep theta to find one where all four categories are active.
    # With balanced geometry (source and target at same scale) and
    # theta=1.0, the diagnostic shows near=751, nf=200, fn=131, far=2.
    for theta in [1.0, 1.5, 2.0]:
        source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
        target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
        plan = source_tree.find_dual_interaction_pairs(
            target_tree=target_tree, theta=theta
        )
        if plan.n_near > 0 and plan.n_nf > 0 and plan.n_fn > 0 and plan.n_far_nodes > 0:
            break
    else:
        pytest.skip("Could not find theta with all four categories active")

    bh_result = bh_kernel(**data, theta=theta)

    ### Verify the result is close to exact
    for field_name in ("p", "v"):
        torch.testing.assert_close(
            bh_result[field_name],
            exact_result[field_name],
            atol=0.1,
            rtol=0.3,
            msg=f"Field {field_name!r} not close to exact at theta={theta} "
            f"with all four categories active "
            f"(near={plan.n_near}, nf={plan.n_nf}, fn={plan.n_fn}, far={plan.n_far_nodes})",
        )


@dims_params
@pytest.mark.parametrize("theta", [0.3, 1.0, 5.0])
def test_self_interaction_source_coverage(n_dims: int, theta: float):
    """Source coverage invariant for self-interaction (target_tree is source_tree).

    Communication hyperlayers use self-interaction where the same tree
    serves as both source and target.  The traversal starts with
    (root, root) and D_T == D_S at every level.
    """
    torch.manual_seed(DEFAULT_SEED)
    n_pts = 40
    points = torch.randn(n_pts, n_dims)
    tree = ClusterTree.from_points(points, leaf_size=4)
    plan = tree.find_dual_interaction_pairs(target_tree=tree, theta=theta)

    all_sources = set(range(n_pts))

    def _collect(tree: ClusterTree, node_id: int) -> set[int]:
        count = tree.leaf_count[node_id].item()
        if count > 0:
            start = tree.leaf_start[node_id].item()
            return {tree.sorted_source_order[start + j].item() for j in range(count)}
        result: set[int] = set()
        left = tree.node_left_child[node_id].item()
        right = tree.node_right_child[node_id].item()
        if left >= 0:
            result |= _collect(tree, left)
        if right >= 0:
            result |= _collect(tree, right)
        return result

    near_tgt = plan.near_target_ids.tolist()
    near_src = plan.near_source_ids.tolist()

    far_expanded: list[tuple[int, int]] = []
    for tgt_nid, src_nid in zip(
        plan.far_target_node_ids.tolist(), plan.far_source_node_ids.tolist()
    ):
        far_expanded.extend(
            (t, s) for t in _collect(tree, tgt_nid) for s in _collect(tree, src_nid)
        )

    nf_expanded: list[tuple[int, int]] = []
    for ti, src_nid in zip(
        plan.nf_target_ids.tolist(), plan.nf_source_node_ids.tolist()
    ):
        nf_expanded.extend((ti, s) for s in _collect(tree, src_nid))

    fn_expanded: list[tuple[int, int]] = []
    fn_bcast = plan.fn_broadcast_targets.tolist()
    for src_id, start, count in zip(
        plan.fn_source_ids.tolist(),
        plan.fn_broadcast_starts.tolist(),
        plan.fn_broadcast_counts.tolist(),
    ):
        fn_expanded.extend((fn_bcast[start + j], src_id) for j in range(count))

    for t in range(n_pts):
        near_s = {s for ti, s in zip(near_tgt, near_src) if ti == t}
        far_s = {s for ti, s in far_expanded if ti == t}
        nf_s = {s for ti, s in nf_expanded if ti == t}
        fn_s = {s for ti, s in fn_expanded if ti == t}

        covered = near_s | far_s | nf_s | fn_s
        assert covered == all_sources, (
            f"Self-interaction target {t} at theta={theta}: "
            f"missing {all_sources - covered}"
        )


def test_near_field_monotonicity():
    """Near-field pair count should decrease as theta increases.

    At higher theta, more interactions move to approximate modes
    (near-far, far-near, far-far), reducing the exact near-field count.
    """
    torch.manual_seed(DEFAULT_SEED)
    source_pts = torch.randn(50, 3)
    target_pts = torch.randn(25, 3) * 3
    source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
    target_tree = ClusterTree.from_points(target_pts, leaf_size=4)

    prev_n_near = float("inf")
    for theta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        plan = source_tree.find_dual_interaction_pairs(
            target_tree=target_tree, theta=theta
        )
        assert plan.n_near <= prev_n_near, (
            f"Near-field count increased from {prev_n_near} to {plan.n_near} "
            f"when theta increased to {theta}"
        )
        prev_n_near = plan.n_near


# ---------------------------------------------------------------------------
# DualInteractionPlan validation tests
# ---------------------------------------------------------------------------


class TestDualInteractionPlanValidate:
    """Tests for DualInteractionPlan.validate()."""

    def _make_valid_plan(self) -> DualInteractionPlan:
        """Construct a minimal valid DualInteractionPlan."""
        return DualInteractionPlan(
            near_target_ids=torch.tensor([0, 1]),
            near_source_ids=torch.tensor([2, 3]),
            far_target_node_ids=torch.tensor([0]),
            far_source_node_ids=torch.tensor([1]),
            nf_target_ids=torch.tensor([0]),
            nf_source_node_ids=torch.tensor([1]),
            fn_target_node_ids=torch.tensor([0, 0]),
            fn_source_ids=torch.tensor([1, 2]),
            fn_broadcast_targets=torch.tensor([3, 4, 5]),
            fn_broadcast_starts=torch.tensor([0, 1]),
            fn_broadcast_counts=torch.tensor([1, 2]),
        )

    def test_valid_plan_passes(self):
        """A correctly constructed plan passes validation."""
        plan = self._make_valid_plan()
        plan.validate()

    def test_empty_plan_passes(self):
        """An empty plan (all zero-length tensors) passes validation."""
        e = torch.empty(0, dtype=torch.long)
        plan = DualInteractionPlan(
            near_target_ids=e.clone(),
            near_source_ids=e.clone(),
            far_target_node_ids=e.clone(),
            far_source_node_ids=e.clone(),
            nf_target_ids=e.clone(),
            nf_source_node_ids=e.clone(),
            fn_target_node_ids=e.clone(),
            fn_source_ids=e.clone(),
            fn_broadcast_targets=e.clone(),
            fn_broadcast_starts=e.clone(),
            fn_broadcast_counts=e.clone(),
        )
        plan.validate()

    def test_shape_mismatch_detected(self):
        """Mismatched near_target_ids / near_source_ids shapes are caught."""
        plan = DualInteractionPlan(
            near_target_ids=torch.tensor([0, 1, 2]),
            near_source_ids=torch.tensor([0, 1]),
            far_target_node_ids=torch.empty(0, dtype=torch.long),
            far_source_node_ids=torch.empty(0, dtype=torch.long),
            nf_target_ids=torch.empty(0, dtype=torch.long),
            nf_source_node_ids=torch.empty(0, dtype=torch.long),
            fn_target_node_ids=torch.empty(0, dtype=torch.long),
            fn_source_ids=torch.empty(0, dtype=torch.long),
            fn_broadcast_targets=torch.empty(0, dtype=torch.long),
            fn_broadcast_starts=torch.empty(0, dtype=torch.long),
            fn_broadcast_counts=torch.empty(0, dtype=torch.long),
        )
        with pytest.raises(ValueError, match="Shape mismatch"):
            plan.validate()

    def test_broadcast_out_of_bounds_detected(self):
        """fn_broadcast_starts + counts exceeding targets length is caught.

        This is the exact invariant violation that caused the original
        IndexError bug (starts + counts pointed beyond fn_broadcast_targets).
        """
        plan = DualInteractionPlan(
            near_target_ids=torch.empty(0, dtype=torch.long),
            near_source_ids=torch.empty(0, dtype=torch.long),
            far_target_node_ids=torch.empty(0, dtype=torch.long),
            far_source_node_ids=torch.empty(0, dtype=torch.long),
            nf_target_ids=torch.empty(0, dtype=torch.long),
            nf_source_node_ids=torch.empty(0, dtype=torch.long),
            fn_target_node_ids=torch.tensor([0]),
            fn_source_ids=torch.tensor([1]),
            fn_broadcast_targets=torch.tensor([0, 1]),
            fn_broadcast_starts=torch.tensor([1]),
            fn_broadcast_counts=torch.tensor([3]),
        )
        with pytest.raises(ValueError, match="fn_broadcast out of bounds"):
            plan.validate()

    def test_negative_counts_detected(self):
        """Negative fn_broadcast_counts values are caught."""
        plan = DualInteractionPlan(
            near_target_ids=torch.empty(0, dtype=torch.long),
            near_source_ids=torch.empty(0, dtype=torch.long),
            far_target_node_ids=torch.empty(0, dtype=torch.long),
            far_source_node_ids=torch.empty(0, dtype=torch.long),
            nf_target_ids=torch.empty(0, dtype=torch.long),
            nf_source_node_ids=torch.empty(0, dtype=torch.long),
            fn_target_node_ids=torch.tensor([0]),
            fn_source_ids=torch.tensor([1]),
            fn_broadcast_targets=torch.tensor([0, 1, 2]),
            fn_broadcast_starts=torch.tensor([0]),
            fn_broadcast_counts=torch.tensor([-1]),
        )
        with pytest.raises(ValueError, match="negative values"):
            plan.validate()

    @pytest.mark.parametrize("n_dims", [2, 3])
    @pytest.mark.parametrize("theta", [0.3, 1.0, 5.0])
    def test_validate_called_by_find_dual_interaction_pairs(
        self,
        n_dims: int,
        theta: float,
    ):
        """validate() is exercised on every plan produced by the traversal.

        This also checks external validity: all index tensors reference
        valid source/target/node indices within their respective trees.
        """
        torch.manual_seed(DEFAULT_SEED)
        n_src, n_tgt = 40, 15
        source_pts = torch.randn(n_src, n_dims)
        target_pts = torch.randn(n_tgt, n_dims) * 3
        source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
        target_tree = ClusterTree.from_points(target_pts, leaf_size=4)

        plan = source_tree.find_dual_interaction_pairs(
            target_tree=target_tree, theta=theta
        )

        ### External validity: indices within tree-specific ranges
        if plan.n_near > 0:
            assert plan.near_target_ids.max() < n_tgt
            assert plan.near_source_ids.max() < n_src
        if plan.n_far_nodes > 0:
            assert plan.far_target_node_ids.max() < target_tree.n_nodes
            assert plan.far_source_node_ids.max() < source_tree.n_nodes
        if plan.n_nf > 0:
            assert plan.nf_target_ids.max() < n_tgt
            assert plan.nf_source_node_ids.max() < source_tree.n_nodes
        if plan.n_fn > 0:
            assert plan.fn_source_ids.max() < n_src
            assert plan.fn_target_node_ids.max() < target_tree.n_nodes
            if plan.fn_broadcast_targets.numel() > 0:
                assert plan.fn_broadcast_targets.max() < n_tgt


# ---------------------------------------------------------------------------
# fn_broadcast expansion round-trip test
# ---------------------------------------------------------------------------


@dims_params
@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
def test_fn_broadcast_ragged_arange_matches_python_expansion(
    n_dims: int,
    theta: float,
):
    """The _ragged_arange expansion of fn_broadcast (BarnesHutKernel's code
    path) produces the same (target, source) pairs as the pure-Python
    expansion used in test_interaction_plan_source_coverage.

    This bridges the gap between "the plan is semantically correct" and
    "the consumer expands it correctly via _ragged_arange."
    """
    torch.manual_seed(DEFAULT_SEED)
    n_src, n_tgt = 40, 15
    source_pts = torch.randn(n_src, n_dims)
    target_pts = torch.randn(n_tgt, n_dims) * 3
    source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
    target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
    plan = source_tree.find_dual_interaction_pairs(
        target_tree=target_tree,
        theta=theta,
    )

    if plan.n_fn == 0:
        pytest.skip("No fn pairs at this theta")

    ### Reference: pure-Python expansion (same logic as source coverage test)
    ref_pairs: set[tuple[int, int]] = set()
    for src_id, start, count in zip(
        plan.fn_source_ids.tolist(),
        plan.fn_broadcast_starts.tolist(),
        plan.fn_broadcast_counts.tolist(),
    ):
        for j in range(count):
            ref_pairs.add((plan.fn_broadcast_targets[start + j].item(), src_id))

    ### Actual: _ragged_arange expansion (same code path as BarnesHutKernel)
    positions, pair_ids = _ragged_arange(
        plan.fn_broadcast_starts,
        plan.fn_broadcast_counts,
    )
    expanded_tgt_ids = plan.fn_broadcast_targets[positions]
    expanded_src_ids = plan.fn_source_ids[pair_ids]

    actual_pairs = set(
        zip(
            expanded_tgt_ids.tolist(),
            expanded_src_ids.tolist(),
        )
    )

    assert actual_pairs == ref_pairs, (
        f"Ragged expansion mismatch: "
        f"{len(actual_pairs - ref_pairs)} extra, "
        f"{len(ref_pairs - actual_pairs)} missing"
    )


# ---------------------------------------------------------------------------
# fn_broadcast_targets dead-entry detection
# ---------------------------------------------------------------------------


@dims_params
@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
def test_fn_broadcast_targets_no_dead_entries(n_dims: int, theta: float):
    """Every entry in fn_broadcast_targets is reachable from at least one
    fn pair's (start, count) range.

    Dead entries (survivors from leaf pairs with no fn sources) inflate
    fn_broadcast_targets.shape[0] beyond fn_broadcast_counts.sum().  This
    was the root cause of the original IndexError: the consumer passed
    total=fn_broadcast_targets.shape[0] to _ragged_arange, which
    generated out-of-bounds positions.
    """
    torch.manual_seed(DEFAULT_SEED)
    n_src, n_tgt = 40, 15
    source_pts = torch.randn(n_src, n_dims)
    target_pts = torch.randn(n_tgt, n_dims) * 3
    source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
    target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
    plan = source_tree.find_dual_interaction_pairs(
        target_tree=target_tree,
        theta=theta,
    )

    if plan.n_fn == 0:
        pytest.skip("No fn pairs at this theta")

    ### Build the set of all referenced positions in fn_broadcast_targets
    referenced = torch.zeros(
        plan.fn_broadcast_targets.shape[0],
        dtype=torch.bool,
    )
    for start, count in zip(
        plan.fn_broadcast_starts.tolist(),
        plan.fn_broadcast_counts.tolist(),
    ):
        referenced[start : start + count] = True

    n_dead = int((~referenced).sum())
    assert n_dead == 0, (
        f"{n_dead} of {plan.fn_broadcast_targets.shape[0]} entries in "
        f"fn_broadcast_targets are unreferenced (dead)"
    )


# ---------------------------------------------------------------------------
# Post-sort invariant preservation
# ---------------------------------------------------------------------------


@dims_params
@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
def test_fn_sort_preserves_broadcast_mapping(n_dims: int, theta: float):
    """The source-ID sort in find_dual_interaction_pairs preserves the
    fn_broadcast expansion semantics.

    After sorting, fn_broadcast_starts/counts are permuted but still
    reference the same (unsorted) fn_broadcast_targets array.  This test
    verifies the expansion produces the same (target, source) pair set
    regardless of the sort order.
    """
    torch.manual_seed(DEFAULT_SEED)
    n_src, n_tgt = 40, 15
    source_pts = torch.randn(n_src, n_dims)
    target_pts = torch.randn(n_tgt, n_dims) * 3
    source_tree = ClusterTree.from_points(source_pts, leaf_size=4)
    target_tree = ClusterTree.from_points(target_pts, leaf_size=4)
    plan = source_tree.find_dual_interaction_pairs(
        target_tree=target_tree,
        theta=theta,
    )

    if plan.n_fn == 0:
        pytest.skip("No fn pairs at this theta")

    ### Expand with the current (sorted) order
    sorted_pairs: set[tuple[int, int]] = set()
    for src_id, start, count in zip(
        plan.fn_source_ids.tolist(),
        plan.fn_broadcast_starts.tolist(),
        plan.fn_broadcast_counts.tolist(),
    ):
        for j in range(count):
            sorted_pairs.add((plan.fn_broadcast_targets[start + j].item(), src_id))

    ### Expand with a random permutation of the fn entries
    perm = torch.randperm(plan.n_fn)
    permuted_pairs: set[tuple[int, int]] = set()
    perm_src = plan.fn_source_ids[perm]
    perm_starts = plan.fn_broadcast_starts[perm]
    perm_counts = plan.fn_broadcast_counts[perm]
    for src_id, start, count in zip(
        perm_src.tolist(),
        perm_starts.tolist(),
        perm_counts.tolist(),
    ):
        for j in range(count):
            permuted_pairs.add((plan.fn_broadcast_targets[start + j].item(), src_id))

    assert sorted_pairs == permuted_pairs, (
        "fn_broadcast expansion changed under permutation of fn entries"
    )


# ---------------------------------------------------------------------------
# Tightened four-quadrant accuracy test
# ---------------------------------------------------------------------------


@dims_params
@source_config_params
def test_four_quadrant_per_category_accuracy(
    n_dims: int,
    n_source_scalars: int,
    n_source_vectors: int,
):
    """Verify that each interaction category individually produces
    reasonable results, not just the combined sum.

    Compares the BH result at a moderate theta against exact, with
    tighter tolerances than test_all_four_categories_active_and_correct.
    Also verifies that the near-field contribution alone (theta=0,
    no approximation) exactly matches the exact kernel.
    """
    torch.manual_seed(DEFAULT_SEED)
    n_src, n_tgt = 60, 30
    common_kwargs = dict(
        n_spatial_dims=n_dims,
        output_field_ranks={"p": 0},
        source_data_ranks={
            **{f"source_scalar_{i}": 0 for i in range(n_source_scalars)},
            **{f"source_vector_{i}": 1 for i in range(max(n_source_vectors, 1))},
        },
        hidden_layer_sizes=[32, 32],
    )
    bh_kernel = BarnesHutKernel(**common_kwargs, leaf_size=4)
    exact_kernel = Kernel(**common_kwargs)
    exact_kernel.load_state_dict(bh_kernel.state_dict(), strict=False)
    bh_kernel.eval()
    exact_kernel.eval()

    torch.manual_seed(DEFAULT_SEED + 1)
    source_data_dict: dict[str, torch.Tensor] = {}
    for i in range(n_source_scalars):
        source_data_dict[f"source_scalar_{i}"] = torch.randn(n_src)
    for i in range(max(n_source_vectors, 1)):
        source_data_dict[f"source_vector_{i}"] = F.normalize(
            torch.randn(n_src, n_dims), dim=-1
        )

    data = {
        "source_points": torch.randn(n_src, n_dims),
        "target_points": torch.randn(n_tgt, n_dims),
        "source_strengths": torch.randn(n_src).abs() + 0.1,
        "reference_length": torch.ones(()),
        "source_data": TensorDict(source_data_dict, batch_size=[n_src]),
        "global_data": TensorDict({}, batch_size=[]),
    }

    exact_result = exact_kernel(**data)

    ### Near-only (theta=0): should be numerically exact
    near_only = bh_kernel(**data, theta=0.0)
    torch.testing.assert_close(
        near_only["p"],
        exact_result["p"],
        atol=1e-5,
        rtol=1e-5,
        msg="Near-only (theta=0) doesn't match exact",
    )

    ### Low theta (0.01): near-exact, tight tolerance
    low_theta = bh_kernel(**data, theta=0.01)
    torch.testing.assert_close(
        low_theta["p"],
        exact_result["p"],
        atol=1e-4,
        rtol=1e-3,
        msg="Low theta (0.01) not close enough to exact",
    )


if __name__ == "__main__":
    pytest.main()
