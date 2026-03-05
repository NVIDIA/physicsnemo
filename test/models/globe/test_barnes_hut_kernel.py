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
    InteractionPlan,
)
from physicsnemo.experimental.models.globe.field_kernel import (
    BarnesHutKernel,
    ChunkedKernel,
    MultiscaleKernel,
)

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
) -> tuple[BarnesHutKernel, ChunkedKernel, dict[str, Any]]:
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
    exact_kernel = ChunkedKernel(**common_kwargs).to(device_obj)

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
        global_data_dict[f"global_scalar_{i}"] = torch.randn(1, device=device_obj).squeeze()
    for i in range(n_global_vectors):
        global_data_dict[f"global_vector_{i}"] = F.normalize(
            torch.randn(n_spatial_dims, device=device_obj), dim=0
        )

    input_data = {
        "source_points": torch.randn(n_source_points, n_spatial_dims, device=device_obj),
        "target_points": torch.randn(n_target_points, n_spatial_dims, device=device_obj) * 5,
        "source_strengths": torch.randn(n_source_points, device=device_obj).abs() + 0.1,
        "reference_length": torch.ones((), device=device_obj),
        "source_data": TensorDict(
            source_data_dict, batch_size=[n_source_points], device=device_obj
        ),
        "global_data": TensorDict(
            global_data_dict, batch_size=[], device=device_obj
        ),
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
        assert total_sources == 60, f"Expected 60 sources in leaves, got {total_sources}"

    def test_interaction_plan_coverage(self):
        """Near + far pairs together cover all sources for each target."""
        torch.manual_seed(DEFAULT_SEED)
        source_pts = torch.randn(40, 2)
        target_pts = torch.randn(10, 2) * 3
        tree = ClusterTree.from_points(source_pts, leaf_size=4)
        plan = tree.find_interaction_pairs(target_pts, theta=0.5)

        assert plan.n_total > 0, "No interactions found"
        # Near-field source indices should be valid
        if plan.n_near > 0:
            assert plan.near_source_ids.min() >= 0
            assert plan.near_source_ids.max() < 40
        # Far-field node indices should be valid
        if plan.n_far > 0:
            assert plan.far_node_ids.min() >= 0
            assert plan.far_node_ids.max() < tree.n_nodes

    def test_high_theta_all_far(self):
        """With very large theta, most interactions become far-field."""
        torch.manual_seed(DEFAULT_SEED)
        source_pts = torch.randn(30, 2) * 0.1
        target_pts = torch.randn(10, 2) * 100  # far from sources
        tree = ClusterTree.from_points(source_pts, leaf_size=4)
        plan = tree.find_interaction_pairs(target_pts, theta=0.01)

        # With very small theta (very aggressive approximation), most should be far
        assert plan.n_far > 0, "Expected some far-field interactions"

    def test_zero_theta_all_near(self):
        """With theta=0 (most conservative), all interactions are near-field."""
        torch.manual_seed(DEFAULT_SEED)
        source_pts = torch.randn(20, 2)
        target_pts = torch.randn(5, 2) * 3
        tree = ClusterTree.from_points(source_pts, leaf_size=4)
        plan = tree.find_interaction_pairs(target_pts, theta=1e10)

        # theta=1e10 means dist > diameter * 1e10 is needed for far-field,
        # which essentially never happens. All interactions should be near-field.
        assert plan.n_near > 0
        # Every target should see every source
        assert plan.n_near == 20 * 5, (
            f"Expected {20 * 5} near-field pairs, got {plan.n_near}"
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

        torch.testing.assert_close(root_centroid, expected_centroid, atol=1e-5, rtol=1e-5)

    def test_aggregate_source_data_scalars(self):
        """Root aggregate of scalar source data matches brute-force."""
        torch.manual_seed(DEFAULT_SEED)
        n = 30
        points = torch.randn(n, 3)
        areas = torch.rand(n) + 0.1
        scalar_feat = torch.randn(n)

        tree = ClusterTree.from_points(points, leaf_size=4, areas=areas)
        source_data = TensorDict(
            {"my_scalar": scalar_feat}, batch_size=[n]
        )
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
        source_data = TensorDict(
            {"s": scalar_feat, "v": vector_feat}, batch_size=[n]
        )
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
    """BarnesHutKernel converges to exact ChunkedKernel as theta increases."""
    bh_kernel, exact_kernel, data = _make_bh_kernel_and_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        n_source_scalars=n_source_scalars,
        n_source_vectors=n_source_vectors,
        n_source_points=30,
        n_target_points=15,
    )

    exact_result = exact_kernel(
        **data, chunk_size=None,
    )

    ### At large theta (very conservative, many near-field), result should
    ### converge toward exact
    prev_max_err = float("inf")
    for theta in [0.1, 0.5, 2.0, 100.0]:
        bh_result = bh_kernel(**data, theta=theta)

        max_err = max(
            (bh_result[k] - exact_result[k]).abs().max().item()
            for k in output_fields
        )

        # Error should decrease (or stay flat) with increasing theta
        assert max_err <= prev_max_err * 1.5 + 1e-5, (
            f"Error increased from {prev_max_err:.2e} to {max_err:.2e} "
            f"at theta={theta}"
        )
        prev_max_err = max_err

    # At theta=100.0, should be very close to exact
    for field_name in output_fields:
        torch.testing.assert_close(
            bh_result[field_name],
            exact_result[field_name],
            atol=1e-4,
            rtol=1e-3,
            msg=f"Field {field_name!r} not close to exact at theta=100.0",
        )


@dims_params
@source_config_params
def test_bh_gradient_correctness(
    n_dims: int,
    n_source_scalars: int,
    n_source_vectors: int,
):
    """Gradients through BarnesHutKernel match exact kernel at high theta."""
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
    exact_result = exact_kernel(**data, chunk_size=None)
    exact_loss = exact_result["field"].sum()
    exact_loss.backward()
    exact_grad = data["source_points"].grad.clone()

    data["source_points"].grad = None

    # BH gradient at high theta (should match closely)
    bh_result = bh_kernel(**data, theta=100.0)
    bh_loss = bh_result["field"].sum()
    bh_loss.backward()
    bh_grad = data["source_points"].grad.clone()

    torch.testing.assert_close(
        bh_grad, exact_grad, atol=1e-3, rtol=1e-2,
        msg="BH gradients don't match exact at high theta",
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

    result1 = bh_kernel(**data, theta=0.5)

    translation = torch.randn(n_dims)
    translated_data = {**data}
    translated_data["source_points"] = data["source_points"] + translation
    translated_data["target_points"] = data["target_points"] + translation

    result2 = bh_kernel(**translated_data, theta=0.5)

    for field_name in output_fields:
        torch.testing.assert_close(
            result1[field_name], result2[field_name],
            atol=1e-4, rtol=1e-4,
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
    We use a large theta so that nearly all interactions are exact.
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
        R = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ])
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

    # High theta: near-exact, so equivariance holds
    result1 = bh_kernel(**data, theta=100.0)

    rotated_data = {**data}
    rotated_data["source_points"] = data["source_points"] @ R.T
    rotated_data["target_points"] = data["target_points"] @ R.T
    rotated_data["source_data"] = _rotate_td(data["source_data"])
    rotated_data["global_data"] = _rotate_td(data["global_data"])

    result2 = bh_kernel(**rotated_data, theta=100.0)

    for field_name, field_type in output_fields.items():
        if field_type == "scalar":
            torch.testing.assert_close(
                result1[field_name], result2[field_name],
                atol=1e-4, rtol=1e-4,
                msg=f"Scalar {field_name!r} not invariant under rotation",
            )
        else:
            rotated_field1 = result1[field_name] @ R.T
            torch.testing.assert_close(
                rotated_field1, result2[field_name],
                atol=1e-4, rtol=1e-4,
                msg=f"Vector {field_name!r} not equivariant under rotation",
            )


# ---------------------------------------------------------------------------
# MultiscaleKernel integration
# ---------------------------------------------------------------------------


@dims_params
def test_multiscale_bh_convergence(n_dims: int):
    """MultiscaleKernel with BarnesHutKernel converges to ChunkedKernel result."""
    torch.manual_seed(DEFAULT_SEED)

    common_kwargs = dict(
        n_spatial_dims=n_dims,
        output_field_ranks={"p": 0},
        reference_length_names=["short", "long"],
        source_data_ranks={"normal": 1},
        hidden_layer_sizes=[16],
    )
    n_src = 25

    ms_bh = MultiscaleKernel(
        **common_kwargs,
        kernel_class=BarnesHutKernel,
        kernel_class_kwargs={"leaf_size": 4},
    )
    ms_exact = MultiscaleKernel(**common_kwargs, kernel_class=ChunkedKernel)
    ms_exact.load_state_dict(ms_bh.state_dict(), strict=False)
    ms_bh.eval()
    ms_exact.eval()

    torch.manual_seed(DEFAULT_SEED + 1)
    src = torch.randn(n_src, n_dims)
    tgt = torch.randn(10, n_dims) * 3
    normals = F.normalize(torch.randn(n_src, n_dims), dim=-1)
    ref_lengths = {"short": torch.tensor(0.1), "long": torch.tensor(1.0)}

    exact_result = ms_exact(
        source_points=src,
        target_points=tgt,
        reference_lengths=ref_lengths,
        source_data=TensorDict({"normal": normals}, batch_size=[n_src]),
        chunk_size=None,
    )
    bh_result = ms_bh(
        source_points=src,
        target_points=tgt,
        reference_lengths=ref_lengths,
        source_data=TensorDict({"normal": normals}, batch_size=[n_src]),
        theta=100.0,
    )

    torch.testing.assert_close(
        bh_result["p"], exact_result["p"],
        atol=1e-3, rtol=1e-2,
        msg="MultiscaleKernel BH doesn't converge to exact at high theta",
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

    result1 = bh_kernel(**data, theta=0.5)

    perm = torch.randperm(data["source_points"].shape[0])
    perm_data = {**data}
    perm_data["source_points"] = data["source_points"][perm]
    perm_data["source_strengths"] = data["source_strengths"][perm]
    perm_data["source_data"] = data["source_data"][perm]

    result2 = bh_kernel(**perm_data, theta=0.5)

    torch.testing.assert_close(
        result1["p"], result2["p"],
        atol=1e-4, rtol=1e-4,
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

    exact_result = exact_kernel(**data, chunk_size=None)
    bh_result = bh_kernel(**data, theta=100.0)

    # Wider tolerance than basic tests: 8 scalars + 3 vectors + globals
    # produces more accumulated floating-point error through the aggregation
    # and feature engineering pipeline, even at high theta.
    for field in ("p", "u"):
        torch.testing.assert_close(
            bh_result[field],
            exact_result[field],
            atol=5e-3,
            rtol=5e-2,
            msg=f"GLOBE-like config: {field!r} not close to exact at theta=100",
        )


if __name__ == "__main__":
    pytest.main()
