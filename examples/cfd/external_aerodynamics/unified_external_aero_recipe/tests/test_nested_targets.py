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

"""Nested-field addressing through the recipe's target / field-name paths.

A dataset may store its fields in nested groups
(``interior.point_data.solution.gauge_pressure``). Every name-keyed path in
the recipe -- targets, loss, metrics, output normalization, nondim, collate,
and inference output naming -- must reach such a leaf via the
``"solution.gauge_pressure"`` spelling, with no flatten / unflatten step.
"""

from __future__ import annotations

import torch
from collate import build_collate_fn
from forward_kwargs import extract_targets
from loss import LossCalculator
from metrics import MetricCalculator
from nondim import NonDimensionalizeByMetadata, freestream_scales
from output_normalize import normalize_output_to_tensordict, split_concat_by_target
from tensordict import TensorDict
from utils import validate_field_coverage

from physicsnemo.mesh import DomainMesh, Mesh

TARGETS = {"solution.gauge_pressure": "scalar", "solution.wall_shear_stress": "vector"}


def _nested_domain(n: int = 8) -> DomainMesh:
    interior = Mesh(
        points=torch.randn(n, 3),
        point_data={
            "solution": {
                "gauge_pressure": torch.randn(n),
                "wall_shear_stress": torch.randn(n, 3),
            },
            "sdf": torch.randn(n),
        },
    )
    vehicle = Mesh(
        points=torch.randn(12, 3),
        cells=torch.randint(0, 12, (n, 3)),
        cell_data={"normals": torch.randn(n, 3)},
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
            "p_inf": torch.tensor(100.0),
            "rho_inf": torch.tensor(1.225),
        },
    )


class TestExtractTargets:
    def test_nested_targets_selected_with_nesting_kept(self):
        domain = _nested_domain()
        targets = extract_targets(domain, TARGETS)
        assert set(targets.keys(include_nested=True, leaves_only=True)) == {
            ("solution", "gauge_pressure"),
            ("solution", "wall_shear_stress"),
        }
        assert targets.batch_size == domain.interior.point_data.batch_size

    def test_missing_nested_target_lists_leaves(self):
        domain = _nested_domain()
        try:
            extract_targets(domain, {"solution.nope": "scalar"})
        except KeyError as e:
            assert "solution.gauge_pressure" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected KeyError")


class TestOutputNormalize:
    def test_tensor_output_split_matches_nested_targets(self):
        domain = _nested_domain()
        targets = extract_targets(domain, TARGETS).unsqueeze(0)
        pred = normalize_output_to_tensordict(torch.randn(1, 8, 4), TARGETS, "tensors")
        assert set(pred.keys(True, True)) == set(targets.keys(True, True))
        validate_field_coverage(TARGETS, pred, targets)

    def test_split_concat_scalar_squeezed_vector_kept(self):
        pred = split_concat_by_target(torch.randn(1, 8, 4), TARGETS)
        assert pred["solution", "gauge_pressure"].shape == (1, 8)
        assert pred["solution", "wall_shear_stress"].shape == (1, 8, 3)

    def test_mesh_output_select_nested(self):
        domain = _nested_domain()
        pred = normalize_output_to_tensordict(domain.interior, TARGETS, "mesh")
        assert ("solution", "wall_shear_stress") in pred


class TestLossAndMetrics:
    def test_loss_and_metrics_on_nested_tensordicts(self):
        domain = _nested_domain()
        target = extract_targets(domain, TARGETS).unsqueeze(0)
        pred = split_concat_by_target(torch.randn(1, 8, 4), TARGETS)

        total, loss_td = LossCalculator(TARGETS, loss_type="mse")(pred, target)
        assert total.ndim == 0
        ### Loss keys keep the config spelling and stay flat (train.py calls
        ### ``.item()`` on every entry).
        assert "loss/solution.gauge_pressure" in loss_td.keys()
        assert all(not isinstance(v, TensorDict) for v in loss_td.values())

        metrics = MetricCalculator(TARGETS, metrics=["l2"])(pred, target)
        assert set(metrics.keys()) == set(
            MetricCalculator(TARGETS, metrics=["l2"]).expected_keys()
        )
        assert "solution.wall_shear_stress_x_l2" in metrics.keys()

    def test_validate_field_coverage_reports_nested_missing(self):
        pred = TensorDict(
            {"solution": {"gauge_pressure": torch.zeros(2)}}, batch_size=[]
        )
        try:
            validate_field_coverage(TARGETS, pred, pred)
        except KeyError as e:
            assert "solution.wall_shear_stress" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected KeyError")


class TestNonDimensionalize:
    def _nondim(self):
        return NonDimensionalizeByMetadata(
            fields={
                "solution.gauge_pressure": "pressure",
                "solution.wall_shear_stress": "stress",
            },
            association="point_data",
        )

    def test_forward_on_domain_transforms_nested_leaf(self):
        domain = _nested_domain()
        out = self._nondim().apply_to_domain(domain)
        q_inf, p_inf, *_ = freestream_scales(domain.global_data)
        expected = (
            domain.interior.point_data["solution", "gauge_pressure"] - p_inf
        ) / q_inf
        assert torch.allclose(
            out.interior.point_data["solution", "gauge_pressure"], expected
        )
        assert torch.equal(
            out.interior.point_data["sdf"], domain.interior.point_data["sdf"]
        )

    def test_inverse_td_round_trip_nested(self):
        domain = _nested_domain()
        nondim = self._nondim()
        forward = nondim.apply_to_domain(domain).interior.point_data
        q_inf, p_inf, U_inf_mag, rho_inf, T_inf = freestream_scales(domain.global_data)
        back = nondim.inverse_td(
            forward,
            {
                "solution.gauge_pressure": "pressure",
                "solution.wall_shear_stress": "stress",
            },
            q_inf,
            p_inf,
            U_inf_mag,
            rho_inf=rho_inf,
            T_inf=T_inf,
        )
        for key in (("solution", "gauge_pressure"), ("solution", "wall_shear_stress")):
            assert torch.allclose(back[key], domain.interior.point_data[key], atol=1e-4)

    def test_inverse_td_does_not_match_by_leaf_name(self):
        ### A leaf named "p" inside an unrelated group must not be redimmed
        ### by stats declared for the top-level "p".
        nondim = NonDimensionalizeByMetadata(fields={"p": "stress"})
        td = TensorDict(
            {"p": torch.ones(2), "other": {"p": torch.ones(2)}}, batch_size=[]
        )
        out = nondim.inverse_td(
            td, {"p": "stress"}, torch.tensor(4.0), torch.tensor(0.0), torch.tensor(1.0)
        )
        assert torch.allclose(out["p"], torch.full((2,), 4.0))
        assert torch.equal(out["other", "p"], torch.ones(2))


class TestCollate:
    def test_tensors_mode_batches_nested_targets_and_group_kwargs(self):
        domain = _nested_domain()
        collate = build_collate_fn(
            "tensors",
            {"x": "interior.points", "sol": "interior.point_data.solution"},
            TARGETS,
        )
        batch = collate([(domain, {})])
        targets = batch["targets"]
        assert targets["solution", "gauge_pressure"].shape == (1, 8)
        assert targets["solution", "wall_shear_stress"].shape == (1, 8, 3)
        ### A forward kwarg resolving to a whole group is batch-wrapped leaf by
        ### leaf with the same token padding as a bare tensor kwarg: a (N,)
        ### leaf becomes (1, 1, N), a (N, C) leaf becomes (1, N, C).
        sol = batch["forward_kwargs"]["sol"]
        assert isinstance(sol, TensorDict)
        assert sol["gauge_pressure"].shape == (1, 1, 8)
        assert sol["wall_shear_stress"].shape == (1, 8, 3)
        assert batch["forward_kwargs"]["x"].shape == (1, 8, 3)
