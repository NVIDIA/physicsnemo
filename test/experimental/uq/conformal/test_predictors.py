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

"""Tests for the single fitted-predictor value type."""

from collections import OrderedDict

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.experimental.uq.conformal as conformal
from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    AuxDifficulty,
    CellwiseCalibrator,
    ConformalPredictor,
)
from physicsnemo.experimental.uq.conformal._utils import points_fingerprint
from test.experimental.uq.conformal._helpers import count_syncs, fit, make_predictor


def test_nonfinite_deployment_difficulty_is_rejected():
    predictor, _ = fit(
        "risk_control",
        difficulty=AuxDifficulty(key="spread"),
        n_samples=10,
        shape=(20,),
        aux_factory=lambda p, g: {"spread": torch.rand(p.shape, generator=g) + 0.5},
    )
    spread = torch.ones(5)
    spread[2] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        predictor.predict_interval(torch.zeros(5), aux={"spread": spread})


def _sigma(prediction, generator):
    return {"sigma": torch.rand(prediction.shape, generator=generator) + 0.5}


@pytest.mark.parametrize(
    "tier,fields",
    [("cellwise", None), ("risk_control", None), ("risk_control", ["a", "b"])],
)
def test_predict_interval_syncs_once(monkeypatch, tier, fields):
    """All value checks, including the mesh checksum, share one host sync."""
    difficulty = None if tier == "cellwise" else AuxDifficulty()
    kwargs = dict(difficulty=difficulty, shape=(20,), fields=fields, n_samples=10)
    predictor, points = fit(tier, aux_factory=_sigma, **kwargs)
    aux = {"sigma": torch.rand(20) + 0.5}
    prediction = torch.randn(20)
    if fields is not None:
        prediction = TensorDict({key: prediction for key in fields}, batch_size=[])
        aux = {key: aux for key in fields}
    calls = count_syncs(monkeypatch)
    predictor.predict_interval(prediction, aux=aux, points=points)
    assert calls == ["tolist"]


def test_prediction_container_contract():
    """Superset containers select the fitted fields; missing fields, plain
    mappings, and containers handed to a tensor-mode predictor are named."""
    predictor, points = fit("cellwise", n_samples=10, shape=(4, 2), fields=["a", "b"])
    zeros = torch.zeros(4, 2)
    superset = TensorDict({"a": zeros, "b": zeros, "extra": zeros}, batch_size=[])
    lo, hi = predictor.predict_interval(superset, points=points)
    assert isinstance(lo, TensorDict) and isinstance(hi, TensorDict)
    assert set(lo.keys()) == set(hi.keys()) == {"a", "b"}
    with pytest.raises(KeyError, match="not present"):
        predictor.predict_interval(
            TensorDict({"a": zeros}, batch_size=[]), points=points
        )
    for mapping in (
        {"a": zeros, "b": zeros},
        OrderedDict([("b", zeros), ("a", zeros)]),
    ):
        with pytest.raises(TypeError, match="Tensor or TensorDict"):
            predictor.predict_interval(mapping, points=points)

    tensor_predictor, points = fit("cellwise", n_samples=10, shape=(4, 2))
    lo, hi = tensor_predictor.predict_interval(zeros, points=points)
    assert isinstance(lo, torch.Tensor) and isinstance(hi, torch.Tensor)
    with pytest.raises(TypeError, match="calibrated on a plain tensor"):
        tensor_predictor.predict_interval(
            TensorDict({"other": zeros}, batch_size=[]), points=points
        )


def test_exact_cellwise_mesh_identity_and_point_alignment():
    predictor, points = fit("cellwise", n_samples=10, shape=(4, 2))
    assert predictor._mesh_fingerprint == points_fingerprint(points)
    with pytest.raises(ValueError, match="requires points"):
        predictor.predict_interval(torch.zeros(4, 2))
    changed = points.clone()
    changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(torch.inf))
    with pytest.raises(ValueError, match="exact calibration mesh"):
        predictor.predict_interval(torch.zeros(4, 2), points=changed)
    with pytest.raises(ValueError, match="leading entry per point"):
        predictor.predict_interval(torch.zeros(3, 2), points=points)


@pytest.mark.parametrize("mutation", ["data", "numpy"])
def test_mesh_storage_mutations_are_rejected(mutation):
    points = torch.arange(4.0).reshape(4, 1)
    prediction = torch.zeros(4, 2)
    calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
    for _ in range(3):
        calibrator.update(prediction, prediction + 1, points=points)
    predictor = calibrator.finalize()
    predictor.predict_interval(prediction, points=points)
    version = points._version
    if mutation == "data":
        points.data[0, 0] += 1
    else:
        points.numpy()[0, 0] += 1
    assert points._version == version
    with pytest.raises(ValueError, match="same mesh"):
        calibrator.update(prediction, prediction + 1, points=points)
    assert calibrator.n_cal == 3
    with pytest.raises(ValueError, match="exact calibration mesh"):
        predictor.predict_interval(prediction, points=points)


class _CustomScore(AbsoluteErrorScore):
    pass


class _CustomDifficulty(AuxDifficulty):
    pass


_CELL = {"tier": "cellwise", "thresholds": torch.ones(2), "points": torch.ones(2, 1)}
# fmt: off
CONSTRUCTOR_REJECTIONS = [  # (id, make_predictor overrides, error, match)
    ("custom-score", {**_CELL, "score": _CustomScore()}, TypeError, "Subclasses are not supported"),
    ("cellwise-with-difficulty", {**_CELL, "difficulty": AuxDifficulty()}, ValueError, "does not take difficulty="),
    ("cellwise-without-mesh", {"tier": "cellwise", "thresholds": torch.ones(2)}, ValueError, "requires points="),
    ("cellwise-scalar-threshold", {**_CELL, "thresholds": torch.tensor(1.0)}, ValueError, "at least one dimension"),
    ("custom-difficulty", {"tier": "functional", "difficulty": _CustomDifficulty()}, TypeError, "Subclasses are not supported"),
    ("risk-with-points", {"points": torch.ones(2, 1)}, ValueError, "does not take points="),
    ("dict-thresholds", {"thresholds": {"pressure": torch.tensor(1.0)}}, TypeError, "Tensor or TensorDict"),
    ("empty-tensordict-thresholds", {"thresholds": TensorDict({})}, ValueError, "at least one"),
    ("integer-thresholds", {"thresholds": torch.ones((), dtype=torch.int32)}, TypeError, "floating"),
    ("empty-thresholds", {"thresholds": torch.empty(0)}, ValueError, "empty"),
    ("nonscalar-threshold", {"thresholds": torch.ones(3)}, ValueError, "must be scalars"),
    ("negative-threshold", {"thresholds": torch.tensor(-1.0)}, ValueError, "^Plain tensor: negative threshold"),  # no internal key
    ("unknown-tier", {"tier": "bogus"}, ValueError, "tier must be one of"),
    ("provenance-is-save-only", {"provenance": {}}, TypeError, "unexpected keyword"),
]
# fmt: on


@pytest.mark.parametrize(
    "overrides,error,match",
    [pytest.param(*row[1:], id=row[0]) for row in CONSTRUCTOR_REJECTIONS],
)
def test_constructor_tier_invariants_and_threshold_guards(overrides, error, match):
    with pytest.raises(error, match=match):
        make_predictor(**overrides)


def test_public_api_exports():
    expected = {
        "AbsoluteErrorScore",
        "AuxDifficulty",
        "CellwiseCalibrator",
        "ConformalPredictor",
        "CoverageAccumulator",
        "FunctionalBandCalibrator",
        "NormalizedErrorScore",
        "QuantileRegressionScore",
        "RiskControlCalibrator",
    }
    assert set(conformal.__all__) == expected
    assert all(hasattr(conformal, name) for name in expected)


def test_cellwise_predict_rejects_output_shape_drift_on_the_same_mesh():
    points = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    predictor = make_predictor(
        tier="cellwise",
        thresholds=torch.ones(6, 3),
        points=points,
    )
    with pytest.raises(ValueError, match="calibrated shape"):
        predictor.predict_interval(torch.zeros(6, 2), points=points)


def test_to_moves_thresholds_in_place_and_preserves_predictions_and_provenance(
    device, tmp_path
):
    fitted, points = fit("cellwise", n_samples=10, shape=(4, 2))
    fitted.save(tmp_path / "predictor.pt", provenance={"dataset": "drivaer"})
    predictor = ConformalPredictor.load(tmp_path / "predictor.pt")
    lo_ref, hi_ref = predictor.predict_interval(torch.zeros(4, 2), points=points)
    thresholds_ref = predictor.thresholds.clone()

    def state():
        p = predictor
        return (p.tier, p.alpha, p.n_cal, p._mesh_fingerprint, p.provenance)

    expected = ("cellwise", 0.2, 10, fitted._mesh_fingerprint, {"dataset": "drivaer"})
    assert state() == expected
    moved = predictor.to(device)
    assert moved is predictor  # nn.Module convention: in place, returns self
    assert predictor.thresholds.device.type == torch.device(device).type
    assert state() == expected
    torch.testing.assert_close(predictor.thresholds.cpu(), thresholds_ref)

    lo, hi = predictor.predict_interval(
        torch.zeros(4, 2, device=device), points=points.to(device)
    )
    torch.testing.assert_close(lo.cpu(), lo_ref)
    torch.testing.assert_close(hi.cpu(), hi_ref)


def test_aux_and_points_are_keyword_only():
    predictor, points = fit("cellwise", n_samples=10, shape=(4, 2))
    with pytest.raises(TypeError, match="positional"):
        predictor.predict_interval(torch.zeros(4, 2), points)
    with pytest.raises(TypeError, match="positional"):
        CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.2).update(
            torch.zeros(4, 2), torch.zeros(4, 2), points
        )
    score = predictor.score
    for method in (score.score, score.interval):
        with pytest.raises(TypeError, match="positional"):
            method(torch.zeros(2), torch.zeros(2), {})
