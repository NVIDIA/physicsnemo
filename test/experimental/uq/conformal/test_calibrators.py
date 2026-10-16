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

"""Tests for the contracted conformal calibrators."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    AuxDifficulty,
    CellwiseCalibrator,
    ConformalPredictor,
    FunctionalBandCalibrator,
    NormalizedErrorScore,
    QuantileRegressionScore,
)
from physicsnemo.experimental.uq.conformal._utils import TENSOR_KEY, points_fingerprint
from test.experimental.uq.conformal._helpers import (
    CALIBRATOR_CLASSES,
    TIERS,
    fit,
    make_predictor,
)


def _td(**fields):
    return TensorDict(fields, batch_size=[])


def test_cellwise_uses_exact_conformal_rank_and_returns_one_predictor():
    calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.4)
    points = torch.arange(2.0).reshape(2, 1)
    for target in ([1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]):
        calibrator.update(torch.zeros(2), torch.tensor(target), points=points)

    predictor = calibrator.finalize()
    assert type(predictor) is ConformalPredictor
    assert predictor.tier == "cellwise"
    assert predictor.n_cal == 4
    # k = ceil(5 * 0.6) = 3: the third-smallest per cell.
    assert torch.equal(predictor.thresholds, torch.tensor([3.0, 3.0]))
    assert predictor._mesh_fingerprint == points_fingerprint(points)


def test_plain_tensor_errors_do_not_leak_internal_key():
    calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.2)
    with pytest.raises(ValueError, match="^Plain tensor: empty sample") as excinfo:
        calibrator.update(torch.empty(0), torch.empty(0))
    assert TENSOR_KEY not in str(excinfo.value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_cellwise_retains_contiguous_scores_without_changing_bits(dtype):
    """Flattening retained scores needs no corpus copy or dtype conversion."""
    calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
    points = torch.arange(5.0).reshape(5, 1)
    reference = []
    for offset in (0.1, 2.0, 1.0):
        target = (torch.arange(15, dtype=dtype).reshape(3, 5) + offset).T
        prediction = torch.zeros_like(target, requires_grad=True)
        assert not target.is_contiguous()
        reference.append(target.abs())
        calibrator.update(prediction, target, points=points)

    for stored in calibrator._scores[TENSOR_KEY]:
        assert (
            stored.untyped_storage().nbytes() == stored.numel() * stored.element_size()
        )
    expected_thresholds = torch.stack(reference).sort(dim=0).values[1]
    assert torch.equal(calibrator.finalize().thresholds, expected_thresholds)


def test_cellwise_requires_and_fingerprints_points_transactionally():
    calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
    with pytest.raises(TypeError, match="points"):
        calibrator.update(torch.zeros(3), torch.zeros(3))
    assert calibrator.n_cal == 0
    assert calibrator._mesh_fingerprint is None

    points = torch.arange(3.0).reshape(3, 1)
    with pytest.raises(ValueError, match="leading entry per point"):
        calibrator.update(torch.zeros(2), torch.zeros(2), points=points)
    assert calibrator.n_cal == 0
    assert calibrator._mesh_fingerprint is None

    calibrator.update(torch.zeros(3), torch.zeros(3), points=points)
    fingerprint = calibrator._mesh_fingerprint
    calibrator.update(torch.zeros(3), torch.ones(3), points=points.clone())
    assert calibrator.n_cal == 2
    assert calibrator._mesh_fingerprint == fingerprint

    for changed in (points.flip(0), points.to(torch.float64)):
        with pytest.raises(ValueError, match="same mesh"):
            calibrator.update(torch.zeros(3), torch.zeros(3), points=changed)
    assert calibrator.n_cal == 2


def test_cellwise_rejects_output_layout_drift_on_the_same_mesh():
    points = torch.arange(4.0).reshape(4, 1)
    calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
    calibrator.update(torch.zeros(4, 1), torch.zeros(4, 1), points=points)
    with pytest.raises(ValueError, match="score shape"):
        calibrator.update(torch.zeros(4, 2), torch.zeros(4, 2), points=points)
    assert calibrator.n_cal == 1


@pytest.mark.parametrize("tier", TIERS)
def test_tensordict_and_tensor_modes_have_identical_thresholds(tier):
    """A one-field container must fit bit-identical thresholds to the plain
    tensor path on the same data."""
    kwargs = {"n_samples": 8, "shape": (5, 2)}
    plain, _ = fit(tier, generator=torch.Generator().manual_seed(9), **kwargs)
    fields, _ = fit(
        tier, generator=torch.Generator().manual_seed(9), fields=["pressure"], **kwargs
    )
    assert isinstance(fields.thresholds, TensorDict)
    assert fields.keys == ["pressure"] and plain.keys is None
    assert torch.equal(fields.thresholds["pressure"], plain.thresholds)


_Z3 = torch.zeros(3)
_AB = _td(a=_Z3, b=_Z3)
_AC = _td(a=_Z3, c=_Z3)
# fmt: off
SCHEMA_REJECTIONS = [  # (id, first accepted sample or None, prediction, target, error, match)
    ("within-update-key-mismatch", None, _AB, _AC, KeyError, "field mismatch"),
    ("within-update-container-mixing", None, _Z3, _td(a=_Z3), TypeError, "both"),
    ("cross-update-schema-drift", _AB, _AC, _AC, KeyError, "Fields changed"),
    ("cross-update-container-mixing", _Z3, _td(field=_Z3), _td(field=_Z3), TypeError, "mix"),
    ("prediction-target-shape-mismatch", _Z3, _Z3, torch.zeros(3, 1), ValueError, "match exactly"),
]
# fmt: on


@pytest.mark.parametrize(
    "first,prediction,target,error,match",
    [pytest.param(*row[1:], id=row[0]) for row in SCHEMA_REJECTIONS],
)
def test_schema_drift_is_rejected_transactionally(
    first, prediction, target, error, match
):
    calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.5)
    if first is not None:
        calibrator.update(first, first.clone())
    expected_n_cal = calibrator.n_cal
    with pytest.raises(error, match=match):
        calibrator.update(prediction, target)
    assert calibrator.n_cal == expected_n_cal


@pytest.mark.parametrize("cls", CALIBRATOR_CLASSES)
def test_late_field_rejection_preserves_finalized_thresholds_transactionally(cls):
    """A failed second field must not append the first field's staged scores."""
    calibrator = cls(NormalizedErrorScore(), alpha=0.5)
    prediction = _td(a=torch.zeros(3), b=torch.zeros(3))
    aux = {key: {"sigma": torch.ones(3)} for key in ("a", "b")}
    kwargs = {"points": torch.arange(3.0).reshape(3, 1)}
    for value in (1.0, 2.0, 3.0, 4.0):
        target = _td(a=torch.full((3,), value), b=torch.full((3,), value + 1))
        calibrator.update(prediction, target, aux=aux, **kwargs)
    before = calibrator.finalize().thresholds

    with pytest.raises(ValueError, match="missing.*sigma"):
        calibrator.update(
            prediction, prediction, aux={"a": aux["a"], "b": {}}, **kwargs
        )

    assert calibrator.n_cal == 4
    after = calibrator.finalize().thresholds
    for key in ("a", "b"):
        assert torch.equal(after[key], before[key])


def test_functional_band_uses_scalar_rank_across_varying_meshes():
    calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.4)
    for index, maximum in enumerate([1.0, 4.0, 2.0, 3.0], start=2):
        target = torch.linspace(0.0, maximum, index)
        points = torch.arange(float(index)).reshape(index, 1)
        calibrator.update(torch.zeros_like(target), target, points=points)
    predictor = calibrator.finalize()
    assert predictor.tier == "functional"
    assert predictor.thresholds.ndim == 0
    assert float(predictor.thresholds) == 3.0  # k = ceil(5 * 0.6) = 3 of the sups

    query = torch.zeros(7)
    lo, hi = predictor.predict_interval(query, points=torch.arange(7.0).reshape(7, 1))
    assert lo.shape == query.shape == hi.shape


def test_difficulty_adaptation_and_strategy_snapshot_are_fixed():
    """Calibrators and predictors hold snapshots of their strategies: neither
    caller-side mutation, accessor mutation, nor threshold mutation reaches
    the fitted rule, and semantic properties are read-only."""
    score = NormalizedErrorScore(eps=1.0)
    shared = AuxDifficulty("scale")
    calibrator = FunctionalBandCalibrator(score, alpha=0.5, difficulty=shared)
    target = torch.ones(4, dtype=torch.float64)
    zeros = torch.zeros_like(target)
    aux = {"scale": torch.full((4,), 2.0, dtype=torch.float64), "sigma": zeros}
    bad_scale = torch.tensor([2.0, torch.inf, 2.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite"):
        calibrator.update(zeros, target, aux={**aux, "scale": bad_scale})
    assert calibrator.n_cal == 0

    calibrator.update(zeros, target, aux=aux)
    shared.key = "other"  # the calibrator holds a snapshot, not the caller's object
    score.eps = 0.01
    for _ in range(2):
        calibrator.update(zeros, target, aux=aux)
    predictor = calibrator.finalize()
    # sigma=0 clamps to eps=1: score |1 - 0| / 1 = 1, normalized by scale 2.
    assert float(predictor.thresholds) == 0.5
    assert predictor.difficulty.key == "scale"
    assert predictor.score.eps == 1.0
    calibrator.difficulty.key = "mutated"  # accessors are defensive copies
    assert calibrator.difficulty.key == "scale"

    lo, hi = predictor.predict_interval(zeros, aux=aux)
    torch.testing.assert_close(hi, torch.ones(4, dtype=torch.float64))
    predictor.thresholds.mul_(1e6)
    predictor.score.__dict__["eps"] = 123.0
    predictor.difficulty.key = "mutated"
    lo_again, hi_again = predictor.predict_interval(zeros, aux=aux)
    assert torch.equal(lo, lo_again) and torch.equal(hi, hi_again)
    for name, value in [
        ("tier", "cellwise"),
        ("alpha", 0.1),
        ("n_cal", 1),
        ("score", AbsoluteErrorScore()),
        ("difficulty", AuxDifficulty()),
    ]:
        with pytest.raises(AttributeError, match="no setter"):
            setattr(predictor, name, value)


@pytest.mark.parametrize("calibrator_cls", [FunctionalBandCalibrator])
def test_scaled_tiers_reject_points_that_contradict_the_point_axis(calibrator_cls):
    """A supplied coordinate tensor must align with the field's leading
    (point) axis: the statistic is computed over that axis, so silently
    accepting a contradictory points= would make it look load-bearing."""
    calibrator = calibrator_cls(AbsoluteErrorScore(), alpha=0.5)
    field = torch.zeros(2, 2, 1)
    with pytest.raises(ValueError, match="leading entry per point"):
        calibrator.update(field, field.clone(), points=torch.zeros(4, 3))
    assert calibrator.n_cal == 0

    for _ in range(3):
        calibrator.update(field, torch.ones(2, 2, 1), points=torch.zeros(2, 3))
    predictor = calibrator.finalize()
    with pytest.raises(ValueError, match="leading entry per point"):
        predictor.predict_interval(field, points=torch.zeros(4, 3))


# fmt: off
DOUBLE_SCALING_BOUNDARIES = [
    pytest.param(lambda s, d: FunctionalBandCalibrator(s, alpha=0.5, difficulty=d), id="functional"),
    pytest.param(lambda s, d: make_predictor(tier="functional", score=s, difficulty=d), id="direct_predictor"),
]
# fmt: on


@pytest.mark.parametrize("build", DOUBLE_SCALING_BOUNDARIES)
def test_double_scaling_pairing_is_rejected_everywhere(build):
    """NormalizedErrorScore already divides by sigma; an AuxDifficulty('sigma')
    would scale intervals by ~sigma**2. Every construction boundary must
    reject it, and must keep allowing additive-aux pairings."""
    with pytest.raises(ValueError, match="twice"):
        build(NormalizedErrorScore(), AuxDifficulty("sigma"))
    # Positive controls: reading a key is not dividing by it.
    build(AbsoluteErrorScore(), AuxDifficulty("spread"))
    build(QuantileRegressionScore(), AuxDifficulty("lo"))


def test_keys_subset_round_trips_through_predict_save_and_diagnostics(tmp_path):
    generator = torch.Generator().manual_seed(7)
    calibrator = FunctionalBandCalibrator(
        AbsoluteErrorScore(), alpha=0.25, keys=["pressure"]
    )
    full = _td(pressure=torch.zeros(12), velocity=torch.zeros(12, 3))
    for _ in range(8):
        target = _td(
            pressure=torch.randn(12, generator=generator),
            velocity=torch.randn(12, 3, generator=generator),
        )
        calibrator.update(full, target)
    predictor = calibrator.finalize()
    assert predictor.keys == ["pressure"]

    # The model's natural full output round-trips without a manual select.
    lo, hi = predictor.predict_interval(full)
    assert set(lo.keys()) == set(hi.keys()) == {"pressure"}

    path = tmp_path / "subset.pt"
    predictor.save(path)
    loaded = ConformalPredictor.load(path)
    assert loaded.keys == ["pressure"]
    lo_loaded, hi_loaded = loaded.predict_interval(full)
    torch.testing.assert_close(lo_loaded["pressure"], lo["pressure"])
    torch.testing.assert_close(hi_loaded["pressure"], hi["pressure"])

    # The held-out loop works with the natural full target too: the
    # accumulator selects the fitted fields itself.
    accumulator = loaded.coverage_accumulator()
    accumulator.update(lo_loaded, hi_loaded, full)
    report = accumulator.finalize()
    assert set(report["fields"]) == {"pressure"}
    assert report["fields"]["pressure"]["n_samples"] == 1


class _CustomScore(AbsoluteErrorScore):
    pass


class _CustomDifficulty(AuxDifficulty):
    pass


def test_only_shipped_exact_strategy_types_are_accepted():
    with pytest.raises(TypeError, match="Subclasses are not supported"):
        FunctionalBandCalibrator(_CustomScore(), alpha=0.5)
    with pytest.raises(TypeError, match="Subclasses are not supported"):
        FunctionalBandCalibrator(
            AbsoluteErrorScore(), alpha=0.5, difficulty=_CustomDifficulty()
        )
    # Every shipped score remains constructible.
    CellwiseCalibrator(NormalizedErrorScore(), alpha=0.5)


def test_finalize_without_samples_fails_closed():
    with pytest.raises(RuntimeError, match="No calibration samples"):
        FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.5).finalize()


@pytest.mark.parametrize("tier", TIERS)
def test_calibrate_finalize_predict_round_trip_on_device(tier, device):
    generator = torch.Generator(device=device).manual_seed(101)
    predictor, points = fit(
        tier, generator=generator, n_samples=5, shape=(6, 2), device=device
    )
    prediction = torch.zeros(6, 2, device=device)
    lo, hi = predictor.predict_interval(prediction, points=points)
    assert lo.device == prediction.device == hi.device
    assert lo.shape == prediction.shape == hi.shape
