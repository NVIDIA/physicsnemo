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
    ConformalPredictor,
    NormalizedErrorScore,
    PointwiseCalibrator,
)
from physicsnemo.experimental.uq.conformal._utils import TENSOR_KEY, points_fingerprint
from test.experimental.uq.conformal._helpers import (
    CALIBRATOR_CLASSES,
    CALIBRATORS,
    TIERS,
    fit,
)


def _td(**fields):
    return TensorDict(fields, batch_size=[])


def test_pointwise_uses_exact_conformal_rank_and_returns_one_predictor():
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.4)
    points = torch.arange(2.0).reshape(2, 1)
    for target in ([1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]):
        calibrator.update(torch.zeros(2), torch.tensor(target), points=points)

    predictor = calibrator.finalize()
    assert type(predictor) is ConformalPredictor
    assert predictor.tier == "pointwise"
    assert predictor.n_cal == 4
    # k = ceil(5 * 0.6) = 3: the third-smallest per cell.
    assert torch.equal(predictor.thresholds, torch.tensor([3.0, 3.0]))
    assert predictor._mesh_fingerprint == points_fingerprint(points)


def test_plain_tensor_errors_do_not_leak_internal_key():
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.2)
    with pytest.raises(ValueError, match="^Plain tensor: empty sample") as excinfo:
        calibrator.update(torch.empty(0), torch.empty(0), points=torch.zeros(1, 1))
    assert TENSOR_KEY not in str(excinfo.value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_pointwise_retains_contiguous_scores_without_changing_bits(dtype):
    """Flattening retained scores needs no corpus copy or dtype conversion."""
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
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


def test_pointwise_fingerprints_points_transactionally():
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
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
    with pytest.raises(TypeError, match="first calibration sample had points="):
        calibrator.update(torch.zeros(3), torch.zeros(3))
    assert calibrator.n_cal == 2


def test_pointwise_without_points_calibrates_scalar_observables(tmp_path, device):
    """Three scalar outputs (say drag, lift, moment) need no mesh."""
    generator = torch.Generator(device=device).manual_seed(0)
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.2)
    for _ in range(20):
        prediction = torch.randn(3, generator=generator, device=device)
        noise = torch.tensor([0.1, 1.0, 10.0], device=device)
        target = prediction + noise * torch.randn(3, generator=generator, device=device)
        calibrator.update(prediction, target)
    predictor = calibrator.finalize()
    lo, hi = predictor.predict_interval(torch.zeros(3, device=device))
    width = (hi - lo).cpu()
    assert bool((width[1:] > width[:-1]).all())  # one width per observable

    predictor.save(tmp_path / "scalars.pt")
    loaded = ConformalPredictor.load(tmp_path / "scalars.pt", map_location=device)
    torch.testing.assert_close(
        loaded.predict_interval(torch.zeros(3, device=device))[0], lo
    )

    with pytest.raises(TypeError, match="calibrated without points="):
        predictor.predict_interval(
            torch.zeros(3, device=device), points=torch.zeros(3, 1)
        )
    with pytest.raises(ValueError, match="shape"):
        predictor.predict_interval(torch.zeros(4, device=device))
    with pytest.raises(TypeError, match="did not have points="):
        calibrator.update(torch.zeros(3), torch.zeros(3), points=torch.zeros(3, 1))
    assert calibrator.n_cal == 20


def test_pointwise_rejects_output_layout_drift_on_the_same_mesh():
    points = torch.arange(4.0).reshape(4, 1)
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
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
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
    points = torch.arange(3.0).reshape(3, 1)
    if first is not None:
        calibrator.update(first, first.clone(), points=points)
    expected_n_cal = calibrator.n_cal
    with pytest.raises(error, match=match):
        calibrator.update(prediction, target, points=points)
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


def test_score_snapshot_is_fixed():
    """Calibrators and predictors hold snapshots of their score: neither
    caller-side mutation, accessor mutation, nor threshold mutation reaches
    the fitted rule, and semantic properties are read-only."""
    score = NormalizedErrorScore(eps=1.0)
    calibrator = PointwiseCalibrator(score, alpha=0.5)
    points = torch.arange(4.0).reshape(4, 1)
    target = torch.ones(4, dtype=torch.float64)
    zeros = torch.zeros_like(target)
    aux = {"sigma": zeros}
    calibrator.update(zeros, target, aux=aux, points=points)
    score.eps = 0.01  # the calibrator holds a snapshot, not the caller's object
    for _ in range(2):
        calibrator.update(zeros, target, aux=aux, points=points)
    predictor = calibrator.finalize()
    # sigma=0 clamps to eps=1: score |1 - 0| / 1 = 1.
    assert torch.equal(predictor.thresholds, torch.ones(4, dtype=torch.float64))
    assert predictor.score.eps == 1.0
    calibrator.score.eps = 123.0  # accessors are defensive copies
    assert calibrator.score.eps == 1.0

    lo, hi = predictor.predict_interval(zeros, aux=aux, points=points)
    torch.testing.assert_close(hi, torch.ones(4, dtype=torch.float64))
    predictor.thresholds.mul_(1e6)
    predictor.score.__dict__["eps"] = 123.0
    lo_again, hi_again = predictor.predict_interval(zeros, aux=aux, points=points)
    assert torch.equal(lo, lo_again) and torch.equal(hi, hi_again)
    for name, value in [
        ("tier", "pointwise"),
        ("alpha", 0.1),
        ("n_cal", 1),
        ("score", AbsoluteErrorScore()),
    ]:
        with pytest.raises(AttributeError, match="no setter"):
            setattr(predictor, name, value)


def test_keys_subset_round_trips_through_predict_save_and_diagnostics(tmp_path):
    generator = torch.Generator().manual_seed(7)
    calibrator = PointwiseCalibrator(
        AbsoluteErrorScore(), alpha=0.25, keys=["pressure"]
    )
    points = torch.arange(12.0).reshape(12, 1)
    full = _td(pressure=torch.zeros(12), velocity=torch.zeros(12, 3))
    for _ in range(8):
        target = _td(
            pressure=torch.randn(12, generator=generator),
            velocity=torch.randn(12, 3, generator=generator),
        )
        calibrator.update(full, target, points=points)
    predictor = calibrator.finalize()
    assert predictor.keys == ["pressure"]

    # The model's natural full output round-trips without a manual select.
    lo, hi = predictor.predict_interval(full, points=points)
    assert set(lo.keys()) == set(hi.keys()) == {"pressure"}

    path = tmp_path / "subset.pt"
    predictor.save(path)
    loaded = ConformalPredictor.load(path)
    assert loaded.keys == ["pressure"]
    lo_loaded, hi_loaded = loaded.predict_interval(full, points=points)
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


def test_unregistered_score_subclass_is_rejected():
    with pytest.raises(TypeError, match="a subclass needs its own registration"):
        PointwiseCalibrator(_CustomScore(), alpha=0.5)
    # Every shipped score remains constructible.
    PointwiseCalibrator(NormalizedErrorScore(), alpha=0.5)


def test_finalize_without_samples_fails_closed():
    with pytest.raises(RuntimeError, match="No calibration samples"):
        PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.5).finalize()


@pytest.mark.parametrize("tier", TIERS)
def test_finalize_alpha_reuses_the_collected_scores(tier):
    calibrator = CALIBRATORS[tier](AbsoluteErrorScore(), alpha=0.1)
    points = torch.arange(4.0).reshape(4, 1) if tier == "pointwise" else None
    generator = torch.Generator().manual_seed(3)
    for _ in range(20):
        target = torch.randn(4, 2, generator=generator)
        calibrator.update(torch.zeros(4, 2), target, points=points)
    strict, loose = calibrator.finalize(alpha=0.05), calibrator.finalize(alpha=0.25)
    assert (strict.alpha, loose.alpha) == (0.05, 0.25)
    assert bool((strict.thresholds >= loose.thresholds).all())
    torch.testing.assert_close(
        calibrator.finalize().thresholds,
        calibrator.finalize(alpha=0.1).thresholds,
        rtol=0,
        atol=0,
    )
    assert calibrator.alpha == 0.1 and calibrator.n_cal == 20


@pytest.mark.parametrize(
    "alpha,error,match",
    [
        (1.5, ValueError, r"in \(0, 1\)"),
        ("0.1", TypeError, "real number"),
        (True, TypeError, "real number"),
        (0.1, ValueError, "needs at least 9 calibration samples"),
    ],
)
def test_finalize_rejects_bad_alpha_and_stays_usable(alpha, error, match):
    calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.5)
    for _ in range(3):
        calibrator.update(torch.zeros(2), torch.ones(2))
    with pytest.raises(error, match=match):
        calibrator.finalize(alpha=alpha)
    assert calibrator.finalize().alpha == 0.5


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
