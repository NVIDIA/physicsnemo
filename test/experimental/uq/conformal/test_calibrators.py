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
    AuxScale,
    ConformalPredictor,
    FunctionalBandCalibrator,
    NormalizedErrorScore,
    PointwiseCalibrator,
    QuantileRegressionScore,
    RiskControlCalibrator,
)
from physicsnemo.experimental.uq.conformal._utils import TENSOR_KEY, points_fingerprint
from test.experimental.uq.conformal._helpers import (
    CALIBRATOR_CLASSES,
    CALIBRATORS,
    TIERS,
    fit,
    make_predictor,
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
    calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.2)
    with pytest.raises(ValueError, match="^Plain tensor: empty sample") as excinfo:
        calibrator.update(torch.empty(0), torch.empty(0))
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
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
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


def test_scale_adaptation_and_strategy_snapshot_are_fixed():
    """Calibrators and predictors hold snapshots of their strategies: neither
    caller-side mutation, accessor mutation, nor threshold mutation reaches
    the fitted rule, and semantic properties are read-only."""
    score = NormalizedErrorScore(eps=1.0)
    shared = AuxScale("scale")
    calibrator = FunctionalBandCalibrator(score, alpha=0.5, scale=shared)
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
    assert predictor.scale.key == "scale"
    assert predictor.score.eps == 1.0
    calibrator.scale.key = "mutated"  # accessors are defensive copies
    assert calibrator.scale.key == "scale"

    lo, hi = predictor.predict_interval(zeros, aux=aux)
    torch.testing.assert_close(hi, torch.ones(4, dtype=torch.float64))
    predictor.thresholds.mul_(1e6)
    predictor.score.__dict__["eps"] = 123.0
    predictor.scale.key = "mutated"
    lo_again, hi_again = predictor.predict_interval(zeros, aux=aux)
    assert torch.equal(lo, lo_again) and torch.equal(hi, hi_again)
    for name, value in [
        ("tier", "risk_control"),
        ("alpha", 0.1),
        ("n_cal", 1),
        ("score", AbsoluteErrorScore()),
        ("scale", AuxScale()),
    ]:
        with pytest.raises(AttributeError, match="no setter"):
            setattr(predictor, name, value)


def test_crc_uses_point_events_over_all_trailing_components():
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    for score in (1.0, 2.0, 3.0):
        target = torch.tensor([[score, 0.0], [0.0, 0.0]])
        calibrator.update(torch.zeros_like(target), target)
    # Point reduction yields vectors [score, 0] and selects 1. Component
    # counting would select 0 for this construction.
    predictor = calibrator.finalize()
    assert predictor.tier == "risk_control"
    assert float(predictor.thresholds) == 1.0


def test_crc_exact_rational_feasibility_floor():
    below = RiskControlCalibrator(AbsoluteErrorScore(), alpha=1.0 / 3.0)
    for _ in range(2):
        below.update(torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="calibration samples"):
        below.finalize()

    exact = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.25)
    for _ in range(3):
        exact.update(torch.zeros(4), torch.zeros(4))
    assert float(exact.finalize().thresholds) == 0.0


def test_crc_accepts_varying_mesh_sizes_and_tensordict():
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.25)
    for n in (2, 7, 3, 11):
        calibrator.update(_td(v=torch.zeros(n, 3)), _td(v=torch.ones(n, 3)))
    predictor = calibrator.finalize()
    assert isinstance(predictor.thresholds, TensorDict)
    assert predictor.thresholds["v"].ndim == 0
    assert float(predictor.thresholds["v"]) == 1.0


@pytest.mark.parametrize("constant", [1.0, 3.5, 1e-3])
def test_crc_uniform_weights_reproduce_the_unweighted_threshold_bit_for_bit(
    constant, device
):
    """Equal weights of any size give the unweighted loss; weights may live on
    another device than the fields."""
    generator = torch.Generator(device=device).manual_seed(3)
    unweighted = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.2)
    weighted = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.2)
    for n_points in range(20, 35):
        prediction = torch.randn(n_points, 2, generator=generator, device=device)
        noise = torch.randn(n_points, 2, generator=generator, device=device)
        target = prediction + 0.3 * noise
        weights_device = device if n_points % 2 else "cpu"
        unweighted.update(prediction, target)
        weighted.update(
            prediction,
            target,
            weights=torch.full((n_points,), constant, device=weights_device),
        )
    assert torch.equal(weighted.finalize().thresholds, unweighted.finalize().thresholds)


def test_crc_weighted_threshold_is_conservative_at_the_exact_bound():
    """Weighted tail masses carry a rounding margin, so a total risk that sits
    exactly on the CRC bound passes unweighted but not weighted."""
    unweighted = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    weighted = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    for values in ([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0] * 4):
        target = torch.tensor(values)
        unweighted.update(torch.zeros(4), target)
        weighted.update(torch.zeros(4), target, weights=torch.ones(4))
    # At lambda=0 the exact risk is (1/2 + 1/2 + 0 + 1) / 4 = alpha.
    assert float(unweighted.finalize().thresholds) == 0.0
    assert float(weighted.finalize().thresholds) == 1.0


@pytest.mark.parametrize(
    "weights,expected",
    [([0.1, 0.1, 0.8], 3.0), ([0.8, 0.1, 0.1], 1.0), (None, 2.0)],
    ids=["heavy_tail", "heavy_head", "unweighted"],
)
def test_crc_weights_move_the_threshold_predictably(weights, expected):
    """Three samples with scores [1, 2, 3] at alpha=0.5 may total a risk of 1:
    a 0.8-weight point at score 3 forces lambda=3, at score 1 it allows
    lambda=1 (risk 3 * 0.2), and unweighted lands at 2 (risk 3 * 1/3)."""
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    prediction = _td(a=torch.zeros(3), b=torch.zeros(3, 2))
    target = _td(
        a=torch.tensor([1.0, 2.0, 3.0]),
        b=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
    )
    kwargs = {} if weights is None else {"weights": torch.tensor(weights)}
    for _ in range(3):
        calibrator.update(prediction, target, **kwargs)
    thresholds = calibrator.finalize().thresholds
    assert float(thresholds["a"]) == float(thresholds["b"]) == expected


def test_crc_weighted_records_hold_sorted_scores_and_tail_mass_on_the_cpu(device):
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    target = torch.tensor([3.0, 1.0, 2.0], device=device)
    weights = torch.tensor([1.0, 2.0, 7.0], device=device)
    calibrator.update(torch.zeros(3, device=device), target, weights=weights)
    scores, tail = calibrator._scores[TENSOR_KEY][0]
    assert scores.device.type == tail.device.type == "cpu"
    assert torch.equal(scores, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    # Weights in score order are [2, 7, 1] / 10; tail[j] sums ranks >= j.
    torch.testing.assert_close(tail, torch.tensor([1.0, 0.8, 0.1], dtype=torch.float64))


_W3 = torch.ones(3)
# fmt: off
WEIGHT_REJECTIONS = [  # (id, weights, error, match)
    ("2d", torch.ones(3, 1), ValueError, r"shape \(n_points,\)"),
    ("wrong-length", torch.ones(4), ValueError, r"\(prediction\).*weights has 4"),
    ("negative", torch.tensor([1.0, -1.0, 1.0]), ValueError, "nonnegative"),
    ("nan", torch.tensor([1.0, torch.nan, 1.0]), ValueError, "non-finite"),
    ("inf", torch.tensor([1.0, torch.inf, 1.0]), ValueError, "non-finite"),
    ("all-zero", torch.zeros(3), ValueError, "all zero"),
    ("integer", torch.ones(3, dtype=torch.int64), TypeError, "floating-point"),
    ("not-a-tensor", [1.0, 1.0, 1.0], TypeError, "torch.Tensor"),
]
# fmt: on


@pytest.mark.parametrize(
    "weights,error,match",
    [pytest.param(*row[1:], id=row[0]) for row in WEIGHT_REJECTIONS],
)
def test_crc_rejects_bad_weights_transactionally(weights, error, match):
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    calibrator.update(torch.zeros(3), torch.zeros(3), weights=_W3)
    with pytest.raises(error, match=match):
        calibrator.update(torch.zeros(3), torch.ones(3), weights=weights)
    assert calibrator.n_cal == 1


@pytest.mark.parametrize(
    "first_weighted", [True, False], ids=["weighted-first", "unweighted-first"]
)
def test_crc_weights_are_all_or_none_across_calls(first_weighted):
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    weighted, plain = {"weights": _W3}, {}
    first, second = (weighted, plain) if first_weighted else (plain, weighted)
    calibrator.update(torch.zeros(3), torch.zeros(3), **first)
    had = "had" if first_weighted else "did not have"
    with pytest.raises(TypeError, match=f"first calibration sample {had} weights="):
        calibrator.update(torch.zeros(3), torch.zeros(3), **second)
    assert calibrator.n_cal == 1


def test_crc_weights_must_align_with_every_tensordict_field():
    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5)
    prediction = _td(a=torch.zeros(3), b=torch.zeros(4, 2))
    with pytest.raises(ValueError, match=r"Field 'b' \(prediction\).*weights has 3"):
        calibrator.update(prediction, prediction.clone(), weights=_W3)
    assert calibrator.n_cal == 0


def test_functional_band_does_not_take_weights():
    calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.5)
    with pytest.raises(TypeError, match="weights"):
        calibrator.update(torch.zeros(3), torch.zeros(3), weights=_W3)
    assert calibrator.n_cal == 0


@pytest.mark.parametrize(
    "calibrator_cls", [FunctionalBandCalibrator, RiskControlCalibrator]
)
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
    pytest.param(lambda s, d: FunctionalBandCalibrator(s, alpha=0.5, scale=d), id="functional"),
    pytest.param(lambda s, d: RiskControlCalibrator(s, alpha=0.5, scale=d), id="risk_control"),
    pytest.param(lambda s, d: make_predictor(tier="functional", score=s, scale=d), id="direct_predictor"),
]
# fmt: on


@pytest.mark.parametrize("build", DOUBLE_SCALING_BOUNDARIES)
def test_double_scaling_pairing_is_rejected_everywhere(build):
    """NormalizedErrorScore already divides by sigma; an AuxScale('sigma')
    would scale intervals by ~sigma**2. Every construction boundary must
    reject it, and must keep allowing additive-aux pairings."""
    with pytest.raises(ValueError, match="twice"):
        build(NormalizedErrorScore(), AuxScale("sigma"))
    # Positive controls: reading a key is not dividing by it.
    build(AbsoluteErrorScore(), AuxScale("spread"))
    build(QuantileRegressionScore(), AuxScale("lo"))


def test_keys_subset_round_trips_through_predict_save_and_diagnostics(tmp_path):
    generator = torch.Generator().manual_seed(7)
    calibrator = RiskControlCalibrator(
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


class _CustomScale(AuxScale):
    pass


def test_unregistered_strategy_subclasses_are_rejected():
    with pytest.raises(TypeError, match="a subclass needs its own registration"):
        RiskControlCalibrator(_CustomScore(), alpha=0.5)
    with pytest.raises(TypeError, match="Subclasses are not supported"):
        RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5, scale=_CustomScale())
    # Every shipped score remains constructible.
    PointwiseCalibrator(NormalizedErrorScore(), alpha=0.5)


def test_finalize_without_samples_fails_closed():
    with pytest.raises(RuntimeError, match="No calibration samples"):
        RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.5).finalize()


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
