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

"""Tests for tier-aligned empirical conformal diagnostics."""

import json

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    CoverageAccumulator,
    FunctionalBandCalibrator,
    QuantileRegressionScore,
)
from test.experimental.uq.conformal._helpers import (
    TIERS,
    count_syncs,
    fit,
    fitted_functional,
)


def _accumulator(tier="functional", **fit_kwargs):
    """Diagnostics for a 3-sample, alpha=0.5 predictor (fit kwargs override)."""
    kwargs = {"n_samples": 3, "alpha": 0.5, "shape": (3,), **fit_kwargs}
    return fit(tier, **kwargs)[0].coverage_accumulator()


def _td(**fields):
    return TensorDict(fields, batch_size=[])


def _interval_sample(n_points, n_covered, *, n_components=1, half_width=1.0):
    shape = (n_points,) if n_components == 1 else (n_points, n_components)
    lo = torch.full(shape, -half_width)
    hi = torch.full(shape, half_width)
    target = torch.zeros(shape)
    if n_covered < n_points:
        target[n_covered:, 0 if n_components > 1 else ...] = 2 * half_width
    return lo, hi, target


def test_functional_report_counts_whole_field_containment():
    accumulator = _accumulator("functional")
    accumulator.update(*_interval_sample(100, 99))
    accumulator.update(*_interval_sample(5, 5))
    report = accumulator.finalize()
    assert report["meta"] == {
        "tier": "functional",
        "alpha": 0.5,
        "n_cal": 3,
        "target_coverage": 0.5,
    }
    assert report["fields"]["tensor"] == {
        "n_samples": 2,
        "mean_interval_width": pytest.approx(2.0),
        "whole_field_coverage": pytest.approx(0.5),
    }


def test_cellwise_map_and_summary_are_elementwise():
    accumulator = _accumulator("cellwise")
    for covered in ([True, True, False], [True, False, False], [True, True, True]):
        target = torch.where(torch.tensor(covered), 0.0, 2.0)
        accumulator.update(-torch.ones(3), torch.ones(3), target)

    expected = torch.tensor([1.0, 2.0 / 3.0, 1.0 / 3.0], dtype=torch.float64)
    torch.testing.assert_close(accumulator.empirical_coverage_map(), expected)
    assert accumulator.finalize()["fields"]["tensor"] == {
        "n_samples": 3,
        "mean_interval_width": pytest.approx(2.0),
        "mean_element_coverage": pytest.approx(2.0 / 3.0),
        "minimum_element_coverage": pytest.approx(1.0 / 3.0),
        "fraction_at_target": pytest.approx(2.0 / 3.0),
    }


def test_cellwise_target_fraction_uses_exact_declared_alpha():
    accumulator = _accumulator("cellwise", shape=(2,), alpha=0.58)
    for sample in range(100):
        covered = torch.tensor([sample < 42, sample < 41])
        accumulator.update(
            -torch.ones(2), torch.ones(2), torch.where(covered, 0.0, 2.0)
        )

    report = accumulator.finalize()
    assert report["meta"]["target_coverage"] == pytest.approx(0.42)
    assert report["fields"]["tensor"]["fraction_at_target"] == 0.5


def test_mean_interval_width_uses_all_reported_elements():
    accumulator = _accumulator("functional")
    accumulator.update(*_interval_sample(1, 1, half_width=1.0))
    accumulator.update(*_interval_sample(3, 3, half_width=2.0))
    # (one width-2 element + three width-4 elements) / four elements.
    width = accumulator.finalize()["fields"]["tensor"]["mean_interval_width"]
    assert width == pytest.approx(3.5)


def test_empty_quantile_regression_set_has_zero_width():
    calibrator = FunctionalBandCalibrator(QuantileRegressionScore(), alpha=0.5)
    prediction = torch.zeros(1)
    target = torch.zeros(1)
    for _ in range(3):
        calibrator.update(
            prediction, target, aux={"lo": -torch.ones(1), "hi": torch.ones(1)}
        )

    predictor = calibrator.finalize()
    lo, hi = predictor.predict_interval(
        prediction, aux={"lo": torch.full((1,), -0.1), "hi": torch.full((1,), 0.1)}
    )
    assert bool((lo > hi).all())

    accumulator = predictor.coverage_accumulator()
    accumulator.update(lo, hi, target)
    width = accumulator.finalize()["fields"]["tensor"]["mean_interval_width"]
    assert width == 0.0


def test_tensordict_fields_report_per_field_and_select_fitted_keys():
    accumulator = _accumulator(fields=["p", "v"])
    lo = _td(p=-torch.ones(3), v=-torch.ones(3))
    hi = _td(p=torch.ones(3), v=torch.ones(3))
    accumulator.update(lo, hi, _td(p=torch.zeros(3), v=torch.full((3,), 2.0)))
    # A superset container is auto-selected to the fitted fields (mirroring
    # predict_interval), so the model's natural full output works directly.
    accumulator.update(lo, hi, _td(p=torch.zeros(3), v=torch.zeros(3), w=torch.ones(3)))
    report = accumulator.finalize()
    assert set(report["fields"]) == {"p", "v"}
    assert report["fields"]["p"]["whole_field_coverage"] == 1.0
    assert report["fields"]["v"]["whole_field_coverage"] == 0.5
    assert report["fields"]["p"]["n_samples"] == report["fields"]["v"]["n_samples"] == 2
    json.dumps(report, allow_nan=False)


EMPTY_ENTRIES = {
    "cellwise": {
        "mean_element_coverage": None,
        "minimum_element_coverage": None,
        "fraction_at_target": None,
    },
    "functional": {"whole_field_coverage": None},
}


@pytest.mark.parametrize("tier", TIERS)
def test_empty_report_has_exact_none_schema(tier):
    report = _accumulator(tier).finalize()
    assert report["fields"]["tensor"] == {
        "n_samples": 0,
        "mean_interval_width": None,
        **EMPTY_ENTRIES[tier],
    }
    json.dumps(report, allow_nan=False)


_ONES4, _ONES3 = torch.ones(4), torch.ones(3)
_LO4, _HI4, _T4 = (
    _td(a=-_ONES4, b=-_ONES4),
    _td(a=_ONES4, b=_ONES4),
    _td(a=0 * _ONES4, b=0 * _ONES4),
)
_LO3, _HI3, _T3 = (
    _td(a=-_ONES3, b=-_ONES3),
    _td(a=_ONES3, b=_ONES3),
    _td(a=0 * _ONES3, b=0 * _ONES3),
)
_PLAIN = (-_ONES3, _ONES3, 0 * _ONES3)
_F64 = torch.float64
_Z64, _BIG = torch.zeros(3, dtype=_F64), torch.tensor([1e308, 0.0, 0.0], dtype=_F64)
_WIDE = (_td(a=_Z64, b=_Z64), _td(a=_Z64 + 1, b=_BIG), _td(a=_Z64, b=_Z64))
_WIDER = (_WIDE[0], _td(a=_Z64 + 2, b=_BIG), _WIDE[2])
# fmt: off
TRANSACTIONAL_REJECTIONS = [  # (id, tier, fields, warm-up update, rejected update, error, match)
    ("hi-shape-mismatch", "functional", ["a", "b"], (_LO4, _HI4, _T4), (_LO4, _td(a=_ONES4, b=torch.ones(5)), _T4), ValueError, "shape"),
    ("nonfinite-target", "functional", ["a", "b"], (_LO4, _HI4, _T4), (_LO4, _HI4, _td(a=0 * _ONES4, b=torch.full((4,), torch.inf))), ValueError, "non-finite"),
    ("missing-fitted-field", "functional", ["a", "b"], (_LO4, _HI4, _T4), (_LO4, _HI4, _td(a=0 * _ONES4)), KeyError, "not present"),
    ("width-overflow", "functional", None, _PLAIN, tuple(torch.tensor([v], dtype=_F64) for v in (-1e308, 1e308, 0.0)), ValueError, "width overflows"),
    ("empty-target", "functional", None, _PLAIN, (torch.empty(0),) * 3, ValueError, "Plain tensor: empty"),
    ("container-mode-mismatch", "functional", None, _PLAIN, (_td(value=0 * _ONES3),) * 3, TypeError, "calibrated on a plain tensor; pass lo as a tensor"),
    ("cellwise-second-field-fails", "cellwise", ["a", "b"], (_LO3, _HI3, _T3), (_LO3, _HI3, _td(a=0 * _ONES3, b=0 * _ONES4)), ValueError, "shape"),
    ("width-sum-within-sample", "cellwise", ["a", "b"], (_LO3, _HI3, _T3), (_WIDE[0], _td(a=_Z64 + 2, b=_BIG.roll(1) + _BIG), _WIDE[2]), ValueError, "Field 'b'.*width sum.*[Rr]escale"),
    ("width-sum-cross-call", "cellwise", ["a", "b"], _WIDE, _WIDER, ValueError, "Field 'b'.*width sum.*[Rr]escale"),
    ("nonfinite-bounds", "cellwise", None, _PLAIN, (-_ONES3, torch.full((3,), torch.inf), 0 * _ONES3), ValueError, "non-finite"),
    ("cellwise-sample-shape-drift", "cellwise", None, _PLAIN, (-_ONES4, _ONES4, 0 * _ONES4), ValueError, "fixed sample shape"),
]
# fmt: on


@pytest.mark.parametrize(
    "tier,fields,warmup,rejected,error,match",
    [pytest.param(*row[1:], id=row[0]) for row in TRANSACTIONAL_REJECTIONS],
)
def test_update_rejections_are_transactional(
    tier, fields, warmup, rejected, error, match
):
    """A rejected update leaves scalar counters and per-element hit counts
    exactly as they were, even when the failure is on a later field."""
    accumulator = _accumulator(tier, fields=fields)
    accumulator.update(*warmup)
    before = accumulator.finalize()
    before_map = accumulator.empirical_coverage_map() if tier == "cellwise" else None
    with pytest.raises(error, match=match) as excinfo:
        accumulator.update(*rejected)
    assert "__tensor__" not in str(excinfo.value)
    after = accumulator.finalize()
    assert after == before
    json.dumps(after, allow_nan=False)
    if before_map is not None:
        after_map = accumulator.empirical_coverage_map()
        for key in fields or [None]:
            torch.testing.assert_close(
                after_map if key is None else after_map[key],
                before_map if key is None else before_map[key],
            )


def test_infinite_prediction_bounds_require_recomputing_in_wider_dtype():
    """Conservative fp16 bounds can overflow; upcast before predicting again."""
    predictor, points = fit("cellwise", n_samples=3, alpha=0.5, shape=(1,))
    prediction = torch.full((1,), 65504.0, dtype=torch.float16)
    lo, hi = predictor.predict_interval(prediction, points=points)
    assert torch.isfinite(prediction).all()
    assert lo.dtype == hi.dtype == torch.float16
    assert torch.isposinf(hi).all()
    assert ((lo <= prediction) & (prediction <= hi)).all()

    prediction = prediction.float()
    lo, hi = predictor.predict_interval(prediction, points=points)
    assert lo.dtype == hi.dtype == torch.float32
    assert torch.isfinite(lo).all() and torch.isfinite(hi).all()
    assert ((lo <= prediction) & (prediction <= hi)).all()
    accumulator = predictor.coverage_accumulator()
    accumulator.update(lo, hi, prediction)
    assert accumulator.finalize()["fields"]["tensor"]["n_samples"] == 1


def test_coverage_map_availability_errors():
    with pytest.raises(RuntimeError, match="No samples collected"):
        _accumulator("cellwise").empirical_coverage_map()
    with pytest.raises(RuntimeError, match="only for cellwise"):
        _accumulator("functional").empirical_coverage_map()


@pytest.mark.parametrize("tier", TIERS)
def test_update_syncs_once_and_accepts_a_second_device(monkeypatch, device, tier):
    """One host sync per update; samples on another device join the totals."""
    accumulator = _accumulator(tier, fields=["a", "b"])
    reference = _accumulator(tier, fields=["a", "b"])
    samples = [
        tuple(_td(a=t, b=2 * t) for t in _interval_sample(3, n_covered))
        for n_covered in (3, 1)
    ]
    calls = count_syncs(monkeypatch)
    accumulator.update(*(t.to(device) for t in samples[0]))
    assert calls == ["tolist"]
    accumulator.update(*samples[1])
    monkeypatch.undo()
    for sample in samples:
        reference.update(*sample)
    assert accumulator.finalize() == reference.finalize()


def test_accumulator_constructor_rejects_bare_string_keys():
    with pytest.raises(TypeError, match="not the string"):
        CoverageAccumulator(tier="functional", alpha=0.5, n_cal=3, keys="p")


def test_accumulator_update_rejects_unknown_tier():
    accumulator = CoverageAccumulator(tier="bogus", alpha=0.5, n_cal=3)
    with pytest.raises(ValueError, match="tier must be one of"):
        accumulator.update(torch.zeros(2), torch.ones(2), torch.zeros(2))


def test_report_metadata_is_private():
    accumulator = _accumulator()
    for name, value in (("tier", "cellwise"), ("alpha", 0.1), ("n_cal", 99)):
        setattr(accumulator, name, value)
    accumulator.keys = ("other",)
    assert accumulator.finalize()["meta"] == {
        "tier": "functional",
        "alpha": 0.5,
        "n_cal": 3,
        "target_coverage": 0.5,
    }


def _calibrator_finalizer():
    calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.5)
    for _ in range(3):
        calibrator.update(torch.zeros(2), torch.zeros(2))
    return calibrator.finalize


def _accumulator_finalizer():
    accumulator = fitted_functional().coverage_accumulator()
    accumulator.update(*_PLAIN)
    return accumulator.finalize


def _coverage_map_finalizer():
    accumulator = _accumulator("cellwise")
    accumulator.update(*_PLAIN)
    return accumulator.empirical_coverage_map


FINALIZERS = {
    "calibrator.finalize": _calibrator_finalizer,
    "accumulator.finalize": _accumulator_finalizer,
    "accumulator.empirical_coverage_map": _coverage_map_finalizer,
}


@pytest.mark.parametrize("finalizer", sorted(FINALIZERS))
def test_finalizers_run_per_process_in_multi_rank_job(finalizer, fake_multi_rank):
    """Rank 0 of a multi-GPU job can fit and report on samples it gathered."""
    finalize = FINALIZERS[finalizer]()
    fake_multi_rank()
    assert finalize() is not None
