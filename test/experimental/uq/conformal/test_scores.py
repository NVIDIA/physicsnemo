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

"""Nonconformity-score and scale-field unit tests."""

import pytest
import torch

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    AuxScale,
    NormalizedErrorScore,
    QuantileRegressionScore,
)
from test.experimental.uq.conformal._helpers import (
    CALIBRATORS,
    TIERS,
    assert_admitted_covered,
    assert_predictor_covers_admitted,
)


def test_absolute_error_values(device):
    score = AbsoluteErrorScore()
    pred = torch.tensor([1.0, -2.0, 0.5], device=device)
    target = torch.tensor([1.5, -1.0, 0.5], device=device)
    torch.testing.assert_close(
        score.score(pred, target), torch.tensor([0.5, 1.0, 0.0], device=device)
    )
    lo, hi = score.interval(pred, torch.tensor(0.25, device=device))
    torch.testing.assert_close(lo, pred - 0.25)
    torch.testing.assert_close(hi, pred + 0.25)


def test_normalized_error_values_and_eps(device):
    score = NormalizedErrorScore(eps=1e-3)
    pred = torch.tensor([0.0, 0.0], device=device)
    target = torch.tensor([1.0, 2.0], device=device)
    aux = {"sigma": torch.tensor([0.5, 0.0], device=device)}
    # Second sigma clamps to eps.
    torch.testing.assert_close(
        score.score(pred, target, aux=aux), torch.tensor([2.0, 2000.0], device=device)
    )
    lo, hi = score.interval(pred, torch.tensor(1.0, device=device), aux=aux)
    torch.testing.assert_close(hi, torch.tensor([0.5, 1e-3], device=device))
    torch.testing.assert_close(lo, -hi)


def test_quantile_regression_asymmetry(device):
    score = QuantileRegressionScore()
    aux = {"lo": torch.zeros(3, device=device), "hi": torch.ones(3, device=device)}
    pred = torch.zeros(3, device=device)
    # Below lo, inside, above hi.
    target = torch.tensor([-0.5, 0.5, 1.75], device=device)
    torch.testing.assert_close(
        score.score(pred, target, aux=aux),
        torch.tensor([0.5, -0.5, 0.75], device=device),
    )
    lo, hi = score.interval(pred, torch.tensor(0.1, device=device), aux=aux)
    torch.testing.assert_close(lo, aux["lo"] - 0.1)
    torch.testing.assert_close(hi, aux["hi"] + 0.1)


@pytest.mark.parametrize(
    "score,aux",
    [
        (AbsoluteErrorScore(), None),
        (NormalizedErrorScore(), {"sigma": torch.full((5,), 0.7)}),
        (QuantileRegressionScore(), {"lo": -torch.ones(5), "hi": torch.ones(5)}),
    ],
)
def test_interval_endpoints_invert_score(score, aux, device):
    """At this scale, endpoint scores stay close to the threshold."""
    aux = None if aux is None else {k: v.to(device) for k, v in aux.items()}
    pred = torch.randn(5, device=device)
    threshold = torch.tensor(0.42, device=device)
    lo, hi = score.interval(pred, threshold, aux=aux)
    for endpoint in (lo, hi):
        torch.testing.assert_close(
            score.score(pred, endpoint, aux=aux),
            threshold.expand(5),
            atol=1e-6,
            rtol=1e-5,
        )


def test_aux_scale_channel_max_clamp_and_trailing_reduction():
    scale = AuxScale(key="sigma", eps=torch.tensor(1e-2))
    assert type(scale.eps) is float  # configuration is stored as primitives
    assert type(NormalizedErrorScore(torch.tensor(1e-3)).eps) is float
    sigma = torch.tensor([[0.5, 1.5], [0.0, 0.0]])
    torch.testing.assert_close(scale(aux={"sigma": sigma}), torch.tensor([1.5, 1e-2]))
    # A (points, time, channels) aux reduces to one scale per point.
    assert scale({"sigma": torch.rand(2, 3, 4) + 0.5}).shape == (2,)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64], ids=str)
def test_aux_scale_floors_finite_nonpositive_values(dtype):
    scale = AuxScale("spread")
    raw = torch.tensor([[[-2.0, -1.0]], [[-1.0, 0.0]], [[0.5, 1.5]]], dtype=dtype)
    floor = max(scale.eps, torch.finfo(dtype).tiny)
    torch.testing.assert_close(
        scale(aux={"spread": raw}),
        torch.tensor([floor, floor, 1.5], dtype=dtype),
        rtol=0,
        atol=0,
    )


def test_aux_scale_rejects_eps_unrepresentable_in_aux_dtype():
    sigma = {"sigma": torch.ones(3, dtype=torch.float16)}
    with pytest.raises(ValueError, match="torch.float16"):
        AuxScale("sigma", eps=1e6)(aux=sigma)
    scale = AuxScale("sigma")
    scale.eps = float("nan")
    with pytest.raises(ValueError, match="eps must be finite"):
        scale(aux=sigma)


# fmt: off
RAW_SCALE_REJECTIONS = [
    pytest.param(None, TypeError, "must be a torch.Tensor", id="none"),
    pytest.param(torch.ones(1, dtype=torch.int64), TypeError, "floating", id="int"),
    pytest.param(torch.tensor([float("-inf")]), ValueError, "non-finite", id="negative-inf-clamp"),
    pytest.param(torch.tensor([[float("-inf"), 1.0]]), ValueError, "non-finite", id="negative-inf-reduction"),
    pytest.param(torch.tensor([[float("nan"), 1.0]]), ValueError, "non-finite", id="nan"),
]
# fmt: on


@pytest.mark.parametrize("raw,error,match", RAW_SCALE_REJECTIONS)
def test_aux_scale_validates_raw_input(raw, error, match):
    with pytest.raises(error, match=match) as exc:
        AuxScale("spread")(aux={"spread": raw})
    assert "spread" in str(exc.value)


@pytest.mark.parametrize("tier", TIERS)
def test_unused_aux_can_be_omitted(tier):
    calibrator = CALIBRATORS[tier](AbsoluteErrorScore(), alpha=0.5)
    pred, target = torch.zeros(1), torch.ones(1)
    points = torch.zeros(1, 1) if tier == "pointwise" else None
    for aux in (None, {}, {"spread": None}):
        calibrator.update(pred, target, aux=aux, points=points)
    predictor = calibrator.finalize()
    assert predictor.scale is None
    for aux in (None, {}, {"spread": None}):
        lo, hi = predictor.predict_interval(pred, aux=aux, points=points)
        assert ((lo <= target) & (target <= hi)).all()


@pytest.mark.parametrize("tier", TIERS)
def test_normalized_low_precision_sigma_keeps_fitted_intervals_tight(tier, device):
    """Float16 sigma must not inflate a float64 unit radius to 15.65234375."""
    score = NormalizedErrorScore()
    calibrator = CALIBRATORS[tier](score, alpha=0.5)
    pred = torch.zeros(1, dtype=torch.float64, device=device)
    target = torch.ones_like(pred)
    aux = {"sigma": torch.full((1,), 60000.0, dtype=torch.float16, device=device)}
    points = torch.zeros(1, 1, device=device) if tier == "pointwise" else None
    for _ in range(3):
        calibrator.update(pred, target, aux=aux, points=points)
    predictor = calibrator.finalize()
    assert (score.score(pred, target, aux=aux) <= predictor.thresholds.to(device)).all()
    lo, hi = assert_predictor_covers_admitted(
        predictor, pred, target, aux=aux, points=points
    )
    torch.testing.assert_close(hi, target, atol=0, rtol=1e-12)
    torch.testing.assert_close(lo, -target, atol=0, rtol=1e-12)


@pytest.mark.parametrize(
    "pred_dtype,target_dtype,sigma_dtype",
    [
        (torch.float64, torch.float64, torch.float16),
        (torch.float32, torch.float32, torch.float16),
        (torch.float16, torch.float16, torch.float64),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float64, torch.float16, torch.float16),
        (torch.float16, torch.float64, torch.bfloat16),
    ],
    ids=[
        "double-half-sigma",
        "single-half-sigma",
        "half-double-sigma",
        "bf16-half",
        "double-half-target",
        "half-double-target",
    ],
)
def test_normalized_mixed_dtype_score_boundary_is_contained(
    pred_dtype, target_dtype, sigma_dtype, device
):
    score = NormalizedErrorScore(eps=1e-3)
    pred = torch.tensor(
        [0.0, 1000.0, -1000.0, 1.0, 0.0], dtype=pred_dtype, device=device
    )
    target = torch.tensor(
        [1.0, 1000.5, -999.5, -1.0, 0.0], dtype=target_dtype, device=device
    )
    sigma = torch.tensor(
        [60000.0, 0.0, -1.0, 0.3, 60000.0], dtype=sigma_dtype, device=device
    )
    aux = {"sigma": sigma}
    threshold = score.score(pred, target, aux=aux)
    assert threshold.dtype == torch.result_type(pred, target)
    assert torch.isfinite(threshold).all()
    assert_admitted_covered(score, pred, target, threshold, aux)


_D = AuxScale(key="sigma")
_S = NormalizedErrorScore()
_Z2, _O2 = torch.zeros(2), torch.ones(2)
# fmt: off
STRATEGY_REJECTIONS = [  # (id, thunk, error, match)
    ("scale-missing-aux-key", lambda: _D(torch.rand(2, 2)), ValueError, "sigma"),
    ("scale-aux-not-mapping", lambda: _D(aux=[_O2]), ValueError, "requires aux entry"),
    ("score-missing-sigma", lambda: _S.score(_Z2, _O2), ValueError, "sigma"),
    ("score-aux-not-mapping", lambda: _S.score(_Z2, _O2, aux=[_O2]), ValueError, "requires aux entries"),
    ("key-int", lambda: AuxScale(key=7), TypeError, "key must be a string"),
]
# fmt: on


@pytest.mark.parametrize(
    "thunk,error,match",
    [pytest.param(*row[1:], id=row[0]) for row in STRATEGY_REJECTIONS],
)
def test_strategy_inputs_and_configuration_are_validated(thunk, error, match):
    with pytest.raises(error, match=match):
        thunk()
