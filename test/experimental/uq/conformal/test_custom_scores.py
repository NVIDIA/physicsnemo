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

"""Custom nonconformity scores: registration, the inversion check, and save/load."""

import pytest
import torch

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    ConformalPredictor,
    NonconformityScore,
    NormalizedErrorScore,
    PointwiseCalibrator,
    QuantileRegressionScore,
    check_score,
    register_score,
)
from physicsnemo.experimental.uq.conformal.scores import _SCORE_REGISTRY
from test.experimental.uq.conformal._helpers import (
    CALIBRATORS,
    TIERS,
    assert_predictor_covers_admitted,
    fit,
)


@pytest.fixture(autouse=True)
def restore_registry():
    """Undo the registrations each test makes."""
    saved = dict(_SCORE_REGISTRY)
    yield
    _SCORE_REGISTRY.clear()
    _SCORE_REGISTRY.update(saved)


class _RelativeErrorScore(NonconformityScore):
    """``|y - p| / max(|p|, floor)``; reads aux ``weight`` to exercise aux_keys."""

    aux_keys = ("weight",)
    config_fields = ("floor",)

    def __init__(self, floor: float = 1e-3):
        self.floor = floor

    def score(self, prediction, target, *, aux=None):
        scale = prediction.abs().clamp_min(self.floor) * aux["weight"]
        return (target - prediction).abs() / scale

    def bounds(self, prediction, threshold, *, aux=None):
        scale = prediction.abs().clamp_min(self.floor) * aux["weight"].double()
        return prediction - threshold * scale, prediction + threshold * scale


class _HalfWidthScore(AbsoluteErrorScore):
    """Broken: bounds() is half as wide as score() requires."""

    def bounds(self, prediction, threshold, *, aux=None):
        return prediction - threshold / 2, prediction + threshold / 2


def _weight(prediction, generator):
    draw = torch.rand(prediction.shape, generator=generator, device=prediction.device)
    return {"weight": (0.5 + draw).to(prediction.dtype)}


@pytest.mark.parametrize("tier", TIERS)
def test_custom_score_calibrates_predicts_and_round_trips(tmp_path, tier, device):
    register_score("test_relative_error")(_RelativeErrorScore)
    predictor, points = fit(
        tier,
        score=_RelativeErrorScore(floor=0.25),
        aux_factory=_weight,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(1)
    prediction = torch.randn(200, generator=generator, device=device)
    target = prediction + 0.3 * torch.randn(200, generator=generator, device=device)
    aux = _weight(prediction, generator)
    lo, hi = assert_predictor_covers_admitted(
        predictor, prediction, target, aux=aux, points=points
    )

    predictor.save(tmp_path / "relative.pt")
    loaded = ConformalPredictor.load(tmp_path / "relative.pt", map_location=device)
    assert type(loaded.score) is _RelativeErrorScore and loaded.score.floor == 0.25
    lo2, hi2 = loaded.predict_interval(prediction, aux=aux, points=points)
    torch.testing.assert_close(lo2, lo, rtol=0, atol=0)
    torch.testing.assert_close(hi2, hi, rtol=0, atol=0)

    del _SCORE_REGISTRY["test_relative_error"]
    with pytest.raises(ValueError, match="import the module that registers it"):
        ConformalPredictor.load(tmp_path / "relative.pt")


@pytest.mark.parametrize("tier", TIERS)
def test_custom_score_requires_its_aux_entries(tier):
    register_score("test_relative_error")(_RelativeErrorScore)
    calibrator = CALIBRATORS[tier](_RelativeErrorScore(), alpha=0.5)
    points = torch.zeros(3, 1) if tier == "pointwise" else None
    with pytest.raises(ValueError, match=r"requires aux entries \['weight'\]"):
        calibrator.update(torch.ones(3), torch.ones(3), points=points)


@pytest.mark.parametrize("tier", TIERS)
def test_first_sample_rejects_a_score_whose_bounds_do_not_invert_it(tier):
    register_score("test_half_width")(_HalfWidthScore)
    calibrator = CALIBRATORS[tier](_HalfWidthScore(), alpha=0.5)
    points = torch.zeros(4, 1) if tier == "pointwise" else None
    with pytest.raises(ValueError, match="does not invert score"):
        calibrator.update(torch.zeros(4), torch.ones(4), points=points)
    assert calibrator.n_cal == 0
    with pytest.raises(ValueError, match="4 of 4 targets"):
        check_score(_HalfWidthScore(), torch.zeros(4), torch.ones(4))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], ids=str
)
def test_check_score_accepts_builtin_scores_in_low_precision(dtype, device):
    generator = torch.Generator(device=device).manual_seed(0)
    prediction = (100 * torch.randn(500, generator=generator, device=device)).to(dtype)
    target = prediction + torch.randn(500, generator=generator, device=device).to(dtype)
    spread = (torch.rand(500, generator=generator, device=device) + 0.1).to(dtype)
    check_score(AbsoluteErrorScore(), prediction, target)
    check_score(NormalizedErrorScore(), prediction, target, aux={"sigma": spread})
    check_score(
        QuantileRegressionScore(),
        prediction,
        target,
        aux={"lo": prediction - spread, "hi": prediction + spread},
    )


def test_registration_is_validated():
    register_score("test_relative_error")(_RelativeErrorScore)
    # Re-running the defining module re-registers the same class.
    register_score("test_relative_error")(_RelativeErrorScore)
    with pytest.raises(ValueError, match="already registered"):
        register_score("absolute_error")(_RelativeErrorScore)
    with pytest.raises(TypeError, match="non-empty string"):
        register_score("")
    with pytest.raises(TypeError, match="NonconformityScore subclass"):
        register_score("test_not_a_score")(dict)

    class BadFields(NonconformityScore):
        config_fields = "floor"

    with pytest.raises(TypeError, match="tuple of attribute names"):
        register_score("test_bad_fields")(BadFields)


def test_config_values_must_be_saveable_primitives():
    register_score("test_relative_error")(_RelativeErrorScore)
    with pytest.raises(TypeError, match="must be a bool, int, float, or str"):
        PointwiseCalibrator(_RelativeErrorScore(floor=torch.tensor(0.1)), alpha=0.5)
