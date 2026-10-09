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

"""Shared tables, fitters, and containment assertions for the conformal suite.

Plain module attributes, imported explicitly
(``from test.experimental.uq.conformal._helpers import ...``). Fixtures
live in ``conftest.py``.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    CellwiseCalibrator,
    ConformalPredictor,
)
from physicsnemo.experimental.uq.conformal._utils import TIERS

__all__ = [
    "CALIBRATORS",
    "CALIBRATOR_CLASSES",
    "MESH_POINTS",
    "TIERS",
    "assert_admitted_covered",
    "assert_predictor_covers_admitted",
    "count_syncs",
    "fit",
    "make_predictor",
]

CALIBRATORS = {
    "cellwise": CellwiseCalibrator,
}
CALIBRATOR_CLASSES = [
    pytest.param(CALIBRATORS[tier], id=tier) for tier in sorted(CALIBRATORS)
]
MESH_POINTS = torch.arange(3.0).reshape(3, 1)


def fit(
    tier: str,
    *,
    generator: torch.Generator | None = None,
    score=None,
    alpha: float = 0.2,
    n_samples: int = 30,
    shape: tuple[int, ...] = (200,),
    fields: list[str] | None = None,
    aux_factory=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Fit one predictor on ``n_samples`` seeded ``pred + 0.3 * noise`` samples.

    Returns ``(predictor, points)``: the predictor calibrates on a fixed
    ``(shape[0], 1)`` mesh passed on every call (and needed again at
    prediction time). ``fields``
    switches to ``TensorDict`` mode (one independent draw per field, so a
    single-field container sees exactly the tensor-mode data for the same
    seed). ``aux_factory(prediction, generator)`` builds the per-field aux
    mapping.
    """
    generator = generator or torch.Generator(device=device).manual_seed(0)
    calibrator = CALIBRATORS[tier](score or AbsoluteErrorScore(), alpha)
    points = torch.arange(shape[0], dtype=torch.float64, device=device)
    points = points.reshape(shape[0], 1)

    def draw():
        pred = torch.randn(shape, generator=generator, device=device)
        target = pred + 0.3 * torch.randn(shape, generator=generator, device=device)
        return pred.to(dtype), target.to(dtype)

    for _ in range(n_samples):
        if fields is None:
            pred, target = draw()
            aux = aux_factory(pred, generator) if aux_factory else None
        else:
            pairs = {key: draw() for key in fields}
            pred = TensorDict({k: p for k, (p, _) in pairs.items()}, batch_size=[])
            target = TensorDict({k: t for k, (_, t) in pairs.items()}, batch_size=[])
            aux = (
                {k: aux_factory(p, generator) for k, (p, _) in pairs.items()}
                if aux_factory
                else None
            )
        calibrator.update(pred, target, aux=aux, points=points)
    return calibrator.finalize(), points


def make_predictor(**overrides) -> ConformalPredictor:
    """Directly constructed predictor with valid defaults on a 3-point mesh."""
    kwargs = {
        "tier": "cellwise",
        "score": AbsoluteErrorScore(),
        "alpha": 0.5,
        "n_cal": 3,
        "thresholds": torch.full((3,), 0.5),
        "points": MESH_POINTS,
    }
    kwargs.update(overrides)
    return ConformalPredictor(**kwargs)


def assert_admitted_covered(score, pred, target, threshold, aux) -> None:
    """Score-level property: ``score <= threshold`` implies ``lo <= target <= hi``."""
    s = score.score(pred, target, aux=aux)
    admitted = torch.isfinite(s) & (s <= threshold) & torch.isfinite(target)
    lo, hi = score.interval(pred, threshold, aux=aux)
    assert lo.dtype == pred.dtype and hi.dtype == pred.dtype
    bad = admitted & ~((target >= lo) & (target <= hi))
    assert not bad.any(), (
        f"{int(bad.sum())} score-admitted target(s) excluded; first at "
        f"index {int(bad.nonzero()[0])}"
    )


def assert_predictor_covers_admitted(
    predictor, prediction, target, *, aux=None, points=None
):
    """Predictor-level property through the fitted threshold.

    Returns the interval so callers can make further assertions.
    """
    score = predictor.score.score(prediction, target, aux=aux).double()
    threshold = predictor.thresholds.double()
    lo, hi = predictor.predict_interval(prediction, aux=aux, points=points)
    inside = (target.double() >= lo.double()) & (target.double() <= hi.double())
    bad = (score <= threshold) & ~inside
    assert not bad.any(), f"{int(bad.sum())} admitted target(s) excluded"
    return lo, hi


_SYNC_METHODS = ("__bool__", "__int__", "__float__", "__index__", "item", "tolist")


def count_syncs(monkeypatch) -> list[str]:
    """Record every tensor-to-Python conversion (a device sync on GPU).

    Returns the list that collects the name of each conversion method called.
    """
    calls: list[str] = []
    for name in _SYNC_METHODS:
        original = getattr(torch.Tensor, name)

        def counted(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, counted)
    return calls
