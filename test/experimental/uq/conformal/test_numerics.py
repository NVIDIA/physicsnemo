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

"""Numerical guarantees at the theorem boundary.

The theorem-preserving containment property: score-admitted targets stay
inside the reconstructed interval across scores and dtypes.
"""

import pytest
import torch

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    NormalizedErrorScore,
    QuantileRegressionScore,
)
from physicsnemo.experimental.uq.conformal._utils import cast_directed
from test.experimental.uq.conformal._helpers import assert_admitted_covered

_H, _BF, _F32, _F64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
FLOAT_DTYPES = [_H, _BF, _F32, _F64]

# =========================================================================
# The theorem-preserving numerical-policy property.
#
# The product contract is exact finite-sample coverage, so the
# finite-precision realization must satisfy, for every score and dtype::
#
#     score(prediction, target, aux) <= radius   (working-dtype arithmetic)
#         implies
#     lo <= target <= hi
# =========================================================================


def _adversarial_pairs(dtype: torch.dtype, generator: torch.Generator):
    """Predictions/targets spanning magnitudes, signs, and near-cancellation.

    Includes the hazardous regimes: |prediction| >> radius (endpoint ulp is
    the hazard) and |prediction| << radius (near-cancellation endpoints).
    """
    exps = torch.tensor([-4.0, -1.0, 0.0, 1.0, 3.0], dtype=_F64)
    pred_mag = (10.0**exps).repeat_interleave(exps.numel())
    delta_mag = (10.0**exps).repeat(exps.numel())
    signs_p = torch.where(
        torch.rand(pred_mag.shape, generator=generator) < 0.5, -1.0, 1.0
    )
    signs_d = torch.where(
        torch.rand(delta_mag.shape, generator=generator) < 0.5, -1.0, 1.0
    )
    pred64 = signs_p * pred_mag
    target64 = pred64 + signs_d * delta_mag
    noise_p = torch.randn(400, generator=generator, dtype=_F64)
    noise_t = noise_p + 0.1 * torch.randn(400, generator=generator, dtype=_F64)
    pred = torch.cat([pred64, noise_p]).to(dtype)
    target = torch.cat([target64, noise_t]).to(dtype)
    return pred, target


def _sigma_aux(pred, generator):
    scale = 10.0 ** (4.0 * torch.rand(pred.shape, generator=generator) - 2.0)
    return {"sigma": scale.to(pred.dtype)}


def _cqr_aux(pred, generator):
    spread = (0.5 * torch.rand(pred.shape, generator=generator) + 0.1).to(pred.dtype)
    return {"lo": pred - spread, "hi": pred + spread}


SCORE_CASES = [
    pytest.param(AbsoluteErrorScore(), None, id="absolute"),
    pytest.param(NormalizedErrorScore(), _sigma_aux, id="normalized"),
    pytest.param(QuantileRegressionScore(), _cqr_aux, id="quantile_regression"),
]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("score,aux_factory", SCORE_CASES)
def test_score_boundary_inversion(score, aux_factory, dtype):
    """Worst case (every element's radius equals its own score) and the
    scalar-threshold shape both keep admitted targets inside the band."""
    generator = torch.Generator().manual_seed(11)
    pred, target = _adversarial_pairs(dtype, generator)
    aux = aux_factory(pred, generator) if aux_factory else None
    radius = score.score(pred, target, aux=aux)
    assert_admitted_covered(score, pred, target, radius, aux)
    finite = radius[torch.isfinite(radius)]
    assert_admitted_covered(score, pred, target, finite.median(), aux)


# fmt: off
# Reproduced historical counterexamples:
# name: (score, score-side prediction, interval-side prediction, target, aux, threshold or None for the sup)
KNOWN_COUNTEREXAMPLES = {
    # fp16 sigma with an unrepresentable eps must give finite scores AND a consistent interval.
    "fp16_sigma_floor": (
        NormalizedErrorScore(), torch.zeros(4, dtype=_H), torch.zeros(4, dtype=_H),
        torch.tensor([0.1, -0.2, 0.05, 0.0], dtype=_H), {"sigma": torch.zeros(4, dtype=_H)}, None,
    ),
    # fp16 CQR heads with a float64 prediction: slack must be sized by the coarsest dtype.
    "fp16_cqr_mixed_dtype": (
        QuantileRegressionScore(), torch.zeros(1, dtype=_H), torch.zeros(1, dtype=_F64),
        torch.tensor([-29504.0], dtype=_H),
        {"lo": torch.tensor([-8440.0], dtype=_H), "hi": torch.tensor([-8440.0], dtype=_H)},
        torch.tensor(21056.0),
    ),
}
# fmt: on


@pytest.mark.parametrize("case", sorted(KNOWN_COUNTEREXAMPLES))
def test_known_counterexamples_are_contained(case):
    score, pred, interval_pred, target, aux, threshold = KNOWN_COUNTEREXAMPLES[case]
    s = score.score(pred, target, aux=aux)
    assert bool(torch.isfinite(s).all())
    threshold = s.amax() if threshold is None else threshold
    assert bool((s <= threshold).all())
    lo, hi = score.interval(interval_pred, threshold, aux=aux)
    inside = (target.double() >= lo.double()) & (target.double() <= hi.double())
    assert bool(inside.all())


@pytest.mark.parametrize("dtype", [_F32, _F64], ids=str)
def test_interval_width_stays_tight(dtype):
    """Conservatism is a few ulps, not a blow-up: the excess over the exact
    width ``2r`` is the radius inflation (``~4 eps |r|``) plus one
    outward-rounded ulp per endpoint (``~2 eps |endpoint|`` each)."""
    generator = torch.Generator().manual_seed(19)
    pred, target = _adversarial_pairs(dtype, generator)
    score = AbsoluteErrorScore()
    radius = score.score(pred, target)
    finite = torch.isfinite(radius) & (radius > 0)
    lo, hi = score.interval(pred, radius)
    width = hi.double() - lo.double()
    eps = torch.finfo(dtype).eps
    endpoint_mag = torch.maximum(lo.double().abs(), hi.double().abs())
    bound = (
        2.0 * radius.double() + 16.0 * eps * (radius.double() + endpoint_mag) + 1e-30
    )
    assert (width[finite] <= bound[finite]).all()


def test_cast_directed_fp16_bf16_and_narrowing(device):
    # FP16 -> BF16 is a 16->16-bit cast that LOSES mantissa precision; a
    # total-bit-width test would skip the conservative bump.
    t = torch.tensor([1.00390625, -1.00390625], dtype=_H, device=device)
    assert (cast_directed(t, _BF, up=True).double() >= t.double()).all()
    assert (cast_directed(t, _BF, up=False).double() <= t.double()).all()
    # float64 -> float32 narrowing in both directions.
    t64 = torch.tensor([1.0 + 1e-9, -1.0 - 1e-9, 0.1], dtype=_F64, device=device)
    assert (cast_directed(t64, _F32, up=True).double() >= t64).all()
    assert (cast_directed(t64, _F32, up=False).double() <= t64).all()
    # Same-dtype passthrough returns the input itself; upcasts are exact.
    assert cast_directed(t64, _F64, up=True) is t64
    t16 = torch.tensor([0.1, 3.0], dtype=_H)
    assert torch.equal(cast_directed(t16, _F64, up=True), t16.to(_F64))
