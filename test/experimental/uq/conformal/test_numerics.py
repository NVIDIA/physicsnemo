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

Exact-arithmetic release gates (conformal rank in exact decimal), the theorem-preserving
containment property (score-admitted targets stay inside the reconstructed
interval across scores, dtypes, tiers and persistence), and the
quantile/container utility layer.
"""

import math
from decimal import ROUND_CEILING, Decimal

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    ConformalPredictor,
    NormalizedErrorScore,
    QuantileRegressionScore,
)
from physicsnemo.experimental.uq.conformal._utils import (
    TENSOR_KEY,
    cast_directed,
    conformal_quantile_index,
    field_items,
    kth_smallest_of_samples,
    pack_fields,
    slice_aux,
)
from test.experimental.uq.conformal._helpers import (
    CALIBRATOR_CLASSES,
    TIERS,
    assert_admitted_covered,
    assert_predictor_covers_admitted,
    fit,
)

_H, _BF, _F32, _F64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
FLOAT_DTYPES = [_H, _BF, _F32, _F64]
LOW_PRECISION = [_H, _BF, _F32]

# =========================================================================
# Exact-arithmetic release gates.  Approximate stopping rules and floating
# tolerances are not allowed at the theorem boundary.
# =========================================================================


def _rank_oracle(n_cal, alpha):
    """Decimal reference for k = ceil((n+1)(1-alpha))."""
    scaled = Decimal(n_cal + 1) * (Decimal(1) - Decimal(str(alpha)))
    return int(scaled.to_integral_value(rounding=ROUND_CEILING))


# fmt: off
RANK_CASES = [
    (149, 0.18),  # (150)(0.82) floats to 123.00000000000001 -> must stay 123
    (9, 0.49999999995),  # genuine offset -> must round UP to 6
    (9, 0.5), (99, 0.05), (22, 0.1),
    (10, 1 - 1e-12),  # k must clamp to a valid rank, never 0
    (1, 0.5),  # smallest feasible population: k = 1
    (2, 0.32),  # ceil(3 * 0.68) = 3 = n_cal
    (19, 0.05),  # 20 * 0.95 floats to 19.000000000000004 -> must stay 19
    (5, 0.1),  # ceil(6 * 0.9) = 6 > n_cal: infeasible
]
# fmt: on


@pytest.mark.parametrize("n_cal,alpha", RANK_CASES)
def test_conformal_rank_exact_decimal(n_cal, alpha):
    """Rank arithmetic follows the declared decimal alpha exactly, no
    tolerance snapping in either direction; infeasible levels refuse."""
    expected = _rank_oracle(n_cal, alpha)
    if expected > n_cal:
        with pytest.raises(ValueError, match="calibration samples"):
            conformal_quantile_index(n_cal, alpha)
        return
    k = conformal_quantile_index(n_cal, alpha)
    assert k == expected
    assert 1 <= k <= n_cal


@pytest.mark.parametrize("base_alpha", [0.1, 0.2, 0.25, 0.5])
@pytest.mark.parametrize("n_cal", [9, 19, 99])
def test_conformal_rank_nextafter_boundaries(base_alpha, n_cal):
    """Probing one float ulp on each side of an integral rank boundary must
    match the decimal oracle on that exact perturbed value."""
    for alpha in (
        base_alpha,
        math.nextafter(base_alpha, 0.0),
        math.nextafter(base_alpha, 1.0),
    ):
        expected = _rank_oracle(n_cal, alpha)
        if expected > n_cal:
            with pytest.raises(ValueError, match="calibration samples"):
                conformal_quantile_index(n_cal, alpha)
        else:
            assert conformal_quantile_index(n_cal, alpha) == expected, (
                f"alpha={alpha!r}"
            )


# =========================================================================
# The theorem-preserving numerical-policy property.
#
# The product contract is exact finite-sample coverage, so the
# finite-precision realization must satisfy, for every score, dtype, tier and
# persistence path::
#
#     score(prediction, target, aux) <= radius   (working-dtype arithmetic)
#         implies
#     lo <= target <= hi
# =========================================================================


@pytest.mark.parametrize("persist", [False, True], ids=["in_memory", "save_load"])
@pytest.mark.parametrize("dtype", LOW_PRECISION, ids=str)
@pytest.mark.parametrize("tier", TIERS)
def test_admitted_targets_stay_inside_interval(tier, dtype, persist, tmp_path):
    """End-to-end containment: a fresh target admitted by the fitted
    statistic lies inside the reconstructed interval, across tiers,
    low-precision dtypes, and the artifact round trip (which must never
    shrink a threshold)."""
    generator = torch.Generator().manual_seed(31)
    predictor, points = fit(tier, generator=generator, dtype=dtype)
    if persist:
        original = predictor.thresholds.double()
        predictor.save(tmp_path / "artifact.pt")
        predictor = ConformalPredictor.load(tmp_path / "artifact.pt")
        assert bool((predictor.thresholds.double() >= original).all())

    pred = torch.randn(200, generator=generator).to(dtype)
    target = (pred.float() + 0.3 * torch.randn(200, generator=generator)).to(dtype)
    assert_predictor_covers_admitted(predictor, pred, target, points=points)


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


@pytest.mark.parametrize("cls", CALIBRATOR_CLASSES)
def test_empty_samples_rejected_uniformly(cls):
    """Every tier rejects an empty sample with the same transactional error."""
    calibrator = cls(AbsoluteErrorScore(), alpha=0.25)
    with pytest.raises(ValueError, match="empty sample"):
        calibrator.update(torch.zeros(0), torch.zeros(0), points=torch.zeros(1, 1))
    assert calibrator.n_cal == 0


# =========================================================================
# Quantile and container utilities.
# =========================================================================


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided"])
def test_kth_smallest_of_samples_matches_stacked_corpus(layout):
    """The chunked per-sample path must equal the stacked reference for
    every chunking, including one that splits a cell block mid-field."""
    generator = torch.Generator().manual_seed(23)
    scores = torch.randn(9, 5, 3, generator=generator)
    if layout == "transposed":
        scores = scores.transpose(1, 2)
    elif layout == "strided":
        scores = scores[:, ::2, :]
    per_sample = list(scores.unbind(0))
    reference = torch.sort(scores, dim=0).values[3]
    for chunk_numel in (2**26, 16, 9):
        assert torch.equal(
            kth_smallest_of_samples(per_sample, 4, chunk_numel=chunk_numel), reference
        )


def test_kth_smallest_of_samples_promotes_mixed_dtypes():
    """A float32 first sample must not round the float64 k-th order
    statistic down; the result carries the promoted dtype exactly."""
    per_sample = [
        torch.tensor([0.5], dtype=_F32),
        torch.tensor([1.0 + 2**-40], dtype=_F64),
        torch.tensor([2.0], dtype=_F64),
    ]
    out = kth_smallest_of_samples(per_sample, 2)
    assert out.dtype == _F64
    assert float(out) == 1.0 + 2**-40


def test_field_items_pack_fields_and_slice_aux_round_trip():
    t = torch.randn(4)
    items = field_items(t)
    assert items == [(TENSOR_KEY, t)]
    assert pack_fields(dict(items)) is t

    td = TensorDict({"b": torch.randn(3), "a": torch.randn(3)}, batch_size=[])
    items = field_items(td)
    assert [k for k, _ in items] == ["a", "b"]
    assert [k for k, _ in field_items(td, keys=["b"])] == ["b"]
    packed = pack_fields(dict(items))
    assert isinstance(packed, TensorDict)
    torch.testing.assert_close(packed["a"], td["a"])

    aux_tensor_mode = {"sigma": torch.ones(3)}
    assert slice_aux(aux_tensor_mode, TENSOR_KEY) is aux_tensor_mode
    aux_field_mode = {"pressure": {"sigma": torch.ones(3)}}
    assert slice_aux(aux_field_mode, "pressure") == aux_field_mode["pressure"]
    assert slice_aux(aux_field_mode, "velocity") is None
    assert slice_aux(None, "pressure") is None


def _td(**fields):
    return TensorDict(fields, batch_size=[])


# fmt: off
CONTAINER_REJECTIONS = [  # (id, thunk, error, match)
    ("missing-key", lambda: field_items(_td(a=torch.ones(2)), ["c"]), KeyError, "not present"),
    ("empty-tensordict", lambda: field_items(TensorDict({}, batch_size=[])), ValueError, "at least one field"),
    ("reserved-sentinel", lambda: field_items(_td(**{TENSOR_KEY: torch.ones(2)})), ValueError, "reserved"),
    ("keys-with-plain-tensor", lambda: field_items(torch.zeros(3), keys=["pressure"]), TypeError, "plain tensor"),
    ("aux-entry-not-mapping", lambda: slice_aux({"pressure": torch.ones(3)}, "pressure"), TypeError, "mapping"),
    ("flat-aux-for-fields", lambda: slice_aux({"sigma": torch.ones(3)}, "pressure"), TypeError, "nested by field"),
]
# fmt: on


@pytest.mark.parametrize(
    "thunk,error,match",
    [pytest.param(t, e, m, id=i) for i, t, e, m in CONTAINER_REJECTIONS],
)
def test_container_contract_rejections(thunk, error, match):
    with pytest.raises(error, match=match):
        thunk()
