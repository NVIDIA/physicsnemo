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

"""Shared input-contract validators, and their enforcement at every entry point.

The validators are tested once, directly; one matrix then proves that each
public boundary (every calibrator tier, interval construction, diagnostics)
routes through them with the same error and leaves its state untouched.
"""

from fractions import Fraction
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.experimental.uq.conformal import (
    AbsoluteErrorScore,
    NormalizedErrorScore,
)
from physicsnemo.experimental.uq.conformal._utils import (
    TENSOR_KEY,
    broadcast_difficulty,
    check_aux,
    check_exact_shape,
    check_points,
    conformal_quantile_index,
    normalize_keys,
    points_fingerprint,
    positive_finite_float,
    require_matching_keys,
    validate_alpha,
    validate_n_cal,
    validate_provenance,
)
from test.experimental.uq.conformal._helpers import (
    CALIBRATORS,
    TIERS,
    fit,
    fitted_risk,
)

_F = torch.zeros(3)
_I64 = torch.zeros(3, dtype=torch.int64)
_INF3 = torch.tensor([1.0, torch.inf, 1.0])
_N = NormalizedErrorScore()
_P64 = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
_P64_ULP = _P64.clone()
_P64_ULP[1, 0] = torch.nextafter(
    _P64_ULP[1, 0], torch.tensor(torch.inf, dtype=torch.float64)
)
_NOT_REAL = (TypeError, "real number")
_NOT_EXACT = (TypeError, "exactly representable")
_NOT_INT = (TypeError, "integer")
_NOT_POSITIVE = (ValueError, "positive finite")

# fmt: off
# (id, thunk, expected): expected is (ExceptionType, match) or a return value.
VALIDATOR_CASES = [
    ("check_exact_shape", lambda: check_exact_shape("k", "p", _F, "t", torch.zeros(3, 1)), (ValueError, "match exactly")),
    ("require_matching_keys", lambda: require_matching_keys(["a", "b"], ["a", "c"], "m"), (KeyError, r"\['b', 'c'\]")),
    ("check_aux-shape", lambda: check_aux("k", _N, torch.zeros(2, 1), {"sigma": torch.ones(2)}), (ValueError, "aux 'sigma' shape")),
    ("check_aux-not-tensor", lambda: check_aux("k", _N, _F, {"sigma": 1.0}), (TypeError, "must be a torch.Tensor")),
    ("check_aux-integer", lambda: check_aux("k", _N, _F, {"sigma": _I64}), (TypeError, "floating")),
    ("check_aux-nonfinite", lambda: check_aux("k", _N, _F, {"sigma": _INF3}), (ValueError, "non-finite")),
    ("check_aux-plain-label", lambda: check_aux(TENSOR_KEY, _N, _F, {"sigma": 1.0}), (TypeError, "^Plain tensor: aux")),
    ("broadcast_difficulty-plain-label", lambda: broadcast_difficulty(torch.ones(2), _F, TENSOR_KEY), (ValueError, "^Plain tensor: AuxDifficulty gave")),
    ("keys-bare-string", lambda: normalize_keys("pressure"), (TypeError, "not the string")),
    ("keys-order-preserved", lambda: normalize_keys(["b", "a"]), ("b", "a")),
    ("keys-none", lambda: normalize_keys(None), None),
    ("alpha-bool", lambda: validate_alpha(True), _NOT_REAL),
    ("alpha-string", lambda: validate_alpha("0.5"), _NOT_REAL),
    ("alpha-tensor", lambda: validate_alpha(torch.tensor(0.5)), _NOT_REAL),
    ("alpha-zero", lambda: validate_alpha(0.0), (ValueError, r"in \(0, 1\)")),
    ("alpha-one", lambda: validate_alpha(1.0), (ValueError, r"in \(0, 1\)")),
    ("alpha-nan", lambda: validate_alpha(float("nan")), (ValueError, "finite")),
    # Fraction(1, 3) floored to float would select rank 5 instead of the exact-intent
    # 4, and a Fraction just below 1/5 would mask infeasibility: reject, never floor.
    ("alpha-fraction-third", lambda: validate_alpha(Fraction(1, 3)), _NOT_EXACT),
    ("alpha-fraction-below-fifth", lambda: validate_alpha(Fraction(3602879701896396, 18014398509481985)), _NOT_EXACT),
    ("alpha-fraction-exact", lambda: validate_alpha(Fraction(1, 2)), 0.5),
    ("rank-fraction-exact", lambda: conformal_quantile_index(3, Fraction(1, 2)), 2),
    ("n_cal-float", lambda: validate_n_cal(3.7), _NOT_INT),
    ("n_cal-bool", lambda: validate_n_cal(True), _NOT_INT),
    ("n_cal-zero", lambda: validate_n_cal(0), (ValueError, ">= 1")),
    ("rank-infeasible-names-minimum", lambda: conformal_quantile_index(5, 0.1), (ValueError, "at least 9 calibration samples")),
    ("eps-nan", lambda: positive_finite_float(float("nan"), "eps"), _NOT_POSITIVE),
    ("eps-inf", lambda: positive_finite_float(float("inf"), "eps"), _NOT_POSITIVE),
    ("eps-zero", lambda: positive_finite_float(0.0, "eps"), _NOT_POSITIVE),
    ("eps-none", lambda: positive_finite_float(None, "eps"), (ValueError, "positive finite value")),
    ("eps-string", lambda: positive_finite_float("not a number", "eps"), (ValueError, "positive finite value")),
    ("eps-tensor-coerced", lambda: type(positive_finite_float(torch.tensor(1e-3), "eps")) is float, True),
    ("points-1d", lambda: check_points(torch.ones(3)), (ValueError, "shape")),
    ("points-empty", lambda: check_points(torch.empty(0, 2)), (ValueError, "non-empty")),
    ("points-integer", lambda: check_points(torch.ones(3, 2, dtype=torch.int64)), (TypeError, "floating")),
    ("points-nan", lambda: check_points(torch.tensor([[0.0], [torch.nan]])), (ValueError, "non-finite")),
    ("fingerprint-not-tensor", lambda: points_fingerprint([[0.0, 1.0]]), (TypeError, "torch.Tensor")),
    ("fingerprint-dtype-sensitive", lambda: points_fingerprint(_P64) != points_fingerprint(_P64.float()), True),
    ("fingerprint-order-sensitive", lambda: points_fingerprint(_P64) != points_fingerprint(_P64.flip(0)), True),
    ("fingerprint-value-identity", lambda: points_fingerprint(_P64) == points_fingerprint(_P64.clone()), True),
    ("fingerprint-shape-sensitive", lambda: points_fingerprint(_P64) != points_fingerprint(_P64.reshape(-1, 1)), True),
    ("fingerprint-mesh-sensitive", lambda: points_fingerprint(_P64) != points_fingerprint(_P64 + 1), True),
    ("fingerprint-one-ulp", lambda: points_fingerprint(_P64) != points_fingerprint(_P64_ULP), True),
    ("fingerprint-format", lambda: points_fingerprint(_P64).startswith("float64-2x2-"), True),
    ("fingerprint-nan", lambda: points_fingerprint(torch.tensor([[0.0], [torch.nan]])), (ValueError, "non-finite")),
    ("provenance-not-mapping", lambda: validate_provenance([1, 2, 3]), (TypeError, "mapping")),
    ("provenance-path", lambda: validate_provenance({"checkpoint": Path("/ckpt.pt")}), (TypeError, "strict-JSON")),
    ("provenance-tensor", lambda: validate_provenance({"value": torch.tensor(1.0)}), (TypeError, "strict-JSON")),
    ("provenance-tuple", lambda: validate_provenance({"labels": ("a",)}), (TypeError, "strict-JSON")),
    ("provenance-non-string-key", lambda: validate_provenance({7: "x"}), (TypeError, "strict-JSON")),
    ("provenance-nan", lambda: validate_provenance({"m": float("nan")}), (ValueError, "finite")),
    ("provenance-reserved-key", lambda: validate_provenance({"mesh_fingerprint": "x"}), (ValueError, "mesh_fingerprint")),
]
# fmt: on


@pytest.mark.parametrize(
    "thunk,expected", [pytest.param(t, e, id=i) for i, t, e in VALIDATOR_CASES]
)
def test_shared_validators(thunk, expected):
    raises = isinstance(expected, tuple) and isinstance(expected[0], type)
    if raises:
        with pytest.raises(expected[0], match=expected[1]):
            thunk()
    else:
        assert thunk() == expected


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_points_fingerprint_is_device_and_layout_independent(device, dtype):
    points = torch.randn(3, 1000, dtype=torch.float64).to(dtype).T
    expected = points_fingerprint(points.contiguous())
    assert not points.is_contiguous()
    assert points_fingerprint(points) == expected
    assert points_fingerprint(points.to(device)) == expected
    assert points_fingerprint(points.to(device).contiguous()) == expected


# -------------------------------------------------------------------------
# Entry-point matrix: every boundary rejects the same bad inputs with the
# same error and commits nothing.
# -------------------------------------------------------------------------

BAD_INPUTS = {
    "nonfinite": (torch.tensor([0.0, torch.inf, 1.0]), ValueError, "non-finite"),
    "integer_dtype": (_I64, TypeError, "floating"),
    "non_container": ("oops", TypeError, "torch.Tensor or TensorDict"),
    "tensordict_for_tensor": (
        TensorDict({"p": _F}, batch_size=[]),
        TypeError,
        "TensorDict",
    ),
}
_POINTS = torch.arange(3.0).reshape(3, 1)


def _calibrator_entry(tier):
    def build():
        calibrator = CALIBRATORS[tier](AbsoluteErrorScore(), alpha=0.5)
        points = _POINTS if tier == "cellwise" else None
        calibrator.update(_F, _F, points=points)
        return (
            lambda bad: calibrator.update(_F, bad, points=points),
            lambda: calibrator.n_cal,
        )

    return build


def _predictor_entry():
    predictor = fitted_risk()
    return predictor.predict_interval, lambda: predictor.thresholds


def _accumulator_entry():
    accumulator = fitted_risk().coverage_accumulator()
    accumulator.update(-torch.ones(3), torch.ones(3), _F)
    return (
        lambda bad: accumulator.update(-torch.ones(3), torch.ones(3), bad),
        accumulator.finalize,
    )


ENTRY_POINTS = {f"calibrator[{tier}].update": _calibrator_entry(tier) for tier in TIERS}
ENTRY_POINTS["predictor.predict_interval"] = _predictor_entry
ENTRY_POINTS["accumulator.update"] = _accumulator_entry


@pytest.mark.parametrize("bad_kind", sorted(BAD_INPUTS))
@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_entry_points_reject_bad_inputs_transactionally(entry_point, bad_kind):
    call, state = ENTRY_POINTS[entry_point]()
    bad, error, match = BAD_INPUTS[bad_kind]
    before = state()
    with pytest.raises(error, match=match) as excinfo:
        call(bad)
    assert TENSOR_KEY not in str(excinfo.value)
    after = state()
    if isinstance(before, torch.Tensor):
        assert torch.equal(before, after)
    else:
        assert before == after


def test_field_mode_entry_points_reject_plain_tensors():
    predictor, _ = fit("risk_control", n_samples=10, shape=(3,), fields=["p"])
    accumulator = predictor.coverage_accumulator()
    message = r"calibrated on TensorDict fields \['p'\]; pass {} as a TensorDict"
    with pytest.raises(TypeError, match=message.format("prediction")):
        predictor.predict_interval(_F)
    with pytest.raises(TypeError, match=message.format("lo")):
        accumulator.update(_F, _F, _F)


def test_tensordict_inputs_take_the_same_validation_path():
    """Field containers route each field through the shared validators."""
    calibrator = CALIBRATORS["risk_control"](AbsoluteErrorScore(), alpha=0.5)
    bad = TensorDict({"bad": torch.full((3,), torch.nan)}, batch_size=[])
    with pytest.raises(ValueError, match="Field 'bad': 3 non-finite"):
        calibrator.update(bad, bad)
    assert calibrator.n_cal == 0
