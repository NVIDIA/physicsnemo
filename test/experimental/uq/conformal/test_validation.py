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

"""Shared input-contract validators, tested once, directly."""

import pytest
import torch

from physicsnemo.experimental.uq.conformal._utils import positive_finite_float

_NOT_POSITIVE = (ValueError, "positive finite")

# fmt: off
# (id, thunk, expected): expected is (ExceptionType, match) or a return value.
VALIDATOR_CASES = [
    ("eps-nan", lambda: positive_finite_float(float("nan"), "eps"), _NOT_POSITIVE),
    ("eps-inf", lambda: positive_finite_float(float("inf"), "eps"), _NOT_POSITIVE),
    ("eps-zero", lambda: positive_finite_float(0.0, "eps"), _NOT_POSITIVE),
    ("eps-none", lambda: positive_finite_float(None, "eps"), (ValueError, "positive finite value")),
    ("eps-string", lambda: positive_finite_float("not a number", "eps"), (ValueError, "positive finite value")),
    ("eps-tensor-coerced", lambda: type(positive_finite_float(torch.tensor(1e-3), "eps")) is float, True),
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
