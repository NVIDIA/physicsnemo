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

"""Tests for dtype-aware numerical tolerances."""

import math

import pytest
import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
class TestSafeEps:
    """Verify safe_eps returns principled, dtype-aware floor values."""

    def test_matches_formula(self, dtype: torch.dtype) -> None:
        """safe_eps should equal tiny ** 0.25 for the given dtype."""
        expected = torch.finfo(dtype).tiny ** 0.25
        assert safe_eps(dtype) == expected

    def test_positive(self, dtype: torch.dtype) -> None:
        """safe_eps must be strictly positive."""
        assert safe_eps(dtype) > 0.0

    def test_reciprocal_does_not_overflow(self, dtype: torch.dtype) -> None:
        """1 / safe_eps must be representable (not inf)."""
        assert math.isfinite(1.0 / safe_eps(dtype))

    def test_reciprocal_squared_does_not_overflow(self, dtype: torch.dtype) -> None:
        """1 / safe_eps**2 must be representable (not inf).

        This matters for inverse-distance weights where the denominator
        may be squared before clamping.
        """
        assert math.isfinite(1.0 / safe_eps(dtype) ** 2)

    def test_smaller_than_machine_epsilon(self, dtype: torch.dtype) -> None:
        """safe_eps should be far below machine epsilon (it guards against
        exact zeros, not rounding errors)."""
        if dtype == torch.float16:
            pytest.skip(
                "float16 has a 5-bit exponent; tiny^0.25 exceeds eps, "
                "which is expected - the overflow-safety constraint dominates"
            )
        assert safe_eps(dtype) < torch.finfo(dtype).eps
