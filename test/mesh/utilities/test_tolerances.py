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

import pytest
import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
)
class TestSafeEps:
    """Verify safe_eps returns principled, dtype-aware floor values."""

    def test_matches_formula(self, dtype: torch.dtype) -> None:
        """safe_eps should equal min(tiny ** 0.25, machine_eps)."""
        info = torch.finfo(dtype)
        expected = min(info.tiny**0.25, info.eps)
        assert safe_eps(dtype) == expected

    def test_positive(self, dtype: torch.dtype) -> None:
        """safe_eps must be strictly positive."""
        assert safe_eps(dtype) > 0.0

    def test_reciprocal_does_not_overflow(self, dtype: torch.dtype) -> None:
        """1 / safe_eps must not overflow in the dtype's own arithmetic."""
        eps_tensor = torch.tensor(safe_eps(dtype), dtype=dtype)
        assert torch.isfinite(1.0 / eps_tensor)

    def test_reciprocal_squared_does_not_overflow(self, dtype: torch.dtype) -> None:
        """1 / safe_eps**2 must not overflow for wide-exponent types.

        Float16's 5-bit exponent cannot satisfy both 'small eps' and
        '1/eps^2 fits' simultaneously; the cap at machine epsilon
        prioritizes keeping the clamp floor small.
        """
        if dtype == torch.float16:
            pytest.skip("float16 trades 1/eps^2 safety for a usable clamp floor")
        eps_tensor = torch.tensor(safe_eps(dtype), dtype=dtype)
        assert torch.isfinite(1.0 / eps_tensor**2)

    def test_at_most_machine_epsilon(self, dtype: torch.dtype) -> None:
        """safe_eps must not exceed machine epsilon, so it never corrupts
        values that are numerically meaningful in the dtype."""
        assert safe_eps(dtype) <= torch.finfo(dtype).eps
