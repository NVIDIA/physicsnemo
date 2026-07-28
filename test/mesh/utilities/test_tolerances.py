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

from physicsnemo.mesh.utilities._tolerances import safe_eps, safe_normalize

_ALL_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]


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


def _assert_unit_length(vectors: torch.Tensor, dtype: torch.dtype) -> None:
    """Assert every row is unit length, to within the dtype's own precision."""
    norms = vectors.to(torch.float64).norm(dim=-1)
    torch.testing.assert_close(
        norms, torch.ones_like(norms), rtol=0.0, atol=8 * torch.finfo(dtype).eps
    )


### Component magnitudes small enough that squaring them underflows to zero, so
### the raw norm evaluates to 0 and any implementation that divides by it --
### clamped like ``F.normalize`` or masked -- returns a non-unit vector.
### float16 has no such regime: its norm accumulates in float32, whose range
### comfortably spans all of float16.
_UNDERFLOWING_MAGNITUDE = {
    torch.bfloat16: 1e-30,
    torch.float32: 1e-30,
    torch.float64: 1e-200,
}


class TestSafeNormalize:
    """Verify safe_normalize stays exact at both ends of each dtype's range.

    ``torch.nn.functional.normalize`` clamps the norm at a hardcoded
    ``eps=1e-12``, which fails three separate ways; one test below pins each.
    Two of those failures survive the tempting simplification of dropping the
    max-abs rescale and masking only exact zeros, so these tests are what stop
    that rescale from being optimized back out.
    """

    @pytest.mark.parametrize("dtype", _ALL_DTYPES)
    def test_exact_zero_row_normalizes_to_zero(self, dtype: torch.dtype) -> None:
        """Degenerate rows give zero, not the NaN that 0/0 produces in fp16."""
        vectors = torch.zeros(3, 3, dtype=dtype)
        vectors[1] = torch.tensor([0.0, 2.0, 0.0], dtype=dtype)

        result = safe_normalize(vectors, dim=-1)

        assert result.isfinite().all()
        assert torch.equal(result[0], torch.zeros(3, dtype=dtype))
        assert torch.equal(result[2], torch.zeros(3, dtype=dtype))
        ### A degenerate neighbour must not perturb a healthy row.
        torch.testing.assert_close(
            result[1], torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        )

    @pytest.mark.parametrize("dtype", list(_UNDERFLOWING_MAGNITUDE))
    def test_underflowing_magnitude_stays_unit_length(self, dtype: torch.dtype) -> None:
        """Vectors too small to square keep unit length and direction.

        This is the nanoscale-geometry case the module exists to protect, and
        it is the test that a mask-only implementation fails: with the raw norm
        underflowing to zero, masking leaves the vector at its original tiny
        magnitude instead of rescaling it.
        """
        magnitude = _UNDERFLOWING_MAGNITUDE[dtype]
        vectors = torch.tensor([[3.0 * magnitude, 4.0 * magnitude, 0.0]], dtype=dtype)
        assert vectors.norm(dim=-1).item() == 0.0, (
            f"{magnitude=} no longer underflows in {dtype}; this test has lost "
            f"its power to discriminate against a mask-only implementation"
        )

        result = safe_normalize(vectors, dim=-1)

        _assert_unit_length(result, dtype)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.6, 0.8, 0.0]], dtype=dtype),
            rtol=0.0,
            atol=8 * torch.finfo(dtype).eps,
        )

    @pytest.mark.parametrize("dtype", _ALL_DTYPES)
    def test_overflowing_magnitude_stays_unit_length(self, dtype: torch.dtype) -> None:
        """Vectors whose squared norm overflows keep unit length.

        Every component is representable, but their sum of squares is not, so
        dividing by the raw norm means dividing by ``inf`` -- silently turning
        a well-conditioned cell into a zero normal. Both ``F.normalize`` and a
        mask-only implementation fail this in every dtype.
        """
        vectors = torch.full((1, 3), torch.finfo(dtype).max * 0.6, dtype=dtype)
        assert vectors.isfinite().all(), "components must themselves be representable"
        assert vectors.norm(dim=-1).isinf().all(), (
            f"the squared norm no longer overflows in {dtype}; this test has "
            f"lost its power to discriminate"
        )

        result = safe_normalize(vectors, dim=-1)

        _assert_unit_length(result, dtype)

    @pytest.mark.parametrize("dtype", _ALL_DTYPES)
    def test_finite_input_gives_finite_output(self, dtype: torch.dtype) -> None:
        """The contract that makes this safe: finite in, finite out.

        Rescaling bounds the norm below by one, so no finite input can divide
        by zero or by infinity anywhere in the dtype's range.
        """
        info = torch.finfo(dtype)
        magnitudes = [0.0, info.tiny, info.eps, 1.0, 1.0 / info.eps, info.max * 0.6]
        vectors = torch.tensor([[m, -m, m] for m in magnitudes], dtype=dtype)

        result = safe_normalize(vectors, dim=-1)

        assert result.isfinite().all(), f"non-finite output for {magnitudes=}"

    @pytest.mark.parametrize("dtype", _ALL_DTYPES)
    def test_well_conditioned_input_matches_float64_reference(
        self, dtype: torch.dtype
    ) -> None:
        """Ordinary vectors are unaffected: guarding zeros must not cost accuracy."""
        generator = torch.Generator().manual_seed(0)
        reference = torch.randn(256, 3, generator=generator, dtype=torch.float64)
        expected = reference / reference.norm(dim=-1, keepdim=True)

        result = safe_normalize(reference.to(dtype), dim=-1)

        torch.testing.assert_close(
            result.to(torch.float64),
            expected,
            rtol=0.0,
            atol=8 * torch.finfo(dtype).eps,
        )

    def test_zero_size_component_dim_is_returned_unchanged(self) -> None:
        """A zero-size reduction dim short-circuits, since ``amax`` rejects it."""
        vectors = torch.zeros(4, 0)

        result = safe_normalize(vectors, dim=-1)

        assert result.shape == (4, 0)
