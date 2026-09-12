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

"""Tests for vector normalization across dtypes and scales."""

import inspect

import pytest
import torch

from benchmarks.physicsnemo.nn.functional._spec_utils import build_benchmark_plan
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.nn.functional import safe_normalize
from physicsnemo.nn.functional.normalization import SafeNormalize

_ALL_DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]


def _assert_unit_length(vectors: torch.Tensor, dtype: torch.dtype) -> None:
    """Assert every row is unit length, to within the dtype's own precision."""
    norms = vectors.to(torch.float64).norm(dim=-1)
    torch.testing.assert_close(
        norms, torch.ones_like(norms), rtol=0.0, atol=8 * torch.finfo(dtype).eps
    )


# Magnitudes whose squares underflow. float16 norms accumulate in float32,
# whose exponent range spans all representable float16 values.
_UNDERFLOWING_MAGNITUDE = {
    torch.bfloat16: 1e-30,
    torch.float32: 1e-30,
    torch.float64: 1e-200,
}


class TestSafeNormalize:
    """Verify zero handling, unit length, and dtype preservation."""

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
        """Vectors whose raw norm underflows retain unit length and direction."""
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
        """Vectors whose raw norm overflows retain unit length."""
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
        """Finite input produces finite output across representable scales."""
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

    @pytest.mark.parametrize("dtype", _ALL_DTYPES)
    @pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("implementation", [None, "torch"])
    def test_autocast_preserves_dtype_and_values(
        self, device, dtype: torch.dtype, autocast_dtype: torch.dtype, implementation
    ) -> None:
        """Autocast preserves the input dtype, zero rows, and unit vectors."""
        info = torch.finfo(dtype)
        vectors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [info.tiny, -info.tiny, info.tiny],
                [info.max * 0.6] * 3,
            ],
            dtype=dtype,
            device=device,
        )

        with torch.autocast(torch.device(device).type, dtype=autocast_dtype):
            result = safe_normalize(vectors, dim=-1, implementation=implementation)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.8, 0.0],
                [3**-0.5, -(3**-0.5), 3**-0.5],
                [3**-0.5] * 3,
            ],
            dtype=dtype,
            device=device,
        )
        torch.testing.assert_close(result, expected, rtol=0.0, atol=8 * info.eps)
        assert torch.equal(result[0], torch.zeros_like(result[0]))
        _assert_unit_length(result[1:], dtype)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_normalizes_selected_dimension(self, device, dim: int) -> None:
        """Normalize the requested axis of a noncontiguous batched tensor."""
        vectors = (
            torch.arange(1, 25, dtype=torch.float64, device=device)
            .reshape(2, 3, 4)
            .transpose(0, 1)
        )
        expected = vectors / vectors.norm(dim=dim, keepdim=True)

        result = safe_normalize(vectors, dim=dim)

        torch.testing.assert_close(result, expected)

    def test_public_signature(self) -> None:
        """Keep the required dimension and expose keyword-only backend selection."""
        parameters = inspect.signature(safe_normalize).parameters

        assert list(parameters) == ["vectors", "dim", "implementation"]
        assert parameters["dim"].default is inspect.Parameter.empty
        assert parameters["implementation"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["implementation"].default is None

    @pytest.mark.parametrize("implementation", [None, "torch"])
    @pytest.mark.parametrize("compiled", [False, True])
    def test_forward_and_backward(self, device, implementation, compiled) -> None:
        """Eager and compiled dispatch preserve values and analytical gradients."""
        vectors = torch.tensor(
            [[3.0, 4.0, 0.0], [-2.0, 1.0, 5.0]],
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        normalize = (
            torch.compile(safe_normalize, fullgraph=True)
            if compiled
            else safe_normalize
        )

        result = normalize(vectors, dim=-1, implementation=implementation)
        result.sum().backward()

        lengths = vectors.detach().norm(dim=-1, keepdim=True)
        expected = vectors.detach() / lengths
        expected_gradient = (
            1 - expected * expected.sum(dim=-1, keepdim=True)
        ) / lengths
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(vectors.grad, expected_gradient)

    def test_benchmark_registration(self) -> None:
        """The shared benchmark runner discovers both normalization phases."""
        assert SafeNormalize in FUNCTIONAL_SPECS
        keys, specs = build_benchmark_plan(
            device="cpu",
            phases=("forward", "backward"),
            selected_specs=(SafeNormalize,),
        )

        assert {key[0] for key in keys} == {"forward", "backward"}
        assert {key[2] for key in keys} == {"torch"}
        assert all(spec is SafeNormalize for spec in specs.values())

    @pytest.mark.parametrize("backward", [False, True])
    def test_benchmark_inputs(self, device, backward) -> None:
        """Each benchmark workload runs with unit outputs and finite gradients."""
        make_inputs = (
            SafeNormalize.make_inputs_backward
            if backward
            else SafeNormalize.make_inputs_forward
        )
        for _, (vectors,), kwargs in make_inputs(device=device):
            assert vectors.requires_grad == backward
            result = SafeNormalize.dispatch(vectors, implementation="torch", **kwargs)
            assert result.device == torch.device(device)
            torch.testing.assert_close(
                result.norm(dim=kwargs["dim"]),
                torch.ones_like(result[..., 0]),
            )
            if backward:
                result.sum().backward()
                assert vectors.grad is not None
                assert torch.isfinite(vectors.grad).all()
