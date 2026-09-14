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

"""Tests for scatter_aggregate utility function.

Tests validate scatter-based aggregation operations used for transferring
data between mesh entities (points, cells, facets), including:
- Mean aggregation (weighted and unweighted)
- Sum aggregation
- Multi-dimensional data
- Dtype handling (integer promotion, complex preservation)
- Error handling
"""

import pytest
import torch

from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate


class TestScatterAggregateMean:
    """Tests for mean aggregation mode."""

    def test_mean_simple(self):
        """Test simple unweighted mean aggregation."""
        src_data = torch.tensor([[1.0], [2.0], [3.0]])
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="mean")

        # Point 0: mean of [1.0, 2.0] = 1.5
        # Point 1: mean of [3.0] = 3.0
        expected = torch.tensor([[1.5], [3.0]])
        assert torch.allclose(result, expected)

    def test_mean_scalar_data(self):
        """Test mean aggregation with scalar source data."""
        src_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        src_to_dst = torch.tensor([0, 0, 1, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="mean")

        # Point 0: mean of [1, 2] = 1.5
        # Point 1: mean of [3, 4] = 3.5
        expected = torch.tensor([1.5, 3.5])
        assert torch.allclose(result, expected)

    def test_mean_multiple_sources_per_destination(self):
        """Test mean with varying number of sources per destination."""
        src_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        src_to_dst = torch.tensor([0, 0, 0, 1, 2])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=3, aggregation="mean")

        # Point 0: mean of [1, 2, 3] = 2.0
        # Point 1: mean of [4] = 4.0
        # Point 2: mean of [5] = 5.0
        expected = torch.tensor([2.0, 4.0, 5.0])
        assert torch.allclose(result, expected)

    def test_mean_destination_with_no_sources(self):
        """Test mean when some destinations have no sources."""
        src_data = torch.tensor([1.0, 2.0])
        src_to_dst = torch.tensor([0, 2])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=3, aggregation="mean")

        # Point 0: mean of [1] = 1.0
        # Point 1: no sources, should be 0
        # Point 2: mean of [2] = 2.0
        expected = torch.tensor([1.0, 0.0, 2.0])
        assert torch.allclose(result, expected)


class TestScatterAggregateMeanWithWeights:
    """Tests for mean aggregation with explicit weights."""

    def test_mean_with_weights_simple(self):
        """Test simple weighted mean aggregation."""
        src_data = torch.tensor([1.0, 3.0])
        src_to_dst = torch.tensor([0, 0])
        weights = torch.tensor([1.0, 3.0])

        result = scatter_aggregate(
            src_data, src_to_dst, n_dst=1, weights=weights, aggregation="mean"
        )

        # Weighted mean: (1*1 + 3*3) / (1 + 3) = 10/4 = 2.5
        expected = torch.tensor([2.5])
        assert torch.allclose(result, expected)

    def test_mean_with_equal_weights_equals_unweighted(self):
        """Test that equal weights produce same result as unweighted mean."""
        src_data = torch.tensor([1.0, 2.0, 3.0])
        src_to_dst = torch.tensor([0, 0, 0])
        weights = torch.tensor([1.0, 1.0, 1.0])

        result_weighted = scatter_aggregate(
            src_data, src_to_dst, n_dst=1, weights=weights, aggregation="mean"
        )
        result_unweighted = scatter_aggregate(
            src_data, src_to_dst, n_dst=1, aggregation="mean"
        )

        assert torch.allclose(result_weighted, result_unweighted)

    def test_mean_with_zero_weight_ignored(self):
        """Test that zero-weighted sources are effectively ignored."""
        src_data = torch.tensor([1.0, 2.0, 100.0])  # 100 should be ignored
        src_to_dst = torch.tensor([0, 0, 0])
        weights = torch.tensor([1.0, 1.0, 0.0])  # Zero weight for 100

        result = scatter_aggregate(
            src_data, src_to_dst, n_dst=1, weights=weights, aggregation="mean"
        )

        # Weighted mean: (1*1 + 2*1 + 100*0) / (1 + 1 + 0) = 3/2 = 1.5
        expected = torch.tensor([1.5])
        assert torch.allclose(result, expected)


class TestScatterAggregateSum:
    """Tests for sum aggregation mode."""

    def test_sum_simple(self):
        """Test simple sum aggregation."""
        src_data = torch.tensor([1.0, 2.0, 3.0])
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="sum")

        # Point 0: sum of [1, 2] = 3.0
        # Point 1: sum of [3] = 3.0
        expected = torch.tensor([3.0, 3.0])
        assert torch.allclose(result, expected)

    def test_sum_with_weights(self):
        """Test weighted sum aggregation."""
        src_data = torch.tensor([1.0, 2.0])
        src_to_dst = torch.tensor([0, 0])
        weights = torch.tensor([3.0, 4.0])

        result = scatter_aggregate(
            src_data, src_to_dst, n_dst=1, weights=weights, aggregation="sum"
        )

        # Weighted sum: 1*3 + 2*4 = 11.0
        expected = torch.tensor([11.0])
        assert torch.allclose(result, expected)

    def test_sum_destination_with_no_sources(self):
        """Test sum when some destinations have no sources."""
        src_data = torch.tensor([1.0, 2.0])
        src_to_dst = torch.tensor([0, 2])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=3, aggregation="sum")

        # Point 0: sum of [1] = 1.0
        # Point 1: no sources, should be 0
        # Point 2: sum of [2] = 2.0
        expected = torch.tensor([1.0, 0.0, 2.0])
        assert torch.allclose(result, expected)


class TestScatterAggregateMultiDimensional:
    """Tests for multi-dimensional data aggregation."""

    def test_mean_2d_data(self):
        """Test mean aggregation with 2D source data."""
        src_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="mean")

        # Point 0: mean of [[1,2], [3,4]] = [2, 3]
        # Point 1: [5, 6]
        expected = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
        assert torch.allclose(result, expected)

    def test_mean_3d_data(self):
        """Test mean aggregation with 3D source data (e.g., tensors)."""
        src_data = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        src_to_dst = torch.tensor([0, 0])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1, aggregation="mean")

        expected = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
        assert torch.allclose(result, expected)

    def test_sum_vector_data(self):
        """Test sum aggregation with vector data."""
        src_data = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        src_to_dst = torch.tensor([0, 0, 0])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1, aggregation="sum")

        expected = torch.tensor([[1.0, 1.0, 1.0]])
        assert torch.allclose(result, expected)


class TestScatterAggregateErrors:
    """Tests for error handling."""

    def test_invalid_aggregation_raises(self):
        """Test that invalid aggregation mode raises ValueError."""
        src_data = torch.tensor([1.0, 2.0])
        src_to_dst = torch.tensor([0, 1])

        with pytest.raises(ValueError, match="Invalid aggregation"):
            scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="invalid")

    def test_invalid_aggregation_typo_raises(self):
        """Test that typos in aggregation mode raise ValueError."""
        src_data = torch.tensor([1.0, 2.0])
        src_to_dst = torch.tensor([0, 1])

        with pytest.raises(ValueError, match="Invalid aggregation"):
            scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="average")


class TestScatterAggregateDtypes:
    """Tests for dtype handling."""

    def test_float32_data(self):
        """Test aggregation with float32 data."""
        src_data = torch.tensor([1.0, 2.0], dtype=torch.float32)
        src_to_dst = torch.tensor([0, 0])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1)

        assert result.dtype == torch.float32

    def test_float64_data(self):
        """Test aggregation with float64 data."""
        src_data = torch.tensor([1.0, 2.0], dtype=torch.float64)
        src_to_dst = torch.tensor([0, 0])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1)

        assert result.dtype == torch.float64

    def test_weights_dtype_conversion(self):
        """Test that weights are converted to match data dtype."""
        src_data = torch.tensor([1.0, 2.0], dtype=torch.float64)
        src_to_dst = torch.tensor([0, 0])
        weights = torch.tensor([1.0, 1.0], dtype=torch.float32)  # Different dtype

        # Should not raise, weights should be converted
        result = scatter_aggregate(src_data, src_to_dst, n_dst=1, weights=weights)

        assert result.dtype == torch.float64

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64, torch.bool])
    def test_integer_mean_promotes_to_float64(self, int_dtype):
        """A "mean" of integer/bool data is promoted to float64.

        Computing the mean in the source integer dtype would truncate
        (e.g. ``(1 + 2) // 2 == 1``), and the division guard ``safe_eps`` cannot
        be evaluated on an integer dtype, so the aggregation promotes to float64.
        """
        src_data = torch.tensor([1, 2, 3], dtype=int_dtype)
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="mean")

        assert result.dtype == torch.float64
        if int_dtype != torch.bool:
            # Mean of [1, 2] is 1.5: it must not be truncated back to an integer.
            assert torch.allclose(result, torch.tensor([1.5, 3.0], dtype=torch.float64))

    def test_integer_sum_preserves_dtype(self):
        """A "sum" preserves the native integer dtype (no float promotion)."""
        src_data = torch.tensor([1, 2, 3], dtype=torch.int64)
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="sum")

        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor([3, 3], dtype=torch.int64))

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_mean_preserves_dtype_and_imaginary_part(self, dtype):
        """A "mean" of complex data keeps its dtype and its imaginary part.

        Complex tensors are not "floating point" by ``torch``'s definition, so
        they were previously promoted to float64 like an integer field, which
        silently discarded the imaginary part.
        """
        src_data = torch.tensor([1 + 2j, 3 + 6j, 5 - 4j], dtype=dtype)
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2)

        expected = torch.tensor([2 + 4j, 5 - 4j], dtype=dtype)
        assert result.dtype == dtype
        torch.testing.assert_close(result, expected)

    def test_complex_multidimensional_mean(self):
        """A "mean" of multi-dimensional complex data averages componentwise."""
        src_data = torch.tensor(
            [
                [1 + 1j, 2 + 2j, 3 + 3j],
                [3 + 5j, 4 + 6j, 5 + 7j],
                [8 - 1j, 8 - 2j, 8 - 3j],
            ],
            dtype=torch.complex64,
        )
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2)

        # Destination 0 averages rows 0 and 1; destination 1 gets row 2 alone.
        expected = torch.tensor(
            [[2 + 3j, 3 + 4j, 4 + 5j], [8 - 1j, 8 - 2j, 8 - 3j]],
            dtype=torch.complex64,
        )
        assert result.dtype == torch.complex64
        torch.testing.assert_close(result, expected)

    def test_complex_weighted_mean_uses_real_weights(self):
        """Real weights normalize a complex "mean" without touching the phase."""
        src_data = torch.tensor([1 + 2j, 3 + 6j], dtype=torch.complex64)
        src_to_dst = torch.tensor([0, 0])
        weights = torch.tensor([1.0, 3.0])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1, weights=weights)

        # (1*(1+2j) + 3*(3+6j)) / (1 + 3) = (10 + 20j) / 4
        torch.testing.assert_close(result, torch.tensor([2.5 + 5j]))

    def test_complex_sum_fast_path_preserves_imaginary_part(self):
        """The unweighted-"sum" fast path passes complex data straight through."""
        src_data = torch.tensor([1 + 2j, 3 + 6j, 5 - 4j], dtype=torch.complex64)
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data, src_to_dst, n_dst=2, aggregation="sum")

        assert result.dtype == torch.complex64
        torch.testing.assert_close(result, torch.tensor([4 + 8j, 5 - 4j]))

    def test_complex_weighted_sum_preserves_imaginary_part(self):
        """A real-weighted "sum" of complex data accumulates in complex."""
        src_data = torch.tensor([1 + 2j, 3 + 6j], dtype=torch.complex64)
        src_to_dst = torch.tensor([0, 0])

        result = scatter_aggregate(
            src_data,
            src_to_dst,
            n_dst=1,
            weights=torch.tensor([2.0, 0.5]),
            aggregation="sum",
        )

        torch.testing.assert_close(result, torch.tensor([3.5 + 7j]))

    def test_complex_weights_accepted_for_sum(self):
        """A "sum" accepts complex weights: it has no divisor to guard.

        Phasor weighting is well defined, so complex weights must not be
        rejected, and the accumulator is promoted so the product keeps its
        imaginary part even when the values themselves are real.
        """
        src_to_dst = torch.tensor([0, 0])
        weights = torch.tensor([1j, 2 + 0j])

        ### Complex values with complex weights.
        complex_result = scatter_aggregate(
            torch.tensor([1 + 2j, 3 + 6j], dtype=torch.complex64),
            src_to_dst,
            n_dst=1,
            weights=weights,
            aggregation="sum",
        )
        # 1j*(1+2j) + 2*(3+6j) = (-2+1j) + (6+12j) = 4+13j
        torch.testing.assert_close(complex_result, torch.tensor([4 + 13j]))

        ### Real values with complex weights promote to complex.
        promoted_result = scatter_aggregate(
            torch.tensor([1.0, 3.0]),
            src_to_dst,
            n_dst=1,
            weights=weights,
            aggregation="sum",
        )
        # 1j*1 + 2*3 = 6+1j
        assert promoted_result.dtype == torch.complex64
        torch.testing.assert_close(promoted_result, torch.tensor([6 + 1j]))

    def test_complex_weights_rejected_for_mean(self):
        """A "mean" rejects complex weights, because its divisor is clamped.

        The per-destination weight sums are clamped away from zero to guard the
        division, and ``torch.clamp`` rejects complex dtypes. Silently dropping
        the imaginary part of the weights would give a wrong answer, so this is
        an error rather than a promotion.
        """
        with pytest.raises(TypeError, match="weights must be real-valued"):
            scatter_aggregate(
                torch.tensor([1 + 2j, 3 + 6j]),
                torch.tensor([0, 0]),
                n_dst=1,
                weights=torch.tensor([1j, 2 + 0j]),
                aggregation="mean",
            )


class TestScatterAggregateDevices:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test aggregation on CPU."""
        src_data = torch.tensor([1.0, 2.0], device="cpu")
        src_to_dst = torch.tensor([0, 0], device="cpu")

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1)

        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test aggregation on CUDA."""
        src_data = torch.tensor([1.0, 2.0], device="cuda")
        src_to_dst = torch.tensor([0, 0], device="cuda")

        result = scatter_aggregate(src_data, src_to_dst, n_dst=1)

        assert result.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_cuda_complex_mean(self, dtype):
        """Complex aggregation on CUDA matches the CPU result."""
        src_data = torch.tensor([1 + 2j, 3 + 6j, 5 - 4j], dtype=dtype)
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(src_data.cuda(), src_to_dst.cuda(), n_dst=2)

        assert result.device.type == "cuda"
        assert result.dtype == dtype
        expected = scatter_aggregate(src_data, src_to_dst, n_dst=2)
        torch.testing.assert_close(result.cpu(), expected)


class TestScatterAggregateParametrized:
    """Parametrized tests for scatter_aggregate."""

    @pytest.mark.parametrize("aggregation", ["mean", "sum"])
    def test_all_aggregation_modes(self, aggregation):
        """Test that all aggregation modes work."""
        src_data = torch.tensor([1.0, 2.0, 3.0])
        src_to_dst = torch.tensor([0, 0, 1])

        result = scatter_aggregate(
            src_data, src_to_dst, n_dst=2, aggregation=aggregation
        )

        assert result.shape == (2,)

    @pytest.mark.parametrize("n_dst", [1, 2, 5, 10])
    def test_various_n_dst(self, n_dst):
        """Test with various destination counts."""
        src_data = torch.randn(20)
        src_to_dst = torch.randint(0, n_dst, (20,))

        result = scatter_aggregate(
            src_data, src_to_dst, n_dst=n_dst, aggregation="mean"
        )

        assert result.shape == (n_dst,)

    @pytest.mark.parametrize(
        "data_shape",
        [
            (10,),
            (10, 3),
            (10, 3, 3),
            (10, 2, 4, 6),
        ],
    )
    def test_various_data_shapes(self, data_shape):
        """Test with various data shapes."""
        torch.manual_seed(42)
        src_data = torch.randn(data_shape)
        src_to_dst = torch.randint(0, 3, (data_shape[0],))

        result = scatter_aggregate(src_data, src_to_dst, n_dst=3, aggregation="mean")

        expected_shape = (3,) + data_shape[1:]
        assert result.shape == expected_shape
