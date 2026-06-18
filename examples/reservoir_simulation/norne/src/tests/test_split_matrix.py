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

"""
Smoke tests for Split_Matrix, which splits a 2-D array along axis 1
into equal-width blocks and stacks them into a 3-D array.

The function appears in several files with the same logic; tests run
against an inline copy so no heavyweight imports are needed.
"""

import numpy as np


def split_matrix(matrix, sizee):
    """Mirror of Split_Matrix from compare/sequential/misc_forward_enact.py."""
    x_prime = [matrix[:, i : i + sizee] for i in range(0, matrix.shape[1], sizee)]
    return np.array(x_prime)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_split_matrix_output_shape_exact_divisor():
    """Test split matrix output shape exact divisor."""
    matrix = np.ones((4, 12))
    result = split_matrix(matrix, sizee=4)
    assert result.shape == (3, 4, 4)


def test_split_matrix_output_shape_single_column_block():
    """Test split matrix output shape single column block."""
    matrix = np.ones((5, 5))
    result = split_matrix(matrix, sizee=5)
    assert result.shape == (1, 5, 5)


def test_split_matrix_output_shape_many_blocks():
    """Test split matrix output shape many blocks."""
    matrix = np.arange(30).reshape(3, 10)
    result = split_matrix(matrix, sizee=2)
    assert result.shape == (5, 3, 2)


# ---------------------------------------------------------------------------
# Value tests
# ---------------------------------------------------------------------------


def test_split_matrix_preserves_values():
    """Test split matrix preserves values."""
    matrix = np.arange(8).reshape(2, 4)
    result = split_matrix(matrix, sizee=2)
    # First block: columns 0-1
    assert np.array_equal(result[0], matrix[:, :2])
    # Second block: columns 2-3
    assert np.array_equal(result[1], matrix[:, 2:])


def test_split_matrix_reconstruct():
    """Concatenating blocks back along axis 2 must equal the original."""
    matrix = np.random.rand(6, 20)
    sizee = 4
    result = split_matrix(matrix, sizee=sizee)
    reconstructed = np.concatenate(list(result), axis=1)
    assert np.allclose(reconstructed, matrix)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_split_matrix_single_row():
    """Test split matrix single row."""
    matrix = np.arange(6).reshape(1, 6)
    result = split_matrix(matrix, sizee=3)
    assert result.shape == (2, 1, 3)


def test_split_matrix_float_values():
    """Test split matrix float values."""
    matrix = np.random.randn(8, 8)
    result = split_matrix(matrix, sizee=4)
    assert result.dtype == matrix.dtype
