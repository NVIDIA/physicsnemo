# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
Test suite for domino utils module.

This test file duplicates all the docstring examples from the domino utils
module to ensure that the documented examples work correctly.
"""

import numpy as np

from physicsnemo.utils.domino.utils import (
    area_weighted_shuffle_array,
    calculate_center_of_mass,
    calculate_normal_positional_encoding,
    calculate_pos_encoding,
    combine_dict,
    create_grid,
    mean_std_sampling,
    nd_interpolator,
    normalize,
    pad,
    pad_inp,
    shuffle_array,
    shuffle_array_without_sampling,
    standardize,
    unnormalize,
    unstandardize,
)


def test_calculate_center_of_mass():
    """Test calculate_center_of_mass function with docstring example."""
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    sizes = np.array([1.0, 2.0, 3.0])
    com = calculate_center_of_mass(centers, sizes)
    expected = np.array([[4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]])
    assert np.allclose(com, expected)


def test_normalize():
    """Test normalize function with docstring examples."""
    # Example 1: With explicit min/max
    field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize(field, 5.0, 1.0)
    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.allclose(normalized, expected)

    # Example 2: Auto-compute min/max
    normalized_auto = normalize(field)
    expected_auto = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.allclose(normalized_auto, expected_auto)


def test_unnormalize():
    """Test unnormalize function with docstring example."""
    normalized = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    original = unnormalize(normalized, 5.0, 1.0)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(original, expected)


def test_standardize():
    """Test standardize function with docstring examples."""
    # Example 1: With explicit mean/std
    field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    standardized = standardize(field, 3.0, np.sqrt(2.5))
    expected = np.array([-1.265, -0.632, 0.0, 0.632, 1.265])
    assert np.allclose(standardized, expected, atol=1e-3)

    # Example 2: Auto-compute mean/std
    standardized_auto = standardize(field)
    assert np.allclose(np.mean(standardized_auto), 0.0)
    assert np.allclose(np.std(standardized_auto, ddof=0), 1.0)


def test_unstandardize():
    """Test unstandardize function with docstring example."""
    standardized = np.array([-1.265, -0.632, 0.0, 0.632, 1.265])
    original = unstandardize(standardized, 3.0, np.sqrt(2.5))
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(original, expected, atol=1e-3)


def test_calculate_normal_positional_encoding():
    """Test calculate_normal_positional_encoding function with docstring examples."""
    # Example 1: Basic coordinates
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell_size = [0.1, 0.1, 0.1]
    encoding = calculate_normal_positional_encoding(coords, cell_dimensions=cell_size)
    assert encoding.shape == (2, 12)

    # Example 2: Relative positioning
    coords_b = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    encoding_rel = calculate_normal_positional_encoding(coords, coords_b, cell_size)
    assert encoding_rel.shape == (2, 12)


def test_nd_interpolator():
    """Test nd_interpolator function with docstring example."""
    # Simple 2D interpolation example
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    field_vals = np.array([[1.0], [2.0], [3.0], [4.0]])
    grid_points = np.array([[0.5, 0.5]])
    result = nd_interpolator([coords], field_vals, grid_points)
    assert result.shape[0] == 1  # One grid point


def test_pad():
    """Test pad function with docstring examples."""
    # Example 1: Padding needed
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    padded = pad(arr, 4, -1.0)
    assert padded.shape == (4, 2)
    assert np.array_equal(padded[:2], arr)
    assert bool(np.all(padded[2:] == -1.0))

    # Example 2: No padding needed
    same = pad(arr, 2)
    assert np.array_equal(same, arr)


def test_pad_inp():
    """Test pad_inp function with docstring example."""
    arr = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    padded = pad_inp(arr, 4, 0.0)
    assert padded.shape == (4, 1, 2)
    assert np.array_equal(padded[:2], arr)
    assert bool(np.all(padded[2:] == 0.0))


def test_shuffle_array():
    """Test shuffle_array function with docstring example."""
    np.random.seed(42)  # For reproducible results
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    subset, indices = shuffle_array(data, 2)
    assert subset.shape == (2, 2)
    assert indices.shape == (2,)
    assert len(np.unique(indices)) == 2  # No duplicates


def test_shuffle_array_without_sampling():
    """Test shuffle_array_without_sampling function with docstring example."""
    np.random.seed(42)  # For reproducible results
    data = np.array([[1], [2], [3], [4]])
    shuffled, indices = shuffle_array_without_sampling(data)
    assert shuffled.shape == (4, 1)
    assert indices.shape == (4,)
    assert set(indices) == set(range(4))  # All original indices present


def test_calculate_pos_encoding():
    """Test calculate_pos_encoding function with docstring example."""
    positions = np.array([0.0, 1.0, 2.0])
    encodings = calculate_pos_encoding(positions, d=4)
    assert len(encodings) == 4
    assert all(enc.shape == (3,) for enc in encodings)


def test_combine_dict():
    """Test combine_dict function with docstring example."""
    stats1 = {"loss": 0.5, "accuracy": 0.8}
    stats2 = {"loss": 0.3, "accuracy": 0.1}
    combined = combine_dict(stats1, stats2)
    assert combined["loss"] == 0.8
    assert combined["accuracy"] == 0.9


def test_create_grid():
    """Test create_grid function with docstring example."""
    min_bounds = np.array([0.0, 0.0, 0.0])
    max_bounds = np.array([1.0, 1.0, 1.0])
    grid_res = np.array([2, 2, 2])
    grid = create_grid(max_bounds, min_bounds, grid_res)
    assert grid.shape == (2, 2, 2, 3)
    assert np.allclose(grid[0, 0, 0], [0.0, 0.0, 0.0])
    assert np.allclose(grid[1, 1, 1], [1.0, 1.0, 1.0])


def test_mean_std_sampling():
    """Test mean_std_sampling function with docstring example."""
    # Create test data with outliers
    field = np.array([[1.0], [2.0], [3.0], [10.0]])  # 10.0 is outlier
    field_mean = np.array([2.0])
    field_std = np.array([1.0])
    outliers = mean_std_sampling(field, field_mean, field_std, 2.0)
    assert 3 in outliers  # Index 3 (value 10.0) should be detected as outlier


def test_area_weighted_shuffle_array():
    """Test area_weighted_shuffle_array function with docstring example."""
    np.random.seed(42)  # For reproducible results
    mesh_data = np.array([[1.0], [2.0], [3.0], [4.0]])
    cell_areas = np.array([0.1, 0.1, 0.1, 10.0])  # Last point has much larger area
    subset, indices = area_weighted_shuffle_array(mesh_data, 2, cell_areas)
    assert subset.shape == (2, 1)
    assert indices.shape == (2,)
    # The point with large area (index 3) should likely be selected
    assert len(set(indices)) <= 2  # At most 2 unique indices

    # Use higher area_factor for stronger bias toward large areas
    subset_biased, _ = area_weighted_shuffle_array(
        mesh_data, 2, cell_areas, area_factor=2.0
    )
    assert subset_biased.shape == (2, 1)
