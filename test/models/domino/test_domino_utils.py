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
Test suite for domino utils module.

This test file duplicates all the docstring examples from the domino utils
module to ensure that the documented examples work correctly.
"""

import math

import pytest
import torch

from physicsnemo.models.domino.utils import (
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
    centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    sizes = torch.tensor([1.0, 2.0, 3.0])
    com = calculate_center_of_mass(centers, sizes)
    expected = torch.tensor([[4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0]])
    assert torch.allclose(com, expected)


def test_normalize():
    """Test normalize function with docstring examples."""
    # Example 1: With explicit min/max
    field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize(field, max_val=5.0, min_val=1.0)
    expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert torch.allclose(normalized, expected)

    # Example 2: Auto-compute min/max
    normalized_auto = normalize(field)
    expected_auto = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert torch.allclose(normalized_auto, expected_auto)


def test_unnormalize():
    """Test unnormalize function with docstring example."""
    normalized = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    original = unnormalize(normalized, 5.0, 1.0)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(original, expected)


def test_standardize():
    """Test standardize function with docstring examples."""
    # Example 1: With explicit mean/std
    field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    standardized = standardize(field, 3.0, math.sqrt(2.5))
    expected = torch.tensor([-1.265, -0.632, 0.0, 0.632, 1.265])
    assert torch.allclose(standardized, expected, atol=1e-3)

    # Example 2: Auto-compute mean/std
    standardized_auto = standardize(field)
    assert torch.allclose(torch.mean(standardized_auto), torch.tensor(0.0))
    assert torch.allclose(torch.std(standardized_auto, correction=1), torch.tensor(1.0))


def test_unstandardize():
    """Test unstandardize function with docstring example."""
    standardized = torch.tensor([-1.265, -0.632, 0.0, 0.632, 1.265])
    original = unstandardize(standardized, 3.0, math.sqrt(2.5))
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(original, expected, atol=1e-3)


@pytest.mark.parametrize("relative", [True, False])
def test_calculate_normal_positional_encoding(relative):
    """Test calculate_normal_positional_encoding function with docstring examples."""
    # Example 1: Basic coordinates
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell_size = [0.1, 0.1, 0.1]

    # Example 2: Relative positioning
    if relative:
        coords_b = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    else:
        coords_b = None

    encoding_rel = calculate_normal_positional_encoding(coords, coords_b, cell_size)
    assert encoding_rel.shape == (2, 12)


def test_nd_interpolator():
    """Test nd_interpolator function with docstring example."""
    # Simple 2D interpolation example
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    field_vals = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    grid_points = torch.tensor([[0.5, 0.5]])
    result = nd_interpolator(coords, field_vals, grid_points)
    assert result.shape[0] == 1  # One grid point


def test_pad():
    """Test pad function with docstring examples."""
    # Example 1: Padding needed
    arr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded = pad(arr, 4, -1.0)
    assert padded.shape == (4, 2)
    assert torch.allclose(padded[:2], arr)
    assert bool(torch.all(padded[2:] == -1.0))

    # Example 2: No padding needed
    same = pad(arr, 2)
    assert torch.allclose(same, arr)


def test_pad_inp():
    """Test pad_inp function with docstring example."""
    arr = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    padded = pad_inp(arr, 4, 0.0)
    assert padded.shape == (4, 1, 2)
    assert torch.allclose(padded[:2], arr)
    assert bool(torch.all(padded[2:] == 0.0))


def test_shuffle_array():
    """Test shuffle_array function with docstring example."""
    torch.manual_seed(42)  # For reproducible results
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    subset, indices = shuffle_array(data, 2)
    assert subset.shape == (2, 2)
    assert indices.shape == (2,)
    assert len(torch.unique(indices)) == 2  # No duplicates


def test_shuffle_array_without_sampling():
    """Test shuffle_array_without_sampling function with docstring example."""
    torch.manual_seed(42)  # For reproducible results
    data = torch.tensor([[1], [2], [3], [4]])
    shuffled, indices = shuffle_array_without_sampling(data)
    assert shuffled.shape == (4, 1)
    assert indices.shape == (4,)
    assert set(indices.tolist()) == set(range(4))  # All original indices present


def test_calculate_pos_encoding():
    """Test calculate_pos_encoding function with docstring example."""
    positions = torch.tensor([0.0, 1.0, 2.0])
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
    min_bounds = torch.tensor([0.0, 0.0, 0.0])
    max_bounds = torch.tensor([1.0, 1.0, 1.0])
    grid_res = torch.tensor([2, 2, 2])
    grid = create_grid(max_bounds, min_bounds, grid_res)
    assert grid.shape == (2, 2, 2, 3)
    assert torch.allclose(grid[0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(grid[1, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_mean_std_sampling():
    """Test mean_std_sampling function with docstring example."""
    # Create test data with outliers
    field = torch.tensor([[1.0], [2.0], [3.0], [10.0]])  # 10.0 is outlier
    field_mean = torch.tensor([2.0])
    field_std = torch.tensor([1.0])
    outliers = mean_std_sampling(field, field_mean, field_std, 2.0)
    assert 3 in outliers  # Index 3 (value 10.0) should be detected as outlier


def test_area_weighted_shuffle_array():
    """Test area_weighted_shuffle_array function with docstring example."""
    torch.manual_seed(42)  # For reproducible results
    mesh_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    cell_areas = torch.tensor([0.1, 0.1, 0.1, 10.0])  # Last point has much larger area
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
