# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for FieldSlice transform."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms import FieldSlice


def test_index_selection_last_dim():
    """Test selecting specific indices from the last dimension."""
    n_points = 100
    n_features = 10
    selected_indices = [0, 2, 5]

    # Create sample
    features = torch.randn(n_points, n_features)
    sample = TensorDict({"features": features, "coords": torch.randn(n_points, 3)})

    # Create transform
    transform = FieldSlice({"features": {-1: selected_indices}})

    # Apply
    result = transform(sample)

    # Check shape
    assert result["features"].shape == (n_points, len(selected_indices))

    # Check values
    expected = features[:, selected_indices]
    assert torch.allclose(result["features"], expected)

    # Unchanged field
    assert result["coords"].shape == (n_points, 3)


def test_index_selection_first_dim():
    """Test selecting specific indices from the first dimension."""
    data = torch.randn(10, 8, 6)
    sample = TensorDict({"data": data})

    transform = FieldSlice({"data": {0: [1, 3, 5]}})
    result = transform(sample)

    assert result["data"].shape == (3, 8, 6)
    expected = data[[1, 3, 5], :, :]
    assert torch.allclose(result["data"], expected)


def test_slice_selection():
    """Test selecting a slice (start:stop:step)."""
    data = torch.randn(100, 10)
    sample = TensorDict({"data": data})

    # Select first 5 elements of last dimension
    transform = FieldSlice({"data": {-1: {"start": 0, "stop": 5}}})
    result = transform(sample)

    assert result["data"].shape == (100, 5)
    expected = data[:, 0:5]
    assert torch.allclose(result["data"], expected)


def test_slice_with_step():
    """Test slice with step."""
    data = torch.randn(100, 10)
    sample = TensorDict({"data": data})

    # Select every other element: [0, 2, 4, 6, 8]
    transform = FieldSlice({"data": {-1: {"start": 0, "stop": 10, "step": 2}}})
    result = transform(sample)

    assert result["data"].shape == (100, 5)
    expected = data[:, 0:10:2]
    assert torch.allclose(result["data"], expected)


def test_multiple_dimensions():
    """Test slicing multiple dimensions of a single field."""
    data = torch.randn(10, 8, 6)
    sample = TensorDict({"data": data})

    # Slice dim 0 (indices 1, 3) and dim 2 (first 3)
    transform = FieldSlice(
        {
            "data": {
                0: [1, 3],
                2: {"stop": 3},
            }
        }
    )
    result = transform(sample)

    assert result["data"].shape == (2, 8, 3)
    expected = data[[1, 3], :, :][:, :, :3]
    assert torch.allclose(result["data"], expected)


def test_multiple_fields():
    """Test slicing multiple fields."""
    features = torch.randn(100, 10)
    velocity = torch.randn(100, 3)
    sample = TensorDict({"features": features, "velocity": velocity})

    transform = FieldSlice(
        {
            "features": {-1: [0, 2, 5]},
            "velocity": {-1: [0, 1]},  # Keep only x, y
        }
    )
    result = transform(sample)

    assert result["features"].shape == (100, 3)
    assert result["velocity"].shape == (100, 2)


def test_string_keys_for_hydra():
    """Test that string dimension keys work (for Hydra YAML)."""
    data = torch.randn(100, 10)
    sample = TensorDict({"data": data})

    # Use string key "-1" like Hydra would pass
    transform = FieldSlice({"data": {"-1": [0, 2, 5]}})
    result = transform(sample)

    assert result["data"].shape == (100, 3)


def test_missing_field_raises():
    """Test that missing field raises KeyError."""
    sample = TensorDict({"data": torch.randn(10, 10)})

    transform = FieldSlice({"missing": {-1: [0]}})

    with pytest.raises(KeyError, match="missing"):
        transform(sample)
