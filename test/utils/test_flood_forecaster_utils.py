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

"""
Unit tests for FloodForecaster utility modules.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from utils.normalization import (
    collect_all_fields,
    stack_and_fit_transform,
    transform_with_existing_normalizers,
)


# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")


@pytest.mark.parametrize("device", _DEVICES)
def test_collect_all_fields_with_target(device):
    """Test collecting fields with target."""
    # Create mock dataset
    mock_dataset = [
        {
            "geometry": torch.rand(100, 2).to(device),
            "static": torch.rand(100, 7).to(device),
            "boundary": torch.rand(3, 100, 1).to(device),
            "dynamic": torch.rand(3, 100, 3).to(device),
            "target": torch.rand(100, 3).to(device),
        }
        for _ in range(5)
    ]

    geom, static, boundary, dyn, target = collect_all_fields(mock_dataset, expect_target=True)

    assert len(geom) == 5
    assert len(static) == 5
    assert len(boundary) == 5
    assert len(dyn) == 5
    assert len(target) == 5


@pytest.mark.parametrize("device", _DEVICES)
def test_collect_all_fields_without_target(device):
    """Test collecting fields without target."""
    mock_dataset = [
        {
            "geometry": torch.rand(100, 2).to(device),
            "static": torch.rand(100, 7).to(device),
            "boundary": torch.rand(3, 100, 1).to(device),
            "dynamic": torch.rand(3, 100, 3).to(device),
        }
        for _ in range(5)
    ]

    result = collect_all_fields(mock_dataset, expect_target=False)

    assert len(result) == 5  # geom, static, boundary, dyn, target (empty)


@pytest.mark.parametrize("device", _DEVICES)
def test_collect_all_fields_with_cell_area(device):
    """Test collecting fields with cell_area."""
    mock_dataset = [
        {
            "geometry": torch.rand(100, 2).to(device),
            "static": torch.rand(100, 7).to(device),
            "boundary": torch.rand(3, 100, 1).to(device),
            "dynamic": torch.rand(3, 100, 3).to(device),
            "target": torch.rand(100, 3).to(device),
            "cell_area": torch.rand(100).to(device),
        }
        for _ in range(5)
    ]

    result = collect_all_fields(mock_dataset, expect_target=True)

    assert len(result) == 6  # Includes cell_area
    assert len(result[5]) == 5  # 5 cell_area tensors


@pytest.mark.parametrize("device", _DEVICES)
def test_stack_and_fit_transform_creates_normalizers(device):
    """Test that stack_and_fit_transform creates normalizers."""
    geom = [torch.rand(100, 2).to(device) for _ in range(5)]
    static = [torch.rand(100, 7).to(device) for _ in range(5)]
    boundary = [torch.rand(3, 100, 1).to(device) for _ in range(5)]
    dyn = [torch.rand(3, 100, 3).to(device) for _ in range(5)]
    target = [torch.rand(100, 3).to(device) for _ in range(5)]

    normalizers, big_tensors = stack_and_fit_transform(geom, static, boundary, dyn, target)

    assert "geometry" in normalizers
    assert "static" in normalizers
    assert "boundary" in normalizers
    assert "target" in normalizers
    assert "dynamic" in normalizers

    assert big_tensors["geometry"].shape[0] == 5
    assert big_tensors["static"].shape[0] == 5


@pytest.mark.parametrize("device", _DEVICES)
def test_stack_and_fit_transform_uses_existing(device):
    """Test using existing normalizers."""
    geom = [torch.rand(100, 2).to(device) for _ in range(5)]
    static = [torch.rand(100, 7).to(device) for _ in range(5)]
    boundary = [torch.rand(3, 100, 1).to(device) for _ in range(5)]
    dyn = [torch.rand(3, 100, 3).to(device) for _ in range(5)]
    target = [torch.rand(100, 3).to(device) for _ in range(5)]

    # First pass - fit normalizers
    normalizers, _ = stack_and_fit_transform(
        geom, static, boundary, dyn, target, fit_normalizers=True
    )

    # Second pass - use existing
    new_geom = [torch.rand(100, 2).to(device) for _ in range(3)]
    new_static = [torch.rand(100, 7).to(device) for _ in range(3)]
    new_boundary = [torch.rand(3, 100, 1).to(device) for _ in range(3)]
    new_dyn = [torch.rand(3, 100, 3).to(device) for _ in range(3)]
    new_target = [torch.rand(100, 3).to(device) for _ in range(3)]

    _, new_tensors = stack_and_fit_transform(
        new_geom,
        new_static,
        new_boundary,
        new_dyn,
        new_target,
        normalizers=normalizers,
        fit_normalizers=False,
    )

    assert new_tensors["geometry"].shape[0] == 3


@pytest.mark.parametrize("device", _DEVICES)
def test_transform_with_existing_normalizers(device):
    """Test transform_with_existing_normalizers."""
    # Create and fit normalizers
    geom = [torch.rand(100, 2).to(device) for _ in range(5)]
    static = [torch.rand(100, 7).to(device) for _ in range(5)]
    boundary = [torch.rand(3, 100, 1).to(device) for _ in range(5)]
    dyn = [torch.rand(3, 100, 3).to(device) for _ in range(5)]
    target = [torch.rand(100, 3).to(device) for _ in range(5)]

    normalizers, _ = stack_and_fit_transform(geom, static, boundary, dyn, target)

    # Transform new data
    new_geom = [torch.rand(100, 2).to(device) for _ in range(3)]
    new_static = [torch.rand(100, 7).to(device) for _ in range(3)]
    new_boundary = [torch.rand(3, 100, 1).to(device) for _ in range(3)]
    new_dyn = [torch.rand(3, 100, 3).to(device) for _ in range(3)]

    transformed = transform_with_existing_normalizers(
        new_geom, new_static, new_boundary, new_dyn, normalizers
    )

    assert "geometry" in transformed
    assert "static" in transformed
    assert "boundary" in transformed
    assert "dynamic" in transformed

    assert transformed["geometry"].shape[0] == 3

