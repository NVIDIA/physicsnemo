# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from physicsnemo.experimental.models.healda import scatter_mean


def test_scatter_mean_basic():
    """Test scatter_mean with simple known values"""
    # Create test data:
    # - 5 observations with 2 features each
    # - Scatter into a 3x2 grid (6 cells total)
    # - Some cells will have multiple values (need averaging)
    # - Some cells will be empty (should get fill_value)

    x = torch.tensor(
        [
            [1.0, 10.0],  # goes to cell (0, 0)
            [2.0, 20.0],  # goes to cell (0, 1)
            [3.0, 30.0],  # goes to cell (0, 0) - same as first, should average
            [4.0, 40.0],  # goes to cell (2, 1)
            [5.0, 50.0],  # goes to cell (1, 0)
        ]
    )

    index = torch.tensor(
        [
            [0, 0],  # cell (0, 0)
            [0, 1],  # cell (0, 1)
            [0, 0],  # cell (0, 0)
            [2, 1],  # cell (2, 1)
            [1, 0],  # cell (1, 0)
        ]
    )

    shape = (3, 2)  # 3 rows, 2 columns

    aggregated, present = scatter_mean(x, index, shape)

    # Check shape
    assert aggregated.shape == (3, 2, 2)  # (3, 2) grid with 2 features
    assert present.shape == (3, 2)

    # Check aggregated values
    # Cell (0, 0): mean of [1.0, 10.0] and [3.0, 30.0] = [2.0, 20.0]
    assert torch.allclose(aggregated[0, 0], torch.tensor([2.0, 20.0]))

    # Cell (0, 1): [2.0, 20.0] (single value)
    assert torch.allclose(aggregated[0, 1], torch.tensor([2.0, 20.0]))

    # Cell (1, 0): [5.0, 50.0] (single value)
    assert torch.allclose(aggregated[1, 0], torch.tensor([5.0, 50.0]))

    # Cell (1, 1): empty, should be NaN
    assert torch.isnan(aggregated[1, 1]).all()

    # Cell (2, 0): empty, should be NaN
    assert torch.isnan(aggregated[2, 0]).all()

    # Cell (2, 1): [4.0, 40.0] (single value)
    assert torch.allclose(aggregated[2, 1], torch.tensor([4.0, 40.0]))

    # Check present mask
    expected_present = torch.tensor([[True, True], [True, False], [False, True]])
    assert torch.equal(present, expected_present)


def test_scatter_mean_custom_fill_value():
    """Test scatter_mean with a custom fill value"""
    x = torch.tensor([[1.0, 2.0]])
    index = torch.tensor([[0, 0]])
    shape = (2, 2)
    fill_value = -999.0

    aggregated, present = scatter_mean(x, index, shape, fill_value=fill_value)

    # Cell (0, 0) should have the value
    assert torch.allclose(aggregated[0, 0], torch.tensor([1.0, 2.0]))

    # Other cells should have the fill value
    assert (aggregated[0, 1] == fill_value).all()
    assert (aggregated[1, 0] == fill_value).all()
    assert (aggregated[1, 1] == fill_value).all()

    # Only (0, 0) should be present
    assert present[0, 0]
    assert not present[0, 1]
    assert not present[1, 0]
    assert not present[1, 1]
