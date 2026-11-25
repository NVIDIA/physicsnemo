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

"""Test the documented weighted-mean recipe independently of library tests."""

import pytest
import torch

from weighted_mean import WeightedMeanRule, main

from physicsnemo.metrics.general.tensor_accumulator import TensorAccumulator


def test_weighted_mean_recipe():
    """Run the exact example, including streaming and independent-state combine."""
    main()


@pytest.mark.parametrize("dim", [0, 1, None])
def test_weighted_mean_rule(dim):
    """Unequal weights and zero-weight elements agree with the dense expression."""
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    w = torch.linspace(0, 2, 12, dtype=torch.float64).reshape_as(x)
    state = TensorAccumulator(WeightedMeanRule())
    if dim is None:
        state.update(x, dim=None, weights=w + 1)
        state.update(x + 2, dim=None, weights=w + 2)
        expected = (x * (w + 1) + (x + 2) * (w + 2)) / (2 * w + 3)
    else:
        for data, weight in zip(x.split(1, dim), w.split(1, dim)):
            state.update(data, dim=dim, weights=weight)
        expected = (x * w).sum(dim) / w.sum(dim)
    torch.testing.assert_close(state.finalize(), expected)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_weights_leave_example_state_unchanged(weight):
    """The example's validation rejects bad weights without losing prior data."""
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    state = TensorAccumulator(WeightedMeanRule())
    state.update(x, weights=torch.ones_like(x))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        state.update(x, weights=torch.full_like(x, weight))
    torch.testing.assert_close(state.finalize(), x.mean(0))
    assert state.count == 4


def test_zero_weight_output_is_rejected():
    """A zero-weight output has no defined weighted mean."""
    x = torch.ones(2, 3)
    weights = torch.ones_like(x)
    weights[:, 0] = 0
    state = TensorAccumulator(WeightedMeanRule())
    state.update(x, weights=weights)
    with pytest.raises(ValueError, match="positive finite"):
        state.finalize()
