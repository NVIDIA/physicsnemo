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

"""Tests for uncapped weighted sampling."""

import pytest
import torch

from physicsnemo.utils import _weighted_sampling
from physicsnemo.utils._weighted_sampling import weighted_sample_without_replacement


def test_weighted_sampling_avoids_torch_category_limit(monkeypatch):
    def reject_multinomial(*args, **kwargs):
        raise AssertionError("torch.multinomial must not be used")

    monkeypatch.setattr(torch, "multinomial", reject_multinomial)
    monkeypatch.setattr(_weighted_sampling, "_WEIGHTED_SAMPLE_CHUNK_SIZE", 2)
    weights = torch.tensor([1.0, 0.0, 4.0, 2.0, 3.0])

    first = weighted_sample_without_replacement(
        weights,
        3,
        generator=torch.Generator().manual_seed(0),
    )
    second = weighted_sample_without_replacement(
        weights,
        3,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(first, second)
    assert torch.unique(first).numel() == 3
    assert 1 not in first


def test_weighted_sampling_applies_chunk_offsets(monkeypatch):
    monkeypatch.setattr(_weighted_sampling, "_WEIGHTED_SAMPLE_CHUNK_SIZE", 4)
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    indices = weighted_sample_without_replacement(
        weights,
        2,
        generator=torch.Generator().manual_seed(0),
    )

    assert set(indices.tolist()) == {4, 5}


@pytest.mark.parametrize("count", [-1, 4])
def test_weighted_sampling_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="count"):
        weighted_sample_without_replacement(torch.ones(3), count)


def test_weighted_sampling_rejects_invalid_weights():
    with pytest.raises(ValueError, match="1D"):
        weighted_sample_without_replacement(torch.ones(2, 2), 2)
    with pytest.raises(TypeError, match="floating point"):
        weighted_sample_without_replacement(torch.ones(3, dtype=torch.long), 2)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, torch.nan]),
        torch.tensor([1.0, torch.inf]),
        torch.zeros(2),
    ],
)
def test_weighted_sampling_rejects_invalid_weight_values(weights):
    with pytest.raises(RuntimeError, match="weights"):
        weighted_sample_without_replacement(weights, 1)
