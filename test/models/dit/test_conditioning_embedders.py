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
import pytest
import torch

from physicsnemo.experimental.models.dit import (
    DiTConditionEmbedder,
    EDMConditionEmbedder,
    ZeroConditioningEmbedder,
)
from test import common


def test_dit_condition_embedder_forward(device):
    """Test DiTConditionEmbedder forward pass."""
    torch.manual_seed(42)

    hidden_size = 128
    condition_dim = 64
    batch_size = 4

    model = DiTConditionEmbedder(
        hidden_size=hidden_size, condition_dim=condition_dim
    ).to(device)
    model.eval()

    t = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, condition_dim).to(device)

    with torch.no_grad():
        output = model(t, condition=condition)

    assert common.validate_tensor_accuracy(
        output,
        file_name="models/dit/data/dit_condition_embedder_output.pth",
    )

    out_none = model(t, condition=None)
    assert out_none.shape == (batch_size, hidden_size)


@pytest.mark.parametrize(
    "condition_dim,legacy_condition_bias,expect_map_condition",
    [
        (0, False, False),  # No condition, no legacy -> no map_condition
        (0, True, True),  # No condition, legacy -> map_condition with in_features=0
        (128, False, True),  # With condition -> map_condition
    ],
)
def test_edm_condition_embedder_constructor(
    condition_dim, legacy_condition_bias, expect_map_condition
):
    """Test EDMConditionEmbedder constructor with different condition dims and legacy bias."""
    model = EDMConditionEmbedder(
        emb_channels=256,
        noise_channels=64,
        condition_dim=condition_dim,
        legacy_condition_bias=legacy_condition_bias,
    )
    assert model.output_dim == 256
    if expect_map_condition:
        assert model.map_condition is not None
        assert model.map_condition.in_features == condition_dim
    else:
        assert model.map_condition is None


def test_edm_condition_embedder_forward(device):
    """Test EDMConditionEmbedder forward pass."""
    torch.manual_seed(42)

    emb_channels = 256
    noise_channels = 64
    condition_dim = 32
    batch_size = 4

    model = EDMConditionEmbedder(
        emb_channels=emb_channels,
        noise_channels=noise_channels,
        condition_dim=condition_dim,
    ).to(device)
    model.eval()

    t = torch.rand(batch_size).to(device)
    condition = torch.randn(batch_size, condition_dim).to(device)

    with torch.no_grad():
        output = model(t, condition=condition)

    assert common.validate_tensor_accuracy(
        output,
        file_name="models/dit/data/edm_condition_embedder_output.pth",
    )


def test_zero_conditioning_embedder():
    """Test ZeroConditioningEmbedder forward pass."""
    model = ZeroConditioningEmbedder()
    assert model.output_dim == 0

    t = torch.rand(4)
    out = model(t)
    assert out.shape == (4, 0)
