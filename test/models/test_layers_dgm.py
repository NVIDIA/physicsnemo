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

import pytest
import torch

from physicsnemo.models.layers import DGMLayer


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dgm_layer_initialization(device):
    layer = DGMLayer(2, 2, 3).to(device)
    assert isinstance(layer, DGMLayer)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dgm_layer_forward_pass(device):
    layer = DGMLayer(4, 3, 2).to(device)
    input_tensor_1 = torch.randn(2, 4).to(device)
    input_tensor_2 = torch.randn(2, 3).to(device)
    output_tensor = layer(input_tensor_1, input_tensor_2)
    assert output_tensor.shape == (2, 2)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dgm_layer_parameters_update(device):
    input_tensor_1 = torch.Tensor([[1, 1]]).to(device)
    input_tensor_2 = torch.Tensor([[2, 2]]).to(device)
    layer = DGMLayer(2, 2, 2).to(device)
    prev_weights_1 = torch.clone(layer.linear_1.weight)
    prev_weights_2 = torch.clone(layer.linear_2.weight)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    loss = layer(input_tensor_1, input_tensor_2).sum()
    loss.backward()
    optimizer.step()
    # Check if parameter values have been updated
    with torch.no_grad():
        assert torch.sum(layer.linear_1.weight).item() != prev_weights_1.sum().item()
        assert torch.sum(layer.linear_2.weight).item() != prev_weights_2.sum().item()
