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

import torch

from physicsnemo.nn import SirenLayer


def test_siren_layer_initialization(device):
    layer = SirenLayer(in_features=2, out_features=2).to(device)
    assert isinstance(layer, SirenLayer)


def test_siren_layer_forward_pass(device):
    layer = SirenLayer(in_features=2, out_features=2).to(device)
    input_tensor = torch.randn(10, 2).to(device)
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (10, 2)


def test_siren_layer_sine_cosine_ranges(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    layer = SirenLayer(in_features=2, out_features=2).to(device)
    output_tensor = layer(input_tensor)
    # Check that values are in the range [-1, 1]
    assert (output_tensor <= 1).all() and (output_tensor >= -1).all()


def test_siren_layer_parameters_update(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    layer = SirenLayer(in_features=2, out_features=2).to(device)
    prev_weights = torch.clone(layer.linear.weight)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    loss = layer(input_tensor).sum()
    loss.backward()
    optimizer.step()
    # Check if parameter values have been updated
    with torch.no_grad():
        assert torch.sum(layer.linear.weight).item() != prev_weights.sum().item()
