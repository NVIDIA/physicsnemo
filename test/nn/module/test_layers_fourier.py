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

from physicsnemo.nn import FourierFilter, FourierLayer, GaborFilter


def test_fourier_layer_initialization(device):
    layer = FourierLayer(in_features=2, frequencies=["gaussian", 1, 3]).to(device)
    assert isinstance(layer, FourierLayer)


def test_fourier_layer_forward_pass(device):
    layer = FourierLayer(in_features=2, frequencies=["gaussian", 1, 3]).to(device)
    input_tensor = torch.randn(10, 2).to(device)
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (10, layer.out_features())


def test_fourier_layer_sine_cosine_ranges(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    layer = FourierLayer(in_features=2, frequencies=["gaussian", 1, 3]).to(device)
    output_tensor = layer(input_tensor)
    # Check that values are in the range [-1, 1]
    assert (output_tensor <= 1).all() and (output_tensor >= -1).all()


def test_fourier_filter_initialization(device):
    filter = FourierFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0
    ).to(device)
    assert isinstance(filter, FourierFilter)


def test_fourier_filter_forward_pass(device):
    filter = FourierFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0
    ).to(device)
    input_tensor = torch.randn(10, 2).to(device)
    output_tensor = filter(input_tensor)
    assert output_tensor.shape == (10, 32)


def test_fourier_filter_sine_value_ranges(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    filter = FourierFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0
    ).to(device)
    output_tensor = filter(input_tensor)
    # Check that values are in the range [-1, 1]
    assert (output_tensor <= 1).all() and (output_tensor >= -1).all()


def test_fourier_filter_parameters_update(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    filter = FourierFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0
    ).to(device)
    optimizer = torch.optim.SGD(filter.parameters(), lr=0.01)
    loss = filter(input_tensor).sum()
    loss.backward()
    optimizer.step()
    # Check if parameter values have been updated
    with torch.no_grad():
        assert (
            torch.sum(filter.frequency).item()
            != torch.sum(torch.empty(2, 32).fill_(0)).item()
        )


def test_gabor_filter_initialization(device):
    filter = GaborFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0, alpha=1.0, beta=2.0
    ).to(device)
    assert isinstance(filter, GaborFilter)


def test_gabor_filter_forward_pass(device):
    filter = GaborFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0, alpha=1.0, beta=2.0
    ).to(device)
    input_tensor = torch.randn(10, 2).to(device)
    output_tensor = filter(input_tensor)
    assert output_tensor.shape == (10, 32)


def test_gabor_filter_sine_value_ranges(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    filter = GaborFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0, alpha=1.0, beta=2.0
    ).to(device)
    output_tensor = filter(input_tensor)
    # Check that values are in the range [-1, 1]
    assert (output_tensor <= 1).all() and (output_tensor >= -1).all()


def test_gabor_filter_parameters_update(device):
    input_tensor = torch.Tensor([[1, 1]]).to(device)
    filter = GaborFilter(
        in_features=2, layer_size=32, nr_layers=2, input_scale=1.0, alpha=1.0, beta=2.0
    ).to(device)
    optimizer = torch.optim.SGD(filter.parameters(), lr=0.01)
    loss = filter(input_tensor).sum()
    loss.backward()
    optimizer.step()
    # Check if parameter values have been updated
    with torch.no_grad():
        assert (
            torch.sum(filter.frequency).item()
            != torch.sum(torch.empty(2, 32).fill_(0)).item()
        )
