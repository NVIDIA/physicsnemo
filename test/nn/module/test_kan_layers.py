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
Unit tests for the Kolmogorov–Arnold Network (KAN) layer.
"""

import pytest
import torch

from physicsnemo.nn import KolmogorovArnoldNetwork


def test_kan_initialization(device):
    layer = KolmogorovArnoldNetwork(4, 3, num_harmonics=7).to(device)
    assert layer.fourier_coeffs.shape == (2, 3, 4, 7)
    if layer.add_bias:
        assert layer.bias.shape == (1, 3)


@pytest.mark.parametrize("bias_flag", [True, False])
def test_kan_forward_pass(device, bias_flag):
    batch, in_dim, out_dim = 8, 5, 2
    kan = KolmogorovArnoldNetwork(
        in_dim, out_dim, num_harmonics=4, add_bias=bias_flag
    ).to(device)
    x = torch.randn(batch, in_dim, device=device)
    y = kan(x)
    assert y.shape == (batch, out_dim)
    y.sum().backward()
    for p in kan.parameters():
        assert p.grad is not None


def test_kan_parameter_update(device):
    torch.manual_seed(0)
    kan = KolmogorovArnoldNetwork(3, 1, num_harmonics=3).to(device)
    prev_coeffs = kan.fourier_coeffs.detach().clone()
    prev_bias = kan.bias.detach().clone() if kan.add_bias else None

    x = torch.randn(4, 3, device=device)
    opt = torch.optim.SGD(kan.parameters(), lr=1e-2)
    kan(x).pow(2).mean().backward()
    opt.step()

    assert (prev_coeffs - kan.fourier_coeffs).abs().max() > 0
    if prev_bias is not None:
        assert (prev_bias - kan.bias).abs().max() > 0
