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

import random

import pytest
import torch

from physicsnemo.models.layers.activations import (
    CappedGELU,
    CappedLeakyReLU,
    Identity,
    SquarePlus,
    Stan,
)

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_identity(device):
    """Test identity function in layers"""
    func = Identity().to(device)
    # Random tensor of random size
    tensor_dim = random.randint(1, 5)
    tensor_size = torch.randint(low=1, high=8, size=(tensor_dim,)).tolist()
    invar = torch.randn(*tensor_size, device=device)

    outvar = func(invar)
    assert common.compare_output(invar, outvar)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_stan(device):
    """Test Stan function in layers"""
    func = Stan(out_features=2).to(device)
    # Doc string example handles accuracy
    bsize = random.randint(1, 8)
    invar = torch.randn(bsize, 2).to(device)
    outvar = func(invar)
    # Learnable param should be 1.0 init
    tarvar = (invar + 1) * torch.tanh(invar)
    assert common.compare_output(tarvar, outvar)

    # Also test failure case
    try:
        func = Stan(out_features=random.randint(1, 3)).to(device)
        invar = torch.randn(2, 4).to(device)
        outvar = func(invar)
        assert False, "Failed to error for invalid input feature dimension"
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_squareplus(device):
    """Test square plus function in layers"""
    func = SquarePlus().to(device)
    func.b = 0
    # Ones tensor of random size
    tensor_dim = random.randint(1, 3)
    tensor_size = torch.randint(low=1, high=4, size=(tensor_dim,)).tolist()
    invar = torch.ones(*tensor_size, device=device)

    outvar = func(invar)
    assert common.compare_output(torch.ones_like(invar), outvar)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_capped_leaky_relu(device):
    """Test capped_gelu function in layers"""
    func = CappedLeakyReLU(cap_value=1.0).to(device)
    leaky_relu_func = torch.nn.LeakyReLU()

    # check if identical to leaky relu below capped value
    tensor_dim = random.randint(1, 3)
    tensor_size = torch.randint(low=1, high=4, size=(tensor_dim,)).tolist()
    invar = torch.randint(low=-5, high=1, size=tensor_size, device=device).to(
        torch.float32
    )

    outvar = func(invar)
    assert common.compare_output(leaky_relu_func(invar), outvar)

    # check if value is capped properly
    invar = torch.randint(low=1, high=5, size=tensor_size, device=device).to(
        torch.float32
    )

    outvar = func(invar)
    assert common.compare_output(torch.ones_like(invar), outvar)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_capped_gelu(device):
    """Test capped_gelu function in layers"""
    func = CappedGELU(cap_value=1.0).to(device)
    gelu_func = torch.nn.GELU()

    # check if identical to gelu below capped value
    tensor_dim = random.randint(1, 3)
    tensor_size = torch.randint(low=1, high=4, size=(tensor_dim,)).tolist()
    invar = torch.randint(low=-5, high=1, size=tensor_size, device=device).to(
        torch.float32
    )

    outvar = func(invar)
    assert common.compare_output(gelu_func(invar), outvar)

    # check if value is capped properly
    invar = torch.randint(low=2, high=5, size=tensor_size, device=device).to(
        torch.float32
    )

    outvar = func(invar)
    assert common.compare_output(torch.ones_like(invar), outvar)


@pytest.mark.skipif(
    not common.utils.is_fusion_available("FusionDefinition"),
    reason="nvfuser module is not available or has incorrect version",
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_activation_fused_silu(device):
    """Test fused SiLU implementation"""

    from physicsnemo.models.layers.fused_silu import (
        FusedSiLU,
        FusedSiLU_deriv_1,
        FusedSiLU_deriv_2,
        FusedSiLU_deriv_3,
    )

    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True, device=device)
    assert torch.autograd.gradcheck(
        FusedSiLU.apply, input, eps=1e-6, atol=1e-4
    ), "Failed FusedSiLU autograd check"

    assert torch.autograd.gradcheck(
        FusedSiLU_deriv_1.apply, input, eps=1e-6, atol=1e-4
    ), "Failes FusedSiLU_deriv_1 autograd check"

    assert torch.autograd.gradcheck(
        FusedSiLU_deriv_2.apply, input, eps=1e-6, atol=1e-4
    ), "Failes FusedSiLU_deriv_2 autograd check"

    assert torch.autograd.gradcheck(
        FusedSiLU_deriv_3.apply, input, eps=1e-6, atol=1e-4
    ), "Failes FusedSiLU_deriv_3 autograd check"
