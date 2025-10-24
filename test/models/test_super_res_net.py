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

from physicsnemo.models.srrn import SRResNet

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_super_res_net_forward(device):
    """Test super_res_net forward pass"""
    torch.manual_seed(0)
    # Construct super_res_net model
    model_3d = SRResNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    bsize = 8
    invar = torch.randn(bsize, 1, 4, 4, 4).to(device)
    assert common.validate_forward_accuracy(model_3d, (invar,), atol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_super_res_net_constructor(device):
    """Test super_res_net constructor options"""
    # Define dictionary of constructor args
    in_channels = [random.randint(1, 3), random.randint(1, 3)]
    arg_list = [
        {
            "in_channels": in_channels[0],
            "out_channels": in_channels[0],
            "large_kernel_size": 7,
            "small_kernel_size": 3,
            "conv_layer_size": 4,
            "n_resid_blocks": 3,
            "scaling_factor": 2,
        },
        {
            "in_channels": in_channels[1],
            "out_channels": in_channels[1],
            "large_kernel_size": 7,
            "small_kernel_size": 3,
            "conv_layer_size": 3,
            "n_resid_blocks": 4,
            "scaling_factor": 2,
        },
    ]
    for i, kw_args in enumerate(arg_list):
        # Construct FC model
        model = SRResNet(**kw_args).to(device)
        bsize = random.randint(1, 16)
        invar = torch.randn(bsize, in_channels[i], 8, 8, 8).to(device)
        outvar = model(invar)
        assert outvar.shape == (bsize, kw_args["out_channels"], 16, 16, 16)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_super_res_net_optims(device):
    """Test super_res_net optimizations"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        # Construct super_res_net model
        model = SRResNet(in_channels=2, out_channels=2, scaling_factor=2).to(device)

        bsize = 4
        invar = torch.randn(bsize, 2, 8, 8, 8).to(device)
        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_super_res_net_checkpoint(device):
    """Test super_res_net checkpoint save/load"""
    # Construct super_res_net model
    model_1 = SRResNet(
        in_channels=2, out_channels=2, n_resid_blocks=3, scaling_factor=2
    ).to(device)

    model_2 = SRResNet(
        in_channels=2, out_channels=2, n_resid_blocks=3, scaling_factor=2
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, 2, 8, 8, 8).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_super_res_net_deploy(device):
    """Test super_res_net deployment support"""
    # Construct super_res_net model
    model = SRResNet(
        in_channels=1, out_channels=1, n_resid_blocks=4, scaling_factor=2
    ).to(device)

    bsize = random.randint(1, 8)
    invar = torch.randn(bsize, 1, 8, 8, 8).to(device)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
