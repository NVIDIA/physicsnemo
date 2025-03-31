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

from physicsnemo.models.pix2pix import Pix2Pix

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pix2pix_forward(device):
    """Test pix2pix forward pass"""
    torch.manual_seed(0)
    # Construct pix2pix model
    model_3d = Pix2Pix(
        in_channels=1,
        out_channels=1,
        dimension=3,
        conv_layer_size=8,
        n_downsampling=3,
        n_upsampling=3,
        n_blocks=3,
    ).to(device)

    bsize = 8
    invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
    assert common.validate_forward_accuracy(model_3d, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pix2pix_constructor(device):
    """Test pix2pix constructor options"""
    # Define dictionary of constructor args
    arg_list = []
    for dimension in [1, 2, 3]:
        for padding_type in ["reflect", "replicate", "zero"]:
            arg_list += [
                {
                    "in_channels": random.randint(1, 3),
                    "out_channels": random.randint(1, 3),
                    "dimension": dimension,
                    "conv_layer_size": 16,
                    "n_downsampling": 3,
                    "n_upsampling": 3,
                    "n_blocks": 2,
                    "padding_type": padding_type,
                },
                {
                    "in_channels": random.randint(1, 3),
                    "out_channels": random.randint(1, 3),
                    "dimension": dimension,
                    "conv_layer_size": 8,
                    "n_downsampling": 2,
                    "n_upsampling": 2,
                    "n_blocks": 3,
                    "batch_norm": True,
                    "padding_type": padding_type,
                },
            ]
    for i, kw_args in enumerate(arg_list):
        # Construct FC model
        model = Pix2Pix(**kw_args).to(device)
        bsize = random.randint(1, 16)
        if kw_args["dimension"] == 1:
            invar = torch.randn(bsize, kw_args["in_channels"], 16).to(device)
        elif kw_args["dimension"] == 2:
            invar = torch.randn(bsize, kw_args["in_channels"], 16, 16).to(device)
        else:
            invar = torch.randn(bsize, kw_args["in_channels"], 16, 16, 16).to(device)
        outvar = model(invar)
        assert outvar.shape == (bsize, kw_args["out_channels"], *invar.shape[2:])

    # Also test failure case
    try:
        model = Pix2Pix(
            in_channels=1,
            out_channels=1,
            dimension=4,
            conv_layer_size=8,
            n_downsampling=3,
            n_upsampling=3,
            n_blocks=3,
        ).to(device)
        raise AssertionError("Failed to error for invalid dimension")
    except NotImplementedError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pix2pix_optims(device):
    """Test pix2pix optimizations"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        # Construct pix2pix model
        model = Pix2Pix(
            in_channels=2,
            out_channels=2,
            dimension=1,
            conv_layer_size=2,
            n_downsampling=2,
            n_upsampling=2,
            n_blocks=3,
        ).to(device)

        bsize = 4
        invar = torch.randn(bsize, 2, 16).to(device)
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
def test_pix2pix_checkpoint(device):
    """Test pix2pix checkpoint save/load"""
    # Construct pix2pix model
    model_1 = Pix2Pix(
        in_channels=2,
        out_channels=2,
        dimension=2,
        conv_layer_size=4,
        n_downsampling=2,
        n_upsampling=2,
        n_blocks=2,
    ).to(device)

    model_2 = Pix2Pix(
        in_channels=2,
        out_channels=2,
        dimension=2,
        conv_layer_size=4,
        n_downsampling=2,
        n_upsampling=2,
        n_blocks=2,
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, 2, 16, 16).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pix2pix_deploy(device):
    """Test pix2pix deployment support"""
    # Construct pix2pix model
    model = Pix2Pix(
        in_channels=2,
        out_channels=2,
        dimension=3,
        conv_layer_size=8,
        n_downsampling=2,
        n_upsampling=2,
        n_blocks=2,
    ).to(device)

    bsize = random.randint(1, 8)
    invar = torch.randn(bsize, 2, 32, 32, 32).to(device)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("upsample", [1, 2])
def test_pix2pix_upsample(device, upsample):
    """Test pix2pix upsampling functionality"""
    # Construct pix2pix model
    model = Pix2Pix(
        in_channels=2,
        out_channels=2,
        dimension=2,
        conv_layer_size=8,
        n_downsampling=2,
        n_upsampling=(2 + upsample),
        n_blocks=2,
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, 2, 8, 8).to(device)
    outvar = model(invar)
    assert outvar.shape == (bsize, 2, 8 * 2 ** (upsample), 8 * 2 ** (upsample))
