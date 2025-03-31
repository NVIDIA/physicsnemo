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

from physicsnemo.models.fno import FNO

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_fno_forward(device, dimension):
    """Test FNO forward pass"""
    torch.manual_seed(0)
    # Construct FNO model
    model = FNO(
        in_channels=2,
        out_channels=2,
        decoder_layers=1,
        decoder_layer_size=8,
        dimension=dimension,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=4,
        padding=0,
    ).to(device)

    bsize = 4
    if dimension == 1:
        invar = torch.randn(bsize, 2, 16).to(device)
    elif dimension == 2:
        invar = torch.randn(bsize, 2, 16, 16).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 2, 16, 16, 16).to(device)
    else:
        invar = torch.randn(bsize, 2, 16, 16, 16, 16).to(device)

    assert common.validate_forward_accuracy(
        model, (invar,), file_name=f"fno{dimension}d_output.pth", atol=1e-3
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fno_constructor(device):
    """Test FNO constructor options"""

    out_features = random.randint(1, 8)
    # Define dictionary of constructor args
    arg_list = [
        {
            "in_channels": random.randint(1, 4),
            "out_channels": out_features,
            "decoder_layers": 1,
            "decoder_layer_size": 8,
            "dimension": dimension,
            "latent_channels": 32,
            "num_fno_layers": 2,
            "num_fno_modes": 4,
            "padding": 4,
            "coord_features": False,
        }
        for dimension in [1, 2, 3, 4]
    ]

    for kw_args in arg_list:
        # Construct FC model
        model = FNO(**kw_args).to(device)

        bsize = random.randint(1, 4)
        if kw_args["dimension"] == 1:
            invar = torch.randn(bsize, kw_args["in_channels"], 8).to(device)
        elif kw_args["dimension"] == 2:
            invar = torch.randn(bsize, kw_args["in_channels"], 8, 8).to(device)
        elif kw_args["dimension"] == 3:
            invar = torch.randn(bsize, kw_args["in_channels"], 8, 8, 8).to(device)
        else:
            invar = torch.randn(bsize, kw_args["in_channels"], 8, 8, 8, 8).to(device)

        outvar = model(invar)
        assert outvar.shape == (bsize, out_features, *invar.shape[2:])

    # Also test failure case
    try:
        model = FNO(
            in_channels=2,
            out_channels=2,
            decoder_layers=1,
            decoder_layer_size=8,
            dimension=5,
            latent_channels=32,
            num_fno_layers=4,
            num_fno_modes=4,
            padding=0,
        ).to(device)
        raise AssertionError("Failed to error for invalid dimension")
    except NotImplementedError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_fno_optims(device, dimension):
    """Test FNO optimizations"""

    def setup_model():
        """Setups up fresh FNO model and inputs for each optim test"""
        model = FNO(
            in_channels=2,
            out_channels=2,
            decoder_layers=1,
            decoder_layer_size=8,
            dimension=dimension,
            latent_channels=4,
            num_fno_layers=4,
            num_fno_modes=4,
            padding=0,
        ).to(device)

        bsize = random.randint(1, 5)
        if dimension == 1:
            invar = torch.randn(bsize, 2, 8).to(device)
        elif dimension == 2:
            invar = torch.randn(bsize, 2, 8, 8).to(device)
        elif dimension == 3:
            invar = torch.randn(bsize, 2, 8, 8, 8).to(device)
        else:
            invar = torch.randn(bsize, 2, 8, 8, 8, 8).to(device)

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
@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_fno_checkpoint(device, dimension):
    """Test FNO checkpoint save/load"""
    # Construct FNO models
    model_1 = FNO(
        in_channels=2,
        out_channels=2,
        decoder_layers=2,
        decoder_layer_size=8,
        dimension=dimension,
        latent_channels=4,
        num_fno_layers=2,
        num_fno_modes=2,
        padding=0,
    ).to(device)

    model_2 = FNO(
        in_channels=2,
        out_channels=2,
        decoder_layers=2,
        decoder_layer_size=8,
        dimension=dimension,
        latent_channels=4,
        num_fno_layers=2,
        num_fno_modes=2,
        padding=0,
    ).to(device)

    bsize = random.randint(1, 2)
    if dimension == 1:
        invar = torch.randn(bsize, 2, 8).to(device)
    elif dimension == 2:
        invar = torch.randn(bsize, 2, 8, 8).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 2, 8, 8, 8).to(device)
    else:
        invar = torch.randn(bsize, 2, 8, 8, 8, 8).to(device)

    assert common.validate_checkpoint(model_1, model_2, (invar,))


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_fnodeploy(device, dimension):
    """Test FNO deployment support"""
    # Construct AFNO model
    model = FNO(
        in_channels=2,
        out_channels=2,
        decoder_layers=2,
        decoder_layer_size=8,
        dimension=dimension,
        latent_channels=4,
        num_fno_layers=2,
        num_fno_modes=2,
        padding=0,
    ).to(device)

    bsize = random.randint(1, 2)
    bsize = random.randint(1, 2)
    if dimension == 1:
        invar = torch.randn(bsize, 2, 4).to(device)
    elif dimension == 2:
        invar = torch.randn(bsize, 2, 4, 4).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 2, 4, 4, 4).to(device)
    else:
        invar = torch.randn(bsize, 2, 4, 4, 4, 4).to(device)

    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
