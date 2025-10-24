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

from physicsnemo.models.swinvrnn import SwinRNN

from . import common


# Skip CPU tests because too slow
@pytest.mark.parametrize("device", ["cuda:0"])
def test_swinrnn_forward(device):
    """Test SwinRNN forward pass"""
    torch.manual_seed(0)
    model = SwinRNN(
        img_size=(6, 32, 64),
        patch_size=(6, 1, 1),
        in_chans=13,
        out_chans=13,
        embed_dim=768,
        num_groups=32,
        num_heads=8,
        window_size=8,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 13, 6, 32, 64).to(device)
    # Check output size
    with torch.no_grad():
        assert common.validate_forward_accuracy(model, (invar,), atol=5e-3, rtol=1e-3)
    del invar, model
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_swinrnn_constructor(device):
    """Test SwinRNN constructor options"""
    # Define dictionary of constructor args
    arg_list = [
        {
            "img_size": (3, 32, 32),
            "patch_size": (3, 1, 1),
            "in_chans": 13,
            "out_chans": 13,
            "embed_dim": 128,
            "num_groups": 32,
            "num_heads": 8,
            "window_size": 8,
        },
    ]
    for kw_args in arg_list:
        # Construct FC model
        model = SwinRNN(**kw_args).to(device)

        bsize = random.randint(1, 5)
        invar = torch.randn(
            bsize,
            kw_args["in_chans"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
            kw_args["img_size"][2],
        ).to(device)
        outvar = model(invar)
        assert outvar.shape == (
            bsize,
            kw_args["out_chans"],
            kw_args["img_size"][1],
            kw_args["img_size"][2],
        )
    del model, invar, outvar
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_swinrnn_optims(device):
    """Test SwinRNN optimizations"""

    def setup_model():
        """Setups up fresh SwinRNN model and inputs for each optim test"""
        model = SwinRNN(
            img_size=(6, 32, 64),
            patch_size=(6, 1, 1),
            in_chans=13,
            out_chans=13,
            embed_dim=128,
            num_groups=32,
            num_heads=8,
            window_size=8,
        ).to(device)

        bsize = random.randint(1, 5)
        invar = torch.randn(bsize, 13, 6, 32, 64).to(device)
        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))
    # Check JIT
    # model, invar_surface, invar_surface_mask, invar_upper_air = setup_model()
    # assert common.validate_jit(model, (invar_surface, invar_surface_mask, invar_upper_air))
    # Check AMP
    # model, invar_surface, invar_surface_mask, invar_upper_air = setup_model()
    # assert common.validate_amp(model, (invar_surface, invar_surface_mask, invar_upper_air))
    # Check Combo
    # model, invar_surface, invar_surface_mask, invar_upper_air = setup_model()
    # assert common.validate_combo_optims(model, (invar_surface, invar_surface_mask, invar_upper_air))
    del model, invar
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_swinrnn_checkpoint(device):
    """Test SwinRNN checkpoint save/load"""
    # Construct SwinRNN models
    model_1 = SwinRNN(
        img_size=(6, 32, 64),
        patch_size=(6, 1, 1),
        in_chans=13,
        out_chans=13,
        embed_dim=128,
        num_groups=32,
        num_heads=8,
        window_size=8,
    ).to(device)

    model_2 = SwinRNN(
        img_size=(6, 32, 64),
        patch_size=(6, 1, 1),
        in_chans=13,
        out_chans=13,
        embed_dim=128,
        num_groups=32,
        num_heads=8,
        window_size=8,
    ).to(device)

    bsize = random.randint(1, 5)
    invar = torch.randn(bsize, 13, 6, 32, 64).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))
    del model_1, model_2, invar
    torch.cuda.empty_cache()


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_swinrnn_deploy(device):
    """Test SwinRNN deployment support"""
    # Construct SwinRNN model
    model = SwinRNN(
        img_size=(6, 32, 64),
        patch_size=(6, 1, 1),
        in_chans=13,
        out_chans=13,
        embed_dim=128,
        num_groups=32,
        num_heads=8,
        window_size=8,
    ).to(device)

    bsize = random.randint(1, 5)
    invar = torch.randn(bsize, 13, 6, 32, 64).to(device)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
    del model, invar
    torch.cuda.empty_cache()
