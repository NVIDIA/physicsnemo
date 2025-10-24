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

from physicsnemo.models.pangu import Pangu

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pangu_forward(device):
    """Test Pangu forward pass"""
    torch.manual_seed(0)
    model = Pangu(
        img_size=(32, 32),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ).to(device)
    model.eval()

    bsize = 2
    invar_surface = torch.randn(bsize, 4, 32, 32).to(device)
    invar_surface_mask = torch.randn(3, 32, 32).to(device)
    invar_upper_air = torch.randn(bsize, 5, 13, 32, 32).to(device)
    invar = model.prepare_input(invar_surface, invar_surface_mask, invar_upper_air)
    # Check output size
    with torch.no_grad():
        assert common.validate_forward_accuracy(model, (invar,), atol=5e-3)

    del model, invar
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pangu_constructor(device):
    """Test Pangu constructor options"""
    # Define dictionary of constructor args
    arg_list = [
        {
            "img_size": (32, 32),
            "patch_size": (2, 4, 4),
            "embed_dim": 96,
            "num_heads": (6, 12, 12, 6),
            "window_size": (2, 6, 12),
        },
        {
            "img_size": (33, 34),
            "patch_size": (2, 4, 4),
            "embed_dim": 96,
            "num_heads": (6, 12, 12, 6),
            "window_size": (2, 6, 6),
        },
    ]
    for kw_args in arg_list:
        # Construct FC model
        model = Pangu(**kw_args).to(device)
        model.eval()

        bsize = random.randint(1, 5)
        invar_surface = torch.randn(
            bsize, 4, kw_args["img_size"][0], kw_args["img_size"][1]
        ).to(device)
        invar_surface_mask = torch.randn(
            3, kw_args["img_size"][0], kw_args["img_size"][1]
        ).to(device)
        invar_upper_air = torch.randn(
            bsize, 5, 13, kw_args["img_size"][0], kw_args["img_size"][1]
        ).to(device)
        invar = model.prepare_input(invar_surface, invar_surface_mask, invar_upper_air)
        outvar_surface, outvar_upper_air = model(invar)
        assert outvar_surface.shape == (
            bsize,
            4,
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
        assert outvar_upper_air.shape == (
            bsize,
            5,
            13,
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
    del model, invar
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pangu_optims(device):
    """Test Pangu optimizations"""

    def setup_model():
        """Setups up fresh Pangu model and inputs for each optim test"""
        model = Pangu(
            img_size=(32, 32),
            patch_size=(2, 4, 4),
            embed_dim=96,
            num_heads=(6, 12, 12, 6),
            window_size=(2, 6, 12),
        ).to(device)
        model.eval()

        bsize = random.randint(1, 5)
        invar_surface = torch.randn(bsize, 4, 32, 32).to(device)
        invar_surface_mask = torch.randn(3, 32, 32).to(device)
        invar_upper_air = torch.randn(bsize, 5, 13, 32, 32).to(device)
        invar = model.prepare_input(invar_surface, invar_surface_mask, invar_upper_air)
        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))
    # Check JIT
    # model, invar = setup_model()
    # assert common.validate_jit(model, (invar,))
    # Check AMP
    # model, invar = setup_model()
    # assert common.validate_amp(model, (invar,))
    # Check Combo
    # model, invar = setup_model()
    # assert common.validate_combo_optims(model, (invar,))
    del model, invar
    torch.cuda.empty_cache()


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_pangu_deploy(device):
    """Test Pangu deployment support"""
    # Construct Pangu model
    model = Pangu(
        img_size=(32, 32),
        patch_size=(2, 4, 4),
        embed_dim=96,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ).to(device)

    bsize = random.randint(1, 5)
    invar_surface = torch.randn(bsize, 4, 32, 32).to(device)
    invar_surface_mask = torch.randn(3, 32, 32).to(device)
    invar_upper_air = torch.randn(bsize, 5, 13, 32, 32).to(device)
    invar = model.prepare_input(invar_surface, invar_surface_mask, invar_upper_air)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
    del model, invar
    torch.cuda.empty_cache()
