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

from physicsnemo.models.fengwu import Fengwu

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fengwu_forward(device):
    """Test Fengwu forward pass"""
    torch.manual_seed(0)
    model = Fengwu(
        img_size=(32, 32),
        pressure_level=37,
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ).to(device)
    model.eval()

    bsize = 2
    invar_surface = torch.randn(bsize, 4, 32, 32).to(device)
    invar_z = torch.randn(bsize, 37, 32, 32).to(device)
    invar_r = torch.randn(bsize, 37, 32, 32).to(device)
    invar_u = torch.randn(bsize, 37, 32, 32).to(device)
    invar_v = torch.randn(bsize, 37, 32, 32).to(device)
    invar_t = torch.randn(bsize, 37, 32, 32).to(device)
    invar = model.prepare_input(
        invar_surface, invar_z, invar_r, invar_u, invar_v, invar_t
    )
    # Check output size
    with torch.no_grad():
        assert common.validate_forward_accuracy(model, (invar,), atol=5e-3)

    del invar, model
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fengwu_constructor(device):
    """Test Fengwu constructor options"""
    # Define dictionary of constructor args
    arg_list = [
        {
            "img_size": (64, 64),
            "pressure_level": 37,
            "embed_dim": 192,
            "patch_size": (4, 4),
            "num_heads": (6, 12, 12, 6),
            "window_size": (2, 6, 12),
        },
        {
            "img_size": (32, 64),
            "pressure_level": 37,
            "embed_dim": 192,
            "patch_size": (4, 4),
            "num_heads": (6, 12, 12, 6),
            "window_size": (2, 6, 12),
        },
    ]
    for kw_args in arg_list:
        # Construct FC model
        model = Fengwu(**kw_args).to(device)
        model.eval()

        bsize = random.randint(1, 5)
        invar_surface = torch.randn(
            bsize, 4, kw_args["img_size"][0], kw_args["img_size"][1]
        ).to(device)
        invar_z = torch.randn(
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        ).to(device)
        invar_r = torch.randn(
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        ).to(device)
        invar_u = torch.randn(
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        ).to(device)
        invar_v = torch.randn(
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        ).to(device)
        invar_t = torch.randn(
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        ).to(device)
        invar = model.prepare_input(
            invar_surface, invar_z, invar_r, invar_u, invar_v, invar_t
        )
        outvar_surface, outvar_z, outvar_r, outvar_u, outvar_v, outvar_t = model(invar)
        assert outvar_surface.shape == (
            bsize,
            4,
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
        assert outvar_z.shape == (
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
        assert outvar_r.shape == (
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
        assert outvar_u.shape == (
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
        assert outvar_v.shape == (
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
        assert outvar_t.shape == (
            bsize,
            kw_args["pressure_level"],
            kw_args["img_size"][0],
            kw_args["img_size"][1],
        )
    del model, invar
    torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fengu_optims(device):
    """Test Fengu optimizations"""

    def setup_model():
        """Setups up fresh Fengu model and inputs for each optim test"""
        model = Fengwu(
            img_size=(64, 64),
            pressure_level=37,
            embed_dim=192,
            patch_size=(4, 4),
            num_heads=(6, 12, 12, 6),
            window_size=(2, 6, 12),
        ).to(device)
        model.eval()

        bsize = random.randint(1, 5)
        invar_surface = torch.randn(bsize, 4, 64, 64).to(device)
        invar_z = torch.randn(bsize, 37, 64, 64).to(device)
        invar_r = torch.randn(bsize, 37, 64, 64).to(device)
        invar_u = torch.randn(bsize, 37, 64, 64).to(device)
        invar_v = torch.randn(bsize, 37, 64, 64).to(device)
        invar_t = torch.randn(bsize, 37, 64, 64).to(device)
        invar = model.prepare_input(
            invar_surface, invar_z, invar_r, invar_u, invar_v, invar_t
        )
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
def test_fengwu_deploy(device):
    """Test Fengwu deployment support"""
    # Construct Fengwu model
    model = Fengwu(
        img_size=(64, 64),
        pressure_level=37,
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ).to(device)

    bsize = random.randint(1, 5)
    invar_surface = torch.randn(bsize, 4, 64, 64).to(device)
    invar_z = torch.randn(bsize, 37, 64, 64).to(device)
    invar_r = torch.randn(bsize, 37, 64, 64).to(device)
    invar_u = torch.randn(bsize, 37, 64, 64).to(device)
    invar_v = torch.randn(bsize, 37, 64, 64).to(device)
    invar_t = torch.randn(bsize, 37, 64, 64).to(device)
    invar = model.prepare_input(
        invar_surface, invar_z, invar_r, invar_u, invar_v, invar_t
    )
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
    del model, invar
    torch.cuda.empty_cache()
