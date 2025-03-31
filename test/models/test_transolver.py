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

from physicsnemo.models.transolver import Transolver

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_transolver_forward(device):
    """Test FNO forward pass"""
    torch.manual_seed(0)
    # Construct FNO model
    model = Transolver(
        space_dim=2,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=1,
        H=85,
        W=85,
    ).to(device)

    bsize = 4
    pos = torch.randn(bsize, 85, 85).to(device)
    invar = torch.randn(bsize, 85 * 85, 1).to(device)

    assert common.validate_forward_accuracy(
        model,
        (
            pos,
            invar,
        ),
        file_name="transolver_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_transolver_constructor(device):
    """Test transolver constructor options"""
    # Define dictionary of constructor args
    model = Transolver(
        space_dim=2,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=1,
        H=85,
        W=85,
    ).to(device)

    bsize = random.randint(1, 4)
    pos = torch.randn(bsize, 85, 85).to(device)
    invar = torch.randn(bsize, 85 * 85, 1).to(device)

    outvar = model(pos, invar)
    assert outvar.shape == (bsize, 85, 85, 1)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_transolver_optims(device):
    """Test transolver optimizations"""

    def setup_model():
        """Setups up fresh transolver model and inputs for each optim test"""
        model = Transolver(
            space_dim=2,
            n_layers=8,
            n_hidden=64,
            dropout=0,
            n_head=4,
            Time_Input=False,
            act="gelu",
            mlp_ratio=1,
            fun_dim=1,
            out_dim=1,
            slice_num=32,
            ref=8,
            unified_pos=1,
            H=85,
            W=85,
        ).to(device)

        bsize = random.randint(1, 2)
        pos = torch.randn(bsize, 85, 85).to(device)
        invar = torch.randn(bsize, 85 * 85, 1).to(device)

        return model, pos, invar

    # Ideally always check graphs first
    model, pos, invar = setup_model()
    assert common.validate_cuda_graphs(
        model,
        (
            pos,
            invar,
        ),
    )

    # Check JIT
    model, pos, invar = setup_model()
    assert common.validate_jit(
        model,
        (
            pos,
            invar,
        ),
    )
    # Check AMP
    model, pos, invar = setup_model()
    assert common.validate_amp(
        model,
        (
            pos,
            invar,
        ),
    )
    # Check Combo
    model, pos, invar = setup_model()
    assert common.validate_combo_optims(
        model,
        (
            pos,
            invar,
        ),
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_transolver_checkpoint(device):
    """Test transolver checkpoint save/load"""
    # Construct transolver models
    model_1 = Transolver(
        space_dim=2,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=1,
        H=85,
        W=85,
    ).to(device)

    model_2 = Transolver(
        space_dim=2,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=1,
        H=85,
        W=85,
    ).to(device)

    bsize = random.randint(1, 2)
    pos = torch.randn(bsize, 85, 85).to(device)
    invar = torch.randn(bsize, 85 * 85, 1).to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (
            pos,
            invar,
        ),
    )


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_transolverdeploy(device):
    """Test transolver deployment support"""
    # Construct transolver model
    model = Transolver(
        space_dim=2,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=1,
        H=85,
        W=85,
    ).to(device)

    bsize = random.randint(1, 2)
    pos = torch.randn(bsize, 85, 85).to(device)
    invar = torch.randn(bsize, 85 * 85, 1).to(device)

    assert common.validate_onnx_export(
        model,
        (
            pos,
            invar,
        ),
    )
    assert common.validate_onnx_runtime(
        model,
        (
            invar,
            invar,
        ),
        1e-2,
        1e-2,
    )
