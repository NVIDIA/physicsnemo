# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

import physicsnemo
from physicsnemo.models.swinvrnn import SwinRNN
from test import common


# Skip CPU tests because too slow
def test_swinrnn_forward(device):
    """Test SwinRNN forward pass"""

    if device == "cpu":
        pytest.skip("SwinRNN cpu test too slow")

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
        assert common.validate_forward_accuracy(
            model,
            (invar,),
            atol=5e-3,
            rtol=1e-3,
            file_name="models/swinrnn/data/swinrnn_output.pth",
        )
    del invar, model
    torch.cuda.empty_cache()


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
        assert model.img_size == kw_args["img_size"]
        assert model.patch_size == kw_args["patch_size"]
        assert model.in_chans == kw_args["in_chans"]
        assert model.out_chans == kw_args["out_chans"]
        assert model.embed_dim == kw_args["embed_dim"]

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


def test_swinrnn_optims(device):
    """Test SwinRNN optimizations"""
    if device == "cpu":
        pytest.skip("CUDA only")

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


def test_swinrnn_checkpoint(device):
    """Test SwinRNN checkpoint save/load"""

    if device == "cpu":
        pytest.skip("CUDA only")

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


def test_swinrnn_load_checkpoint(device, tmp_path):
    """Test loading SwinRNN from a saved checkpoint path."""
    if device == "cpu":
        pytest.skip("CUDA only")

    model_kwds = {
        "img_size": (6, 32, 64),
        "patch_size": (6, 1, 1),
        "in_chans": 13,
        "out_chans": 13,
        "embed_dim": 128,
        "num_groups": 32,
        "num_heads": 8,
        "window_size": 8,
    }
    model = SwinRNN(**model_kwds).to(device).eval()
    checkpoint_path = tmp_path / "swinrnn_checkpoint.mdlus"
    model.save(str(checkpoint_path))

    loaded = physicsnemo.Module.from_checkpoint(str(checkpoint_path)).to(device).eval()
    assert loaded.img_size == model_kwds["img_size"]
    assert loaded.patch_size == model_kwds["patch_size"]
    assert loaded.in_chans == model_kwds["in_chans"]
    assert loaded.out_chans == model_kwds["out_chans"]
    assert loaded.embed_dim == model_kwds["embed_dim"]

    invar = torch.randn(2, 13, 6, 32, 64).to(device)
    with torch.no_grad():
        out_model = model(invar)
        out_loaded = loaded(invar)
    assert common.compare_output(out_model, out_loaded, rtol=1e-5, atol=1e-5)
    del model, loaded, invar
    torch.cuda.empty_cache()


@common.check_ort_version()
def test_swinrnn_deploy(device):
    """Test SwinRNN deployment support"""

    if device == "cpu":
        pytest.skip("CUDA only")

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
