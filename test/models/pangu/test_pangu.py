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

import torch

import physicsnemo
from physicsnemo.models.pangu import Pangu
from test import common


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
        assert common.validate_forward_accuracy(
            model, (invar,), atol=5e-3, file_name="models/pangu/data/pangu_output.pth"
        )

    del model, invar
    torch.cuda.empty_cache()


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
        assert model.img_size == kw_args["img_size"]
        assert model.patch_size == kw_args["patch_size"]
        assert model.embed_dim == kw_args["embed_dim"]
        assert model.in_channels == 72
        assert model.surface_input_channels == 7
        assert model.upper_air_channels == 5
        assert model.upper_air_levels == 13

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


def test_pangu_checkpoint(device, tmp_path):
    """Test Pangu checkpoint save/load."""
    model_kwds = {
        "img_size": (32, 32),
        "patch_size": (2, 4, 4),
        "embed_dim": 64,
        "num_heads": (4, 8, 8, 4),
        "window_size": (2, 4, 4),
    }

    model_1 = Pangu(**model_kwds).to(device).eval()
    model_2 = Pangu(**model_kwds).to(device).eval()

    bsize = random.randint(1, 2)
    invar_surface = torch.randn(bsize, 4, 32, 32).to(device)
    invar_surface_mask = torch.randn(3, 32, 32).to(device)
    invar_upper_air = torch.randn(bsize, 5, 13, 32, 32).to(device)
    invar = model_1.prepare_input(invar_surface, invar_surface_mask, invar_upper_air)

    # Checkpoint roundtrip checks are run in eval mode to avoid stochastic-depth
    # randomness in this architecture.
    with torch.no_grad():
        out_model_1 = model_1(invar)
        out_model_2 = model_2(invar)
    assert not common.compare_output(out_model_1, out_model_2, rtol=1e-5, atol=1e-5)

    checkpoint_path = tmp_path / "pangu_checkpoint_roundtrip.mdlus"
    model_1.save(str(checkpoint_path))

    # Validate explicit load on an existing model instance.
    model_2.load(str(checkpoint_path))
    model_2.eval()
    with torch.no_grad():
        out_loaded = model_2(invar)
    assert common.compare_output(out_model_1, out_loaded, rtol=1e-5, atol=1e-5)

    # Validate class reconstruction via Module.from_checkpoint.
    from_checkpoint = physicsnemo.Module.from_checkpoint(str(checkpoint_path)).to(
        device
    )
    from_checkpoint.eval()
    with torch.no_grad():
        out_from_checkpoint = from_checkpoint(invar)
    assert common.compare_output(out_model_1, out_from_checkpoint, rtol=1e-5, atol=1e-5)

    del model_1, model_2, from_checkpoint, invar
    torch.cuda.empty_cache()


def test_pangu_load_checkpoint(device, tmp_path):
    """Test Pangu loading from a saved checkpoint path."""
    model_kwds = {
        "img_size": (32, 32),
        "patch_size": (2, 4, 4),
        "embed_dim": 64,
        "num_heads": (4, 8, 8, 4),
        "window_size": (2, 4, 4),
    }
    model = Pangu(**model_kwds).to(device).eval()
    checkpoint_path = tmp_path / "pangu_checkpoint.mdlus"
    model.save(str(checkpoint_path))

    loaded = physicsnemo.Module.from_checkpoint(str(checkpoint_path)).to(device).eval()
    assert loaded.img_size == model_kwds["img_size"]
    assert loaded.patch_size == model_kwds["patch_size"]
    assert loaded.embed_dim == model_kwds["embed_dim"]

    bsize = 2
    invar_surface = torch.randn(bsize, 4, 32, 32).to(device)
    invar_surface_mask = torch.randn(3, 32, 32).to(device)
    invar_upper_air = torch.randn(bsize, 5, 13, 32, 32).to(device)
    invar = loaded.prepare_input(invar_surface, invar_surface_mask, invar_upper_air)

    with torch.no_grad():
        out_model = model(invar)
        out_loaded = loaded(invar)
    assert common.compare_output(out_model, out_loaded, rtol=1e-5, atol=1e-5)
    del model, loaded, invar
    torch.cuda.empty_cache()


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
