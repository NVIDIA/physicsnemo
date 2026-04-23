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

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.flare import FLARE
from test.common import (
    check_ort_version,
    validate_amp,
    validate_checkpoint,
    validate_combo_optims,
    validate_cuda_graphs,
    validate_forward_accuracy,
    validate_jit,
    validate_onnx_export,
    validate_onnx_runtime,
)


@pytest.mark.parametrize(
    "config",
    ["default_structured", "custom_irregular"],
    ids=["with_defaults_structured", "with_custom_irregular"],
)
def test_flare_constructor(config):
    """Test FLARE model constructor and attributes."""
    if config == "default_structured":
        model = FLARE(
            functional_dim=3,
            out_dim=1,
            structured_shape=(64, 64),
            unified_pos=True,
        )
        assert model.n_hidden == 256, "Default n_hidden should be 256"
        assert model.time_input is False, "Default time_input should be False"
        assert model.unified_pos is True
        assert model.structured_shape == (64, 64)
        assert model.embedding_dim == 64  # ref * ref = 8 * 8 = 64
        assert len(model.blocks) == 4, "Default n_layers should be 4"
    else:
        model = FLARE(
            functional_dim=2,
            out_dim=4,
            embedding_dim=3,
            n_layers=8,
            n_hidden=64,
            dropout=0.1,
            n_head=4,
            act="gelu",
            mlp_ratio=2,
            slice_num=16,
            unified_pos=False,
            structured_shape=None,
            time_input=True,
        )
        assert model.n_hidden == 64
        assert model.time_input is True
        assert model.unified_pos is False
        assert model.structured_shape is None
        assert model.embedding_dim == 3
        assert len(model.blocks) == 8

    assert isinstance(model, Module), "FLARE should inherit from physicsnemo.Module"
    assert hasattr(model, "preprocess"), "Model should have preprocess MLP"
    assert hasattr(model, "blocks"), "Model should have transformer blocks"
    assert hasattr(model, "meta"), "Model should have metadata"


def test_flare_2d_forward(device):
    """Test FLARE 2D forward pass"""
    torch.manual_seed(0)
    model = FLARE(
        structured_shape=(85, 85),
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=1,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=True,
    ).to(device)

    bsize = 4

    fx = torch.randn(bsize, 85 * 85, 1).to(device)
    embedding = torch.randn(bsize, 85, 85).to(device)

    assert validate_forward_accuracy(
        model,
        (
            fx,
            embedding,
        ),
        file_name="experimental/models/flare/data/flare_2d_output.pth",
        atol=2e-3,
    )


def test_flare_irregular_forward(device):
    """Test FLARE irregular forward pass"""
    torch.manual_seed(0)
    model = FLARE(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=False,
    ).to(device)

    bsize = 4

    embedding = torch.randn(bsize, 12345, 3).to(device)
    functional_input = torch.randn(bsize, 12345, 2).to(device)

    assert validate_forward_accuracy(
        model,
        (
            embedding,
            functional_input,
        ),
        file_name="experimental/models/flare/data/flare_irregular_output.pth",
        atol=1e-3,
    )


def test_flare_optims(device):
    """Test FLARE optimizations"""

    def setup_model():
        """Set up fresh FLARE model and inputs for each optim test"""

        model = FLARE(
            structured_shape=None,
            n_layers=8,
            n_hidden=64,
            dropout=0,
            n_head=4,
            time_input=False,
            act="gelu",
            mlp_ratio=1,
            functional_dim=2,
            embedding_dim=3,
            out_dim=1,
            slice_num=32,
            ref=1,
            unified_pos=False,
        ).to(device)

        if device == "cuda:0":
            bsize = 4
            n_points = 12345
        else:
            bsize = 1
            n_points = 123

        embedding = torch.randn(bsize, n_points, 3).to(device)
        functional_input = torch.randn(bsize, n_points, 2).to(device)

        return model, embedding, functional_input

    # Ideally always check graphs first
    model, pos, invar = setup_model()
    assert validate_cuda_graphs(
        model,
        (
            pos,
            invar,
        ),
    )

    # Check JIT
    model, pos, invar = setup_model()
    assert validate_jit(
        model,
        (
            pos,
            invar,
        ),
    )
    # Check AMP
    model, pos, invar = setup_model()
    assert validate_amp(
        model,
        (
            pos,
            invar,
        ),
    )
    # Check Combo
    model, pos, invar = setup_model()
    assert validate_combo_optims(
        model,
        (
            pos,
            invar,
        ),
    )


def test_flare_checkpoint(device):
    """Test FLARE checkpoint save/load"""
    model_1 = FLARE(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=False,
    ).to(device)

    model_2 = FLARE(
        structured_shape=None,
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=False,
    ).to(device)

    bsize = random.randint(1, 2)

    embedding = torch.randn(bsize, 12345, 3).to(device)
    functional_input = torch.randn(bsize, 12345, 2).to(device)

    assert validate_checkpoint(
        model_1,
        model_2,
        (
            functional_input,
            embedding,
        ),
    )


@check_ort_version()
def test_flare_deploy(device):
    """Test FLARE deployment support"""
    model = FLARE(
        structured_shape=(85, 85),
        n_layers=8,
        n_hidden=64,
        dropout=0,
        n_head=4,
        time_input=False,
        act="gelu",
        mlp_ratio=1,
        functional_dim=1,
        out_dim=1,
        slice_num=32,
        ref=1,
        unified_pos=True,
    ).to(device)

    bsize = 4

    pos = torch.randn(bsize, 85 * 85, 1).to(device)
    invar = torch.randn(bsize, 85, 85).to(device)

    assert validate_onnx_export(
        model,
        (
            pos,
            invar,
        ),
    )
    assert validate_onnx_runtime(
        model,
        (
            invar,
            invar,
        ),
        1e-2,
        1e-2,
    )
