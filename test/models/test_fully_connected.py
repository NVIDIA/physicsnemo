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

from physicsnemo.models.mlp import FullyConnected

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fully_connected_forward(device):
    """Test fully-connected forward pass"""
    torch.manual_seed(0)
    # Construct FC model
    model = FullyConnected(
        in_features=32,
        out_features=8,
        num_layers=1,
        layer_size=8,
    ).to(device)

    bsize = 8
    invar = torch.randn(bsize, 32).to(device)
    assert common.validate_forward_accuracy(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fully_connected_constructor(device):
    """Test fully-connected constructor options"""
    # Define dictionary of constructor args
    arg_list = [
        {
            "in_features": random.randint(1, 16),
            "out_features": random.randint(1, 16),
            "layer_size": 16,
            "num_layers": 2,
            "skip_connections": False,
            "adaptive_activations": False,
            "weight_norm": False,
            "weight_fact": False,
        },
        {
            "in_features": random.randint(1, 16),
            "out_features": random.randint(1, 16),
            "layer_size": 16,
            "num_layers": 4,
            "activation_fn": ["relu", "silu"],
            "skip_connections": True,
            "adaptive_activations": True,
            "weight_norm": True,
            "weight_fact": False,
        },
        {
            "in_features": random.randint(1, 16),
            "out_features": random.randint(1, 16),
            "layer_size": 16,
            "num_layers": 4,
            "activation_fn": ["relu", "silu"],
            "skip_connections": True,
            "adaptive_activations": True,
            "weight_norm": False,
            "weight_fact": True,
        },
        {
            "in_features": random.randint(1, 16),
            "out_features": random.randint(1, 16),
            "layer_size": 16,
            "num_layers": 4,
            "activation_fn": ["relu", "silu"],
            "skip_connections": True,
            "adaptive_activations": True,
            "weight_norm": True,
            "weight_fact": True,
        },
    ]
    for kw_args in arg_list:
        if kw_args["weight_norm"] and kw_args["weight_fact"]:
            # If both weight_norm and weight_fact are True, expect an AssertionError
            with pytest.raises(
                ValueError,
                match="Cannot apply both weight normalization and weight factorization together, please select one.",
            ):
                model = FullyConnected(**kw_args).to(device)

        else:
            # Construct FC model
            model = FullyConnected(**kw_args).to(device)

            bsize = random.randint(1, 16)
            invar = torch.randn(bsize, kw_args["in_features"]).to(device)
            outvar = model(invar)
            assert outvar.shape == (bsize, kw_args["out_features"])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fully_connected_optims(device):
    """Test fully-connected optimizations"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        # Construct FC model
        model = FullyConnected(
            in_features=32,
            out_features=8,
            num_layers=1,
            layer_size=8,
        ).to(device)

        bsize = random.randint(1, 16)
        invar = torch.randn(bsize, 32).to(device)
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
def test_fully_connected_checkpoint(device):
    """Test fully-connected checkpoint save/load"""
    # Construct FC model
    model_1 = FullyConnected(
        in_features=4,
        out_features=4,
        num_layers=2,
        layer_size=8,
    ).to(device)

    model_2 = FullyConnected(
        in_features=4,
        out_features=4,
        num_layers=2,
        layer_size=8,
    ).to(device)

    bsize = random.randint(1, 16)
    invar = torch.randn(bsize, 4).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fully_connected_deploy(device):
    """Test fully-connected deployment support"""
    # Construct AFNO model
    model = FullyConnected(
        in_features=4,
        out_features=4,
        num_layers=2,
        layer_size=8,
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, 4).to(device)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
