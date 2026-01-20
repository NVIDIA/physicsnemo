# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
from test import common


def test_fully_connected_forward(device):
    """Test fully-connected forward pass with non-regression reference data."""
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
    assert common.validate_forward_accuracy(
        model, (invar,), file_name="models/mlp/data/fullyconnected_output.pth"
    )


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_fully_connected_constructor(device, config):
    """Test FullyConnected constructor and verify public attributes."""
    if config == "default":
        # Test with all default arguments except required ones
        model = FullyConnected(
            in_features=32,
            out_features=16,
        ).to(device)

        # Verify default attributes
        assert model.in_features == 32
        assert model.out_features == 16
        assert model.skip_connections is False
        assert len(model.layers) == 6  # Default num_layers

    else:
        # Test with non-default arguments
        model = FullyConnected(
            in_features=64,
            out_features=32,
            layer_size=128,
            num_layers=4,
            activation_fn="relu",
            skip_connections=True,
            adaptive_activations=False,
            weight_norm=True,
            weight_fact=False,
        ).to(device)

        # Verify custom attributes
        assert model.in_features == 64
        assert model.out_features == 32
        assert model.skip_connections is True
        assert len(model.layers) == 4


def test_fully_connected_weight_norm_fact_exclusive(device):
    """Test that weight_norm and weight_fact cannot both be True."""
    with pytest.raises(
        ValueError,
        match="Cannot apply both weight normalization and weight factorization together",
    ):
        FullyConnected(
            in_features=16,
            out_features=16,
            weight_norm=True,
            weight_fact=True,
        )


def test_fully_connected_optims(device):
    """Test fully-connected optimizations."""

    def setup_model():
        """Set up fresh model and inputs for each optim test."""
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


def test_fully_connected_checkpoint(device):
    """Test fully-connected checkpoint save/load and verify attributes after loading."""
    # Construct FC model with specific configuration
    model_1 = FullyConnected(
        in_features=4,
        out_features=4,
        num_layers=2,
        layer_size=8,
        skip_connections=True,
    ).to(device)

    model_2 = FullyConnected(
        in_features=4,
        out_features=4,
        num_layers=2,
        layer_size=8,
        skip_connections=True,
    ).to(device)

    bsize = random.randint(1, 16)
    invar = torch.randn(bsize, 4).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))


def test_fully_connected_checkpoint_attributes(device):
    """Test that checkpoint loading preserves model attributes."""
    import tempfile
    from pathlib import Path

    from physicsnemo.core import Module

    # Create model with specific configuration
    original_model = FullyConnected(
        in_features=16,
        out_features=8,
        num_layers=3,
        layer_size=32,
        skip_connections=True,
    ).to(device)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_fc.mdlus"
        original_model.save(str(checkpoint_path))

        # Load from checkpoint
        loaded_model = Module.from_checkpoint(str(checkpoint_path)).to(device)

        # Verify attributes are preserved
        assert loaded_model.skip_connections == original_model.skip_connections
        assert len(loaded_model.layers) == len(original_model.layers)

        # Verify outputs match
        torch.manual_seed(42)
        invar = torch.randn(4, 16).to(device)
        with torch.no_grad():
            original_output = original_model(invar)
            loaded_output = loaded_model(invar)
        assert torch.allclose(original_output, loaded_output)


@common.check_ort_version()
def test_fully_connected_deploy(device):
    """Test fully-connected deployment support."""
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
