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

import pytest
import torch

from test import common
from test.conftest import requires_module


@requires_module(["transformer_engine"])
def test_unet_forward(device, pytestconfig):
    """Test UNet forward pass with non-regression reference data."""
    from physicsnemo.models.unet import UNet

    torch.manual_seed(0)
    model = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=3,
        feature_map_channels=[8, 8, 16, 16, 32, 32],
        num_conv_blocks=2,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
    assert common.validate_forward_accuracy(
        model, (invar,), file_name="models/unet/data/unet_output.pth"
    )


@requires_module(["transformer_engine"])
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_unet_constructor_and_forward(device, config, pytestconfig):
    """Test UNet constructor and verify public attributes."""
    from physicsnemo.models.unet import UNet

    if config == "default":
        # Test with minimal required arguments
        model = UNet(
            in_channels=1,
            out_channels=1,
            model_depth=3,
            feature_map_channels=[4, 4, 8, 8, 16, 16],
        ).to(device)

        # Verify default attributes
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.use_attn_gate is False
        assert model.gradient_checkpointing is True

    else:
        # Test with non-default arguments
        model = UNet(
            in_channels=2,
            out_channels=2,
            model_depth=2,
            feature_map_channels=[8, 8, 16, 16],
            pooling_type="AvgPool3d",
            normalization="batchnorm",
            gradient_checkpointing=False,
        ).to(device)

        # Verify custom attributes
        assert model.in_channels == 2
        assert model.out_channels == 2
        assert model.use_attn_gate is False
        assert model.gradient_checkpointing is False

    # Verify model produces correct output shape
    bsize = 2
    in_channels = 1 if config == "default" else 2
    out_channels = 1 if config == "default" else 2
    invar = torch.randn(bsize, in_channels, 8, 8, 8).to(device)
    outvar = model(invar)
    assert outvar.shape == (bsize, out_channels, *invar.shape[2:])


@requires_module(["transformer_engine"])
@pytest.mark.parametrize("pooling_type", ["MaxPool3d", "AvgPool3d"])
@pytest.mark.parametrize("model_depth", [2, 3])
def test_unet_pooling_and_depth(device, model_depth, pooling_type, pytestconfig):
    """Test UNet with different pooling types and depths."""
    from physicsnemo.models.unet import UNet

    if model_depth == 2:
        feature_map_channels = [4, 4, 8, 8]
    else:
        feature_map_channels = [4, 4, 8, 8, 16, 16]

    model = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=model_depth,
        feature_map_channels=feature_map_channels,
        pooling_type=pooling_type,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 1, 8, 8, 8).to(device)
    outvar = model(invar)
    assert outvar.shape == (bsize, 1, *invar.shape[2:])


@requires_module(["transformer_engine"])
def test_unet_checkpoint(device, pytestconfig):
    """Test UNet checkpoint save/load."""
    from physicsnemo.models.unet import UNet

    model_1 = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=2,
        feature_map_channels=[4, 4, 8, 8],
        num_conv_blocks=2,
    ).to(device)

    model_2 = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=2,
        feature_map_channels=[4, 4, 8, 8],
        num_conv_blocks=2,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 1, 8, 8, 8).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))


@requires_module(["transformer_engine"])
def test_unet_checkpoint_attributes(device, pytestconfig):
    """Test that checkpoint loading preserves model attributes."""
    import tempfile
    from pathlib import Path

    from physicsnemo.core import Module
    from physicsnemo.models.unet import UNet

    # Create model with specific configuration
    original_model = UNet(
        in_channels=2,
        out_channels=2,
        model_depth=2,
        feature_map_channels=[4, 4, 8, 8],
        gradient_checkpointing=False,
    ).to(device)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_unet.mdlus"
        original_model.save(str(checkpoint_path))

        # Load from checkpoint
        loaded_model = Module.from_checkpoint(str(checkpoint_path)).to(device)

        # Verify attributes are preserved
        assert loaded_model.use_attn_gate == original_model.use_attn_gate
        assert (
            loaded_model.gradient_checkpointing == original_model.gradient_checkpointing
        )

        # Verify outputs match
        torch.manual_seed(42)
        invar = torch.randn(2, 2, 8, 8, 8).to(device)
        with torch.no_grad():
            original_output = original_model(invar)
            loaded_output = loaded_model(invar)
        assert torch.allclose(original_output, loaded_output)
