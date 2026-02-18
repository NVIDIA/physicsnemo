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

# ruff: noqa: E402
from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet, StormCastUNet
from test import common


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_corrdiff_regression_unet_constructor(device, config):
    """Test CorrDiffRegressionUNet model constructor and attributes (MOD-008a).

    This test verifies:
    1. Model can be instantiated with default arguments
    2. Model can be instantiated with custom arguments
    3. All public attributes have expected values
    """
    if config == "default":
        model = CorrDiffRegressionUNet(
            img_resolution=16,
            img_in_channels=2,
            img_out_channels=3,
            model_type="SongUNet",
        ).to(device)

        # Verify default attribute values
        assert model.img_shape_y == 16
        assert model.img_shape_x == 16
        assert model.img_in_channels == 2
        assert model.img_out_channels == 3
        assert model.use_fp16 is False
        assert model.amp_mode is False
        assert model.profile_mode is False

        # Verify forward pass shape
        x = torch.zeros(1, 2, 16, 16).to(device)
        img_lr = torch.randn(1, 3, 16, 16).to(device)
        output = model(x, img_lr)
        assert output.shape == (1, 3, 16, 16)
    else:
        model = CorrDiffRegressionUNet(
            img_resolution=[16, 32],
            img_in_channels=4,
            img_out_channels=2,
            model_type="SongUNet",
            use_fp16=False,
            model_channels=64,
            channel_mult=[1, 2, 2],
            num_blocks=2,
        ).to(device)

        # Verify custom attribute values
        assert model.img_shape_y == 16
        assert model.img_shape_x == 32
        assert model.img_in_channels == 4
        assert model.img_out_channels == 2
        assert model.use_fp16 is False

        # Verify forward pass shape
        x = torch.zeros(1, 4, 16, 32).to(device)
        img_lr = torch.randn(1, 2, 16, 32).to(device)
        output = model(x, img_lr)
        assert output.shape == (1, 2, 16, 32)

    # Common assertions
    assert isinstance(model, Module)
    assert hasattr(model, "model")
    assert hasattr(model, "meta")


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_stormcast_unet_constructor(device, config):
    """Test StormCastUNet model constructor and attributes (MOD-008a).

    This test verifies:
    1. Model can be instantiated with default arguments
    2. Model can be instantiated with custom arguments
    3. All public attributes have expected values
    """
    if config == "default":
        model = StormCastUNet(
            img_resolution=16,
            img_in_channels=2,
            img_out_channels=3,
        ).to(device)

        # Verify default attribute values
        assert model.img_shape_y == 16
        assert model.img_shape_x == 16
        assert model.img_in_channels == 2
        assert model.img_out_channels == 3
        assert model.use_fp16 is False
        assert model.sigma_min == 0
        assert model.sigma_max == float("inf")
        assert model.sigma_data == 0.5
        assert model.amp_mode is False
        assert model.profile_mode is False

        # Verify forward pass shape
        x = torch.randn(1, 2, 16, 16).to(device)
        output = model(x)
        assert output.shape == (1, 3, 16, 16)
    else:
        model = StormCastUNet(
            img_resolution=[16, 32],
            img_in_channels=4,
            img_out_channels=2,
            sigma_min=0.01,
            sigma_max=80.0,
            sigma_data=1.0,
            model_channels=64,
            channel_mult=[1, 2, 2],
            num_blocks=2,
        ).to(device)

        # Verify custom attribute values
        assert model.img_shape_x == 16
        assert model.img_shape_y == 32
        assert model.img_in_channels == 4
        assert model.img_out_channels == 2
        assert model.sigma_min == 0.01
        assert model.sigma_max == 80.0
        assert model.sigma_data == 1.0

        # Verify forward pass shape
        x = torch.randn(1, 4, 16, 32).to(device)
        output = model(x)
        assert output.shape == (1, 2, 16, 32)

    # Common assertions
    assert isinstance(model, Module)
    assert hasattr(model, "model")
    assert hasattr(model, "meta")


def test_unet_forwards(device):
    """Test forward passes of UNet wrappers"""

    # Construct the UNet model
    res, inc, outc = 64, 2, 3
    model = CorrDiffRegressionUNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)
    input_image = torch.ones([1, inc, res, res]).to(device)
    lr_image = torch.randn([1, outc, res, res]).to(device)
    output = model(x=input_image, img_lr=lr_image)
    assert output.shape == (1, outc, res, res)

    # Construct the StormCastUNet model
    model = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)
    input_image = torch.ones([1, inc, res, res]).to(device)
    output = model(x=input_image)
    assert output.shape == (1, outc, res, res)


def test_unet_fp16_forwards(device):
    """Test forward passes of UNet wrappers with fp16"""

    # Construct the UNet model
    res, inc, outc = 64, 2, 3
    model_fp16 = CorrDiffRegressionUNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
        use_fp16=True,
    ).to(device)

    model_fp32 = CorrDiffRegressionUNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
        use_fp16=False,
    ).to(device)

    input_image = torch.ones([1, inc, res, res]).to(device)
    lr_image = torch.randn([1, outc, res, res]).to(device)
    output_fp16 = model_fp16(x=input_image, img_lr=lr_image)
    output_fp32 = model_fp32(x=input_image, img_lr=lr_image)

    assert output_fp16.shape == (1, outc, res, res)
    assert torch.allclose(output_fp16, output_fp32, rtol=1e-3, atol=1e-3), (
        "FP16 and FP32 outputs differ more than allowed"
    )

    # Construct the StormCastUNet model
    model = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)
    input_image = torch.ones([1, inc, res, res]).to(device)
    output = model(x=input_image)
    assert output.shape == (1, outc, res, res)


def test_unet_optims(device):
    """Test optimizations of U-Net wrappers"""

    res, inc, outc = 64, 2, 3

    def setup_model():
        model = CorrDiffRegressionUNet(
            img_resolution=res,
            img_in_channels=inc,
            img_out_channels=outc,
            model_type="SongUNet",
        ).to(device)
        input_image = torch.ones([1, inc, res, res]).to(device)
        lr_image = torch.randn([1, outc, res, res]).to(device)
        return model, [input_image, lr_image]

    #  Check AMP: with amp_mode=True for the layers, should pass
    model, invar = setup_model()
    model.amp_mode = True
    assert common.validate_amp(model, (*invar,))

    # Check failures (only on GPU, because validate_amp doesn't activate amp on
    # CPU)
    if device == "cuda:0":
        # Check AMP: should fail because amp_mode=False for the layers
        model, invar = setup_model()
        with pytest.raises(RuntimeError):
            assert common.validate_amp(model, (*invar,))

    def setup_model():
        model = StormCastUNet(
            img_resolution=res, img_in_channels=inc, img_out_channels=outc
        ).to(device)
        input_image = torch.ones([1, inc, res, res]).to(device)
        return model, [input_image]

    # Check AMP: with amp_mode=True for the layers, should pass
    model, invar = setup_model()
    model.amp_mode = True
    assert common.validate_amp(model, (*invar,))

    # Check failures (only on GPU, because validate_amp doesn't activate amp on
    # CPU)
    if device == "cuda:0":
        # Check AMP: should fail because amp_mode is False for the layers
        model, invar = setup_model()
        with pytest.raises(RuntimeError):
            assert common.validate_amp(model, (*invar,))


def test_unet_checkpoint(device):
    """Test UNet wrapper checkpoint save/load"""
    # Construct UNet models
    res, inc, outc = 64, 2, 3
    model_1 = CorrDiffRegressionUNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)
    model_2 = CorrDiffRegressionUNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)

    input_image = torch.ones([1, inc, res, res]).to(device)
    lr_image = torch.randn([1, outc, res, res]).to(device)
    assert common.validate_checkpoint(model_1, model_2, (*[input_image, lr_image],))

    # Construct StormCastUNet models
    res, inc, outc = 64, 2, 3
    model_1 = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)
    model_2 = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)

    input_image = torch.ones([1, inc, res, res]).to(device)
    assert common.validate_checkpoint(model_1, model_2, (input_image,))


def test_unet_properties(device):
    """Test UNet wrappers amp_mode and profile_mode properties"""

    res, inc, outc = 32, 1, 1

    model = CorrDiffRegressionUNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)

    # Getter should reflect underlying model value (default False)
    assert model.amp_mode is False

    # Set to True and verify propagation
    model.amp_mode = True
    assert model.amp_mode is True
    if hasattr(model.model, "amp_mode"):
        assert model.model.amp_mode is True
    for sub in model.model.modules():
        if hasattr(sub, "amp_mode"):
            assert sub.amp_mode is True

    # Toggle back to False and verify again
    model.amp_mode = False
    assert model.amp_mode is False
    if hasattr(model.model, "amp_mode"):
        assert model.model.amp_mode is False
    for sub in model.model.modules():
        if hasattr(sub, "amp_mode"):
            assert sub.amp_mode is False

    # Do the same for profile_mode
    # Default value should be False
    assert model.profile_mode is False

    # Set to True and verify propagation
    model.profile_mode = True
    assert model.profile_mode is True
    if hasattr(model.model, "profile_mode"):
        assert model.model.profile_mode is True
    for sub in model.model.modules():
        if hasattr(sub, "profile_mode"):
            assert sub.profile_mode is True

    # Toggle back to False and verify again
    model.profile_mode = False
    assert model.profile_mode is False
    if hasattr(model.model, "profile_mode"):
        assert model.model.profile_mode is False
    for sub in model.model.modules():
        if hasattr(sub, "profile_mode"):
            assert sub.profile_mode is False


def test_unet_backward_compat(device):
    """Test backward compatibility of UNet wrappers"""

    # Construct Load UNet from older version
    CorrDiffRegressionUNet.from_checkpoint(
        file_name=(
            str(
                Path(__file__).parents[1].resolve()
                / Path("diffusion/data")
                / Path("diffusion_unet_0.1.0.mdlus")
            )
        )
    )
