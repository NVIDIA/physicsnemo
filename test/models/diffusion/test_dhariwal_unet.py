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

import pytest
import torch

from physicsnemo.models.diffusion_unets import DhariwalUNet as UNet
from test import common


def test_dhariwal_unet_forward(device):
    torch.manual_seed(0)
    model = UNet(img_resolution=64, in_channels=2, out_channels=2).to(device)
    input_image = torch.ones([1, 2, 64, 64]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels),
        file_name="models/diffusion/data/dhariwal_unet_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_dhariwal_unet_constructor(device, config):
    """Test DhariwalUNet model constructor and attributes (MOD-008a).

    This test verifies:
    1. Model can be instantiated with default arguments
    2. Model can be instantiated with custom arguments
    3. All public attributes have expected values
    """
    if config == "default":
        model = UNet(
            img_resolution=16,
            in_channels=2,
            out_channels=2,
        ).to(device)

        # Verify default attribute values
        assert model.label_dim == 0
        assert model.augment_dim == 0
        assert model.label_dropout == 0.0
        assert model.profile_mode is False
        assert model.amp_mode is False

        # Verify forward pass shape
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)
        output_image = model(input_image, noise_labels, class_labels)
        assert output_image.shape == (1, 2, 16, 16)
    else:
        model = UNet(
            img_resolution=16,
            in_channels=2,
            out_channels=2,
            label_dim=10,
            augment_dim=5,
            model_channels=64,
            channel_mult=[1, 2],
            channel_mult_emb=2,
            num_blocks=2,
            attn_resolutions=[8],
            dropout=0.05,
            label_dropout=0.2,
        ).to(device)

        # Verify custom attribute values
        assert model.label_dim == 10
        assert model.augment_dim == 5
        assert model.label_dropout == 0.2
        assert model.profile_mode is False
        assert model.amp_mode is False

        # Verify forward pass shape
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randn(1, 10).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)
        output_image = model(input_image, noise_labels, class_labels)
        assert output_image.shape == (1, 2, 16, 16)

    # Common assertions
    assert isinstance(model, UNet)
    assert hasattr(model, "enc")
    assert hasattr(model, "dec")
    assert hasattr(model, "out_norm")
    assert hasattr(model, "out_conv")
    assert hasattr(model, "meta")


# Skip CPU tests because too slow
def test_dhariwal_unet_optims(device):
    """Test Dhariwal UNet optimizations"""

    if device == "cpu":
        pytest.skip("CUDA only")

    def setup_model():
        model = UNet(
            img_resolution=8,
            in_channels=2,
            out_channels=2,
        ).to(device)
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 8, 8]).to(device)

        return model, [input_image, noise_labels, class_labels]

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))

    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP with amp_mode=True for the layers: should pass
    model, invar = setup_model()
    model.amp_mode = True
    assert common.validate_amp(model, (*invar,))
    # Check Combo with amp_mode=True for the layers: should pass
    model, invar = setup_model()
    model.amp_mode = True
    assert common.validate_combo_optims(model, (*invar,))

    # Check failures (only on GPU, because validate_amp and validate_combo_optims
    # don't activate amp for DhariwalUNet on CPU)
    if device == "cuda:0":
        # Check AMP: should fail because amp_mode is False for the layers
        with pytest.raises(RuntimeError):
            model, invar = setup_model()
            assert common.validate_amp(model, (*invar,))
        # Check Combo: should fail because amp_mode is False for the layers
        # NOTE: this test doesn't fail because validate_combo_optims doesn't
        # activate amp for DhariwalUNet, even on GPU
        # with pytest.raises(RuntimeError):
        #     model, invar = setup_model()
        #     assert common.validate_combo_optims(model, (*invar,))


# Skip CPU tests because too slow
def test_dhariwal_unet_checkpoint(device):
    """Test Dhariwal UNet checkpoint save/load"""

    if device == "cpu":
        pytest.skip("CUDA only")

    model_1 = UNet(
        img_resolution=16,
        in_channels=2,
        out_channels=2,
    ).to(device)

    model_2 = UNet(
        img_resolution=16,
        in_channels=2,
        out_channels=2,
    ).to(device)
    # This test doesn't like the model outputs to be the same.
    # Change the bias in the last layer of the second model as a hack
    # Because this model is initialized with all zeros
    with torch.no_grad():
        model_2.out_conv.bias.add_(1)

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    assert common.validate_checkpoint(
        model_1, model_2, (*[input_image, noise_labels, class_labels],)
    )


@common.check_ort_version()
def test_dhariwal_unet_deploy(device):
    """Test Dhariwal UNet deployment support"""
    model = UNet(
        img_resolution=16,
        in_channels=2,
        out_channels=2,
    ).to(device)

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)

    assert common.validate_onnx_export(
        model, (*[input_image, noise_labels, class_labels],)
    )
    assert common.validate_onnx_runtime(
        model, (*[input_image, noise_labels, class_labels],)
    )
