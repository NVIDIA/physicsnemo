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

from physicsnemo.models.diffusion_unets import SongUNet as UNet
from test import common


def test_song_unet_forward(device):
    torch.manual_seed(0)
    # Construct the DDM++ UNet model
    model = UNet(img_resolution=64, in_channels=2, out_channels=2).to(device)
    input_image = torch.ones([1, 2, 64, 64]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels),
        file_name="models/diffusion/data/ddmpp_unet_output.pth",
        atol=1e-3,
    )

    torch.manual_seed(0)
    # Construct the NCSN++ UNet model
    model = UNet(
        img_resolution=64,
        in_channels=2,
        out_channels=2,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
    ).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels),
        file_name="models/diffusion/data/ncsnpp_unet_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_song_unet_constructor(device, config):
    """Test SongUNet model constructor and attributes (MOD-008a).

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
        assert model.img_resolution == 16
        assert model.img_shape_y == 16
        assert model.img_shape_x == 16
        assert model.label_dim == 0
        assert model.augment_dim == 0
        assert model.label_dropout == 0.0
        assert model.embedding_type == "positional"
        assert model.emb_channels == 128 * 4  # model_channels * channel_mult_emb
        assert model.additive_pos_embed is False
        assert model.use_apex_gn is False
        assert model.profile_mode is False
        assert model.amp_mode is False

        # Verify forward pass shape
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)
        output_image = model(input_image, noise_labels, class_labels)
        assert output_image.shape == (1, 2, 16, 16)
    else:
        model_channels = 64
        model = UNet(
            img_resolution=[16, 32],
            in_channels=2,
            out_channels=2,
            label_dim=10,
            augment_dim=5,
            model_channels=model_channels,
            channel_mult=[1, 2, 2],
            channel_mult_emb=2,
            num_blocks=2,
            attn_resolutions=[8],
            dropout=0.05,
            label_dropout=0.1,
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            decoder_type="standard",
            resample_filter=[1, 3, 3, 1],
            additive_pos_embed=True,
            bottleneck_attention=False,
        ).to(device)

        # Verify custom attribute values
        assert model.img_resolution == [16, 32]
        assert model.img_shape_y == 16
        assert model.img_shape_x == 32
        assert model.label_dim == 10
        assert model.augment_dim == 5
        assert model.label_dropout == 0.1
        assert model.embedding_type == "fourier"
        assert model.emb_channels == model_channels * 2
        assert model.additive_pos_embed is True
        assert model.spatial_emb.shape == (1, model_channels, 16, 32)
        assert model.profile_mode is False
        assert model.amp_mode is False

        # Verify forward pass shape
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randn(1, 10).to(device)
        input_image = torch.ones([1, 2, 16, 32]).to(device)
        output_image = model(input_image, noise_labels, class_labels)
        assert output_image.shape == (1, 2, 16, 32)

    # Common assertions
    assert isinstance(model, UNet)
    assert hasattr(model, "enc")
    assert hasattr(model, "dec")
    assert hasattr(model, "meta")


def test_song_unet_constructor_failure_cases():
    """Test SongUNet constructor with invalid arguments."""
    with pytest.raises(ValueError):
        UNet(img_resolution=16, in_channels=2, out_channels=2, embedding_type=None)

    with pytest.raises(ValueError):
        UNet(img_resolution=16, in_channels=2, out_channels=2, encoder_type=None)

    with pytest.raises(ValueError):
        UNet(img_resolution=16, in_channels=2, out_channels=2, decoder_type=None)


def test_song_unet_optims(device):
    """Test Song UNet optimizations"""

    def setup_model():
        model = UNet(
            img_resolution=16,
            in_channels=2,
            out_channels=2,
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            resample_filter=[1, 3, 3, 1],
        ).to(device)
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)

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
    # don't activate amp for SongUNet on CPU)
    if device == "cuda:0":
        # Check AMP: should fail because amp_mode is False for the layers
        with pytest.raises(RuntimeError):
            model, invar = setup_model()
            assert common.validate_amp(model, (*invar,))
        # Check Combo: should fail because amp_mode is False for the layers
        # NOTE: this test doesn't fail because validate_combo_optims doesn't
        # activate amp for SongUNet
        # model, invar = setup_model()
        # with pytest.raises(RuntimeError):
        #     model, invar = setup_model()
        #     assert common.validate_combo_optims(model, (*invar,))

    # Check fullgraph compilation
    # run only on GPU
    if device == "cuda:0":
        model, invar = setup_model()
        assert common.validate_torch_compile(
            model, (*invar,), fullgraph=True, error_on_recompile=True
        )


def test_song_unet_checkpoint(device):
    """Test Song UNet checkpoint save/load"""
    # Construct FNO models
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

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    assert common.validate_checkpoint(
        model_1, model_2, (*[input_image, noise_labels, class_labels],)
    )


@common.check_ort_version()
def test_song_unet_deploy(device):
    """Test Song UNet deployment support"""
    model = UNet(
        img_resolution=16,
        in_channels=2,
        out_channels=2,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
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


def test_song_unet_grad_checkpointing(device):
    channels = 2
    img_resolution = 64

    # fix random seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    input_image = torch.ones([1, channels, img_resolution, img_resolution]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    # Construct the DDM++ UNet model
    model = UNet(
        img_resolution=img_resolution, in_channels=channels, out_channels=channels
    ).to(device)
    y_pred = model(input_image, noise_labels, class_labels)

    # dummy loss
    loss = y_pred.sum()

    # compute gradients
    loss.backward()
    computed_grads = {}
    for name, param in model.named_parameters():
        computed_grads[name] = param.grad.clone()

    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    input_image = torch.ones([1, channels, img_resolution, img_resolution]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    # Model with checkpointing enabled
    model_checkpointed = UNet(
        img_resolution=img_resolution,
        in_channels=channels,
        out_channels=channels,
        checkpoint_level=4,
    ).to(device)
    y_pred_checkpointed = model_checkpointed(input_image, noise_labels, class_labels)

    # dummy loss
    loss = y_pred_checkpointed.sum()

    # compute gradients
    loss.backward()
    computed_grads_checkpointed = {}
    for name, param in model.named_parameters():
        computed_grads_checkpointed[name] = param.grad.clone()

    # Check that the results are the same
    assert torch.allclose(y_pred_checkpointed, y_pred), (
        "Outputs do not match. Checkpointing failed!"
    )

    # Compare the gradients
    for name in computed_grads:
        (
            torch.allclose(computed_grads_checkpointed[name], computed_grads[name]),
            "Gradient do not match. Checkpointing failed!",
        )
