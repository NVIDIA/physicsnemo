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

from physicsnemo.models.diffusion_unets import SongUNetPosLtEmbd as UNet
from test import common


def setup_model_learnable_embd(img_resolution, lt_steps, lt_channels, N_pos, seed=0):
    """
    Create a model with similar architecture to CorrDiff (learnable positional
    embeddings, self-attention, learnable lead time embeddings).
    """
    # Smaller architecture variant with learnable positional embeddings
    # (similar to CorrDiff example)
    C_x, C_cond = 4, 3
    attn_res = (
        img_resolution[0] // 4
        if isinstance(img_resolution, list) or isinstance(img_resolution, tuple)
        else img_resolution // 4
    )
    torch.manual_seed(seed)
    model = UNet(
        img_resolution=img_resolution,
        in_channels=C_x + N_pos + C_cond + lt_channels,
        out_channels=C_x,
        model_channels=16,
        channel_mult=[1, 2, 2],
        channel_mult_emb=2,
        num_blocks=2,
        attn_resolutions=[attn_res],
        gridtype="learnable",
        N_grid_channels=N_pos,
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
        prob_channels=[1, 3],
    )
    return model


def generate_data_with_patches(H_p, W_p, device):
    """
    Utility function to generate input data with patches in a consistent way
    accross multiple tests.
    """
    torch.manual_seed(0)
    P, B, C_x, C_cond, lt_steps = 4, 3, 4, 3, 4
    max_offset = 35
    input_image = torch.randn([P * B, C_x + C_cond, H_p, W_p]).to(device)
    noise_label = torch.randn([P * B]).to(device)
    class_label = None
    lead_time_label = torch.randint(0, lt_steps, (B,)).to(device)
    base_grid = torch.stack(
        torch.meshgrid(torch.arange(H_p), torch.arange(W_p), indexing="ij"), dim=0
    )[None].to(device)
    offset = torch.randint(0, max_offset, (P, 2))[:, :, None, None].to(device)
    global_index = base_grid + offset
    return input_image, noise_label, class_label, lead_time_label, global_index


def generate_data_no_patches(H, W, device):
    """
    Utility function to generate input data without patches in a consistent way
    accross multiple tests.
    """
    torch.manual_seed(0)
    B, C_x, C_cond, lt_steps = 3, 4, 3, 4
    input_image = torch.randn([B, C_x + C_cond, H, W]).to(device)
    noise_label = torch.randn([B]).to(device)
    class_label = None
    lead_time_label = torch.randint(0, lt_steps, (B,)).to(device)
    global_index = None
    return input_image, noise_label, class_label, lead_time_label, global_index


def test_song_unet_forward(device):
    torch.manual_seed(0)
    N_pos = 4
    lead_time_channels = 4
    # Construct the DDM++ UNet model
    model = UNet(
        img_resolution=64,
        in_channels=2 + N_pos + lead_time_channels,
        out_channels=2,
        lead_time_channels=lead_time_channels,
    ).to(device)
    input_image = torch.ones([1, 2, 64, 64]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    lead_time_labels = torch.randint(0, 9, (1,)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels, lead_time_labels),
        file_name="models/diffusion/data/ddmpp_unet_output.pth",
        atol=1e-3,
    )

    torch.manual_seed(0)
    # Construct the NCSN++ UNet model
    model = UNet(
        img_resolution=64,
        in_channels=2 + N_pos + lead_time_channels,
        out_channels=2,
        lead_time_channels=4,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
    ).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels, lead_time_labels),
        file_name="models/diffusion/data/ncsnpp_unet_output.pth",
        atol=1e-3,
    )


def test_song_unet_lt_indexing(device):
    torch.manual_seed(0)
    N_pos = 2
    patch_shape_y = 64
    patch_shape_x = 32
    offset_y = 12
    offset_x = 45
    # Construct the DDM++ UNet model
    lead_time_channels = 4
    model = UNet(
        img_resolution=128,
        in_channels=10 + N_pos + lead_time_channels,
        out_channels=10,
        gridtype="test",
        lead_time_channels=lead_time_channels,
        prob_channels=[0, 1, 2, 3],
        N_grid_channels=N_pos,
    ).to(device)
    input_image = torch.ones([1, 10, patch_shape_y, patch_shape_x]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    idx_x = torch.arange(offset_x, offset_x + patch_shape_x)
    idx_y = torch.arange(offset_y, offset_y + patch_shape_y)
    mesh_x, mesh_y = torch.meshgrid(idx_y, idx_x, indexing="ij")
    global_index = torch.stack((mesh_x, mesh_y), dim=0)[None].to(
        device
    )  # (2, patch_shape_y, patch_shape_x)

    # # Define a function to select the embeddings
    def embedding_selector(emb):
        return emb.expand(1, -1, -1, -1)[
            :,
            :,
            offset_y : offset_y + patch_shape_y,
            offset_x : offset_x + patch_shape_x,
        ]

    model.training = True
    output_image_indexing = model(
        input_image,
        noise_labels,
        class_labels,
        lead_time_label=torch.tensor([8]),
        global_index=global_index,
    )
    output_image_selector = model(
        input_image,
        noise_labels,
        class_labels,
        lead_time_label=torch.tensor([8]),
        embedding_selector=embedding_selector,
    )
    assert output_image_indexing.shape == (1, 10, patch_shape_y, patch_shape_x)
    assert torch.allclose(output_image_indexing, output_image_selector, atol=1e-5)

    model.training = False
    output_image_indexing = model(
        input_image,
        noise_labels,
        class_labels,
        lead_time_label=torch.tensor([8]),
        global_index=global_index,
    )
    output_image_selector = model(
        input_image,
        noise_labels,
        class_labels,
        lead_time_label=torch.tensor([8]),
        embedding_selector=embedding_selector,
    )
    assert output_image_indexing.shape == (1, 10, patch_shape_y, patch_shape_x)
    assert torch.allclose(output_image_indexing, output_image_selector, atol=1e-5)


def test_song_unet_global_indexing(device):
    torch.manual_seed(0)
    N_pos = 2
    patch_shape_y = 32
    patch_shape_x = 64
    offset_y = 12
    offset_x = 45
    lead_time_channels = 4
    # Construct the DDM++ UNet model
    model = UNet(
        img_resolution=128,
        in_channels=2 + N_pos + lead_time_channels,
        out_channels=2,
        lead_time_channels=lead_time_channels,
        gridtype="test",
        N_grid_channels=N_pos,
    ).to(device)
    input_image = torch.ones([1, 2, patch_shape_y, patch_shape_x]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    lead_time_labels = torch.randint(0, 9, (1,)).to(device)
    idx_x = torch.arange(offset_x, offset_x + patch_shape_x)
    idx_y = torch.arange(offset_y, offset_y + patch_shape_y)
    mesh_x, mesh_y = torch.meshgrid(idx_y, idx_x, indexing="ij")
    global_index = torch.stack((mesh_x, mesh_y), dim=0)[None].to(
        device
    )  # (2, patch_shape_y, patch_shape_x)

    output_image = model(
        input_image,
        noise_labels,
        class_labels,
        lead_time_label=lead_time_labels,
        global_index=global_index,
    )

    pos_embed = model.positional_embedding_indexing(
        input_image, lead_time_label=lead_time_labels, global_index=global_index
    )
    assert output_image.shape == (1, 2, patch_shape_y, patch_shape_x)
    assert torch.equal(pos_embed[:, :N_pos, :, :], global_index)


def test_song_unet_positional_embedding_indexing_no_patches(device):
    """
    Test for positional_embedding_indexing method. Does not use patches (i.e.
    input image is the entire global image).
    """

    # Common parameters
    B, lt_steps = 3, 4

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_channels = 6, 8
    model = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    inputs = generate_data_no_patches(H, W, device)
    pos_embed = model.positional_embedding_indexing(inputs[0], inputs[4], inputs[3])
    assert pos_embed.shape == (B, N_pos + lt_channels, H, W)
    assert common.validate_tensor_accuracy(
        pos_embed,
        file_name="models/diffusion/data/songunet_pos_lt_embd_pos_embed_indexing_no_patches_corrdiff.pth",
    )
    # TODO: add non-regression tests for other architectures


def test_song_unet_positional_embedding_indexing_with_patches(device):
    """
    Test for positional_embedding_indexing method. Uses patches (i.e. input image
    is only a subset of the global image).
    """

    # Common parameters
    P, B, H_p, W_p, lt_steps = 4, 3, 32, 64, 4

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_channels = 6, 8
    model = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    inputs = generate_data_with_patches(H_p, W_p, device)
    pos_embed = model.positional_embedding_indexing(inputs[0], inputs[4], inputs[3])
    assert pos_embed.shape == (P * B, N_pos + lt_channels, H_p, W_p)
    assert common.validate_tensor_accuracy(
        pos_embed,
        file_name="models/diffusion/data/songunet_pos_lt_embd_pos_embed_indexing_with_patches_corrdiff.pth",
    )
    # TODO: add non-regression tests for other architectures


def test_song_unet_embedding_selector(device):
    torch.manual_seed(0)
    N_pos = 2
    patch_shape_y = 32
    patch_shape_x = 64
    offset_y = 12
    offset_x = 45
    lead_time_channels = 4
    # Construct the DDM++ UNet model
    model = UNet(
        img_resolution=128,
        in_channels=2 + N_pos + lead_time_channels,
        out_channels=2,
        lead_time_channels=lead_time_channels,
        gridtype="test",
        N_grid_channels=N_pos,
    ).to(device)
    input_image = torch.ones([1, 2, patch_shape_y, patch_shape_x]).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    lead_time_labels = torch.randint(0, 9, (1,)).to(device)

    # Expected embeddings should be the same as global_index
    idx_x = torch.arange(offset_x, offset_x + patch_shape_x)
    idx_y = torch.arange(offset_y, offset_y + patch_shape_y)
    mesh_x, mesh_y = torch.meshgrid(idx_y, idx_x, indexing="ij")
    expected_embeds = torch.stack((mesh_x, mesh_y), dim=0)[None].to(
        device
    )  # (2, patch_shape_y, patch_shape_x)

    # Define a function to select the embeddings
    def embedding_selector(emb):
        return emb.expand(1, -1, -1, -1)[
            :,
            :,
            offset_y : offset_y + patch_shape_y,
            offset_x : offset_x + patch_shape_x,
        ]

    output_image = model(
        input_image,
        noise_labels,
        class_labels,
        lead_time_label=lead_time_labels,
        embedding_selector=embedding_selector,
    )
    assert output_image.shape == (1, 2, patch_shape_y, patch_shape_x)

    # Verify that the embeddings are correctly selected
    selected_embeds = model.positional_embedding_selector(
        input_image, embedding_selector, lead_time_label=lead_time_labels
    )
    assert torch.equal(selected_embeds[:, :N_pos, :, :], expected_embeds)


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_song_unet_constructor(device, config):
    """Test SongUNetPosLtEmbd model constructor and attributes (MOD-008a).

    This test verifies:
    1. Model can be instantiated with default arguments
    2. Model can be instantiated with custom arguments
    3. All public attributes have expected values
    """
    if config == "default":
        N_pos = 4
        lead_time_channels = 4
        model = UNet(
            img_resolution=16,
            in_channels=2 + N_pos + lead_time_channels,
            out_channels=2,
            lead_time_channels=lead_time_channels,
        ).to(device)

        # Verify default attribute values
        assert model.img_shape_y == 16
        assert model.img_shape_x == 16
        assert model.gridtype == "sinusoidal"
        assert model.N_grid_channels == 4
        assert model.lead_time_mode is True
        assert model.lead_time_channels == lead_time_channels
        assert model.lead_time_steps == 9
        assert model.prob_channels == []
        assert model.pos_embd is not None
        assert model.pos_embd.shape == (N_pos, 16, 16)
        assert model.lt_embd is not None
        assert model.lt_embd.shape == (9, lead_time_channels, 16, 16)
        assert model.profile_mode is False
        assert model.amp_mode is False

        # Verify forward pass shape
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)
        lead_time_labels = torch.randint(0, 9, (1,)).to(device)
        output_image = model(input_image, noise_labels, class_labels, lead_time_labels)
        assert output_image.shape == (1, 2, 16, 16)
    else:
        N_pos = 8
        lead_time_channels = 6
        lead_time_steps = 5
        model = UNet(
            img_resolution=[16, 32],
            in_channels=2 + N_pos + lead_time_channels,
            out_channels=2,
            gridtype="learnable",
            N_grid_channels=N_pos,
            lead_time_channels=lead_time_channels,
            lead_time_steps=lead_time_steps,
            prob_channels=[0, 1],
            model_channels=64,
            channel_mult=[1, 2, 2],
            num_blocks=2,
        ).to(device)

        # Verify custom attribute values
        assert model.img_shape_y == 16
        assert model.img_shape_x == 32
        assert model.gridtype == "learnable"
        assert model.N_grid_channels == N_pos
        assert model.lead_time_mode is True
        assert model.lead_time_channels == lead_time_channels
        assert model.lead_time_steps == lead_time_steps
        assert model.prob_channels == [0, 1]
        assert model.pos_embd is not None
        assert model.pos_embd.shape == (N_pos, 16, 32)
        assert model.lt_embd is not None
        assert model.lt_embd.shape == (lead_time_steps, lead_time_channels, 16, 32)

        # Verify forward pass shape
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 32]).to(device)
        lead_time_labels = torch.randint(0, lead_time_steps, (1,)).to(device)
        output_image = model(input_image, noise_labels, class_labels, lead_time_labels)
        assert output_image.shape == (1, 2, 16, 32)

    # Common assertions
    assert isinstance(model, UNet)
    assert hasattr(model, "enc")
    assert hasattr(model, "dec")
    assert hasattr(model, "meta")


def test_song_unet_position_embedding(device):
    # build unet
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    # NCSN++
    N_pos = 100
    lead_time_channels = 4
    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels + N_pos + lead_time_channels,
        out_channels=out_channels,
        lead_time_channels=lead_time_channels,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
        gridtype="learnable",
        N_grid_channels=N_pos,
    ).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    lead_time_labels = torch.randint(0, 9, (1,)).to(device)
    output_image = model(input_image, noise_labels, class_labels, lead_time_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)
    assert model.pos_embd.shape == (100, img_resolution, img_resolution)

    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        lead_time_channels=4,
        N_grid_channels=40,
    ).to(device)
    assert model.pos_embd.shape == (40, img_resolution, img_resolution)


def test_fails_if_grid_is_invalid():
    """Test the positional embedding options. "linear" gridtype only support 2 channels, and N_grid_channels in "sinusoidal" should be a factor of 4"""
    img_resolution = 16
    in_channels = 2
    out_channels = 2

    with pytest.raises(ValueError):
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            lead_time_channels=4,
            gridtype="linear",
            N_grid_channels=20,
        )

    with pytest.raises(ValueError):
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            lead_time_channels=4,
            gridtype="sinusoidal",
            N_grid_channels=11,
        )


# Skip CPU tests because too slow
def test_song_unet_optims(device):
    """Test Song UNet optimizations"""

    if device == "cpu":
        pytest.skip("Skip SongUNetPosLtEmbd on cpu")

    def setup_model():
        model = UNet(
            img_resolution=16,
            in_channels=10,
            out_channels=2,
            lead_time_channels=4,
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            resample_filter=[1, 3, 3, 1],
        ).to(device)
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)
        lead_time_labels = torch.randint(0, 9, (1,)).to(device)

        return model, [input_image, noise_labels, class_labels, lead_time_labels]

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
    # don't activate amp for SongUNetPosLtEmbd on CPU)
    if device == "cuda:0":
        # Check AMP: should fail because amp_mode is False for the layers
        with pytest.raises(RuntimeError):
            model, invar = setup_model()
            assert common.validate_amp(model, (*invar,))
        # Check Combo: should fail because amp_mode is False for the layers
        # NOTE: this test doesn't fail because validate_combo_optims doesn't
        # activate amp for SongUNetPosLtEmbd, even on GPU
        # with pytest.raises(RuntimeError):
        #     model, invar = setup_model()
        #     assert common.validate_combo_optims(model, (*invar,))


# Skip CPU tests because too slow
def test_song_unet_checkpoint(device):
    """Test Song UNet checkpoint save/load"""

    if device == "cpu":
        pytest.skip("Skip SongUNetPosLtEmbd on cpu")

    model_1 = UNet(
        img_resolution=16,
        in_channels=10,
        out_channels=2,
        lead_time_channels=4,
    ).to(device)

    model_2 = UNet(
        img_resolution=16,
        in_channels=10,
        out_channels=2,
        lead_time_channels=4,
    ).to(device)

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    lead_time_labels = torch.randint(0, 9, (1,)).to(device)
    assert common.validate_checkpoint(
        model_1,
        model_2,
        (*[input_image, noise_labels, class_labels, lead_time_labels],),
    )


@common.check_ort_version()
def test_son_unet_deploy(device):
    """Test Song UNet deployment support"""
    model = UNet(
        img_resolution=16,
        in_channels=10,
        out_channels=2,
        lead_time_channels=4,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
    ).to(device)

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    lead_time_labels = torch.randint(0, 9, (1,)).to(device)

    assert common.validate_onnx_export(
        model, (*[input_image, noise_labels, class_labels, lead_time_labels],)
    )
    assert common.validate_onnx_runtime(
        model, (*[input_image, noise_labels, class_labels, lead_time_labels],)
    )


@pytest.mark.parametrize("N_grid_channels", [0, 4])
@pytest.mark.parametrize("lead_time_channels", [0, 2])
def test_song_unet_positional_leadtime(device, N_grid_channels, lead_time_channels):
    """Test that both positional and lead-time embeddings can be used independently"""

    lead_time_mode = True

    img_resolution = 16
    out_channels = 2
    lead_time_steps = 2
    in_channels = 2 + N_grid_channels + (lead_time_channels if lead_time_mode else 0)

    def _create_model():
        return UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            N_grid_channels=N_grid_channels,
            lead_time_channels=lead_time_channels,
            lead_time_steps=lead_time_steps,
        ).to(device)

    if (lead_time_channels > 0) != lead_time_mode:
        with pytest.raises(ValueError):
            model = _create_model()
        return
    else:
        model = _create_model()

    noise_labels = torch.randn([2]).to(device)
    class_labels = torch.randint(0, 1, (2, 1)).to(device)
    input_image = torch.ones([2, 2, 16, 16]).to(device)
    lead_time_label = torch.as_tensor([0, 1]).to(device)

    assert bool(N_grid_channels) == (model.pos_embd is not None)
    assert lead_time_mode == (hasattr(model, "lt_embd") and (model.lt_embd is not None))

    if lead_time_mode:
        output_image = model(
            input_image, noise_labels, class_labels, lead_time_label=lead_time_label
        )
    else:
        output_image = model(input_image, noise_labels, class_labels)

    assert output_image.shape == (2, out_channels, img_resolution, img_resolution)
