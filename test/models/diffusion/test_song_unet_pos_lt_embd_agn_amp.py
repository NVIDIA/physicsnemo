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
# ruff: noqa: E402
import os
import sys

import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common

from physicsnemo.models.diffusion import SongUNetPosLtEmbd


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
    model = SongUNetPosLtEmbd(
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
        use_apex_gn=True,
        amp_mode=True,
        prob_channels=[1, 3],
    )
    return model


def setup_model_ddm_plus_plus(img_resolution, lt_steps, lt_channels, seed=0):
    """
    Create a model with similar architecture to DDM++.
    """
    C_x, N_pos, C_cond = 4, 4, 3
    torch.manual_seed(seed)
    model = SongUNetPosLtEmbd(
        img_resolution=img_resolution,
        in_channels=C_x + N_pos + C_cond + lt_channels,
        out_channels=C_x,
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
        use_apex_gn=True,
        amp_mode=True,
        prob_channels=[1, 3],
    )
    return model


def setup_model_ncsn_plus_plus(img_resolution, lt_steps, lt_channels, seed=0):
    """
    Create a model with similar architecture to NCSN++.
    """
    C_x, N_pos, C_cond = 4, 4, 3
    torch.manual_seed(seed)
    model = SongUNetPosLtEmbd(
        img_resolution=img_resolution,
        in_channels=C_x + N_pos + C_cond + lt_channels,
        out_channels=C_x,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
        lead_time_steps=lt_steps,
        lead_time_channels=lt_channels,
        use_apex_gn=True,
        amp_mode=True,
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


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_constructor(device):
    """
    Test the SongUNetPosLtEmbd constructor for different architectures and shapes.
    Also test the shapes of the positional and lead time embeddings.
    """

    # Test DDM++ with square shape
    lt_steps, lt_channels = 4, 8
    H = W = 16
    model = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (4, H, W)
    assert model.lt_embd.shape == (lt_steps, lt_channels, H, W)

    # Test DDM++ with rectangular shape
    lt_steps, lt_channels = 4, 8
    H, W = 16, 32
    model = (
        setup_model_ddm_plus_plus([H, W], lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (4, H, W)
    assert model.lt_embd.shape == (lt_steps, lt_channels, H, W)

    # Test NCSN++ with rectangular shape
    lt_steps, lt_channels = 4, 8
    H, W = 16, 32
    model = (
        setup_model_ncsn_plus_plus([H, W], lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (4, H, W)
    assert model.lt_embd.shape == (lt_steps, lt_channels, H, W)

    # Test corrdiff model with rectangular shape
    N_pos, lt_steps, lt_channels = 6, 4, 8
    H, W = 16, 32
    model = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (N_pos, H, W)
    assert model.lt_embd.shape == (lt_steps, lt_channels, H, W)


# TODO: duplicate tests for model.eval()
@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_forward_no_patches(device):
    """
    Test the forward method of the SongUNetPosLtEmbd for different architectures
    without patches (i.e. input image is the entire global image). Uses AMP, Apex GN,
    and compile (for small models only). Also test backward propagation through
    the model.
    """
    torch._dynamo.reset()

    # Common parameters
    B, C_x, lt_steps = 3, 4, 4

    # DDM++ model with square global shape (no compile because model too large)
    H = W = 128
    N_pos, lt_channels = 4, 8
    model = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(*generate_data_no_patches(H, W, device))
    assert output_image.shape == (B, C_x, H, W)
    loss = output_image.sum()
    loss.backward()
    # TODO: add non-regression test

    # NCSN++ model with square global shape (no compile because model too large)
    H = W = 128
    N_pos, lt_channels = 4, 8
    model = (
        setup_model_ncsn_plus_plus(H, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(*generate_data_no_patches(H, W, device))
    assert output_image.shape == (B, C_x, H, W)
    loss = output_image.sum()
    loss.backward()
    # TODO: add non-regression test

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_channels = 6, 8
    model = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    # Compile model
    model = common.torch_compile_model(model)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(*generate_data_no_patches(H, W, device))
    assert output_image.shape == (B, C_x, H, W)
    loss = output_image.sum()
    loss.backward()
    # TODO: add non-regression test

    return


# TODO: duplicate tests for model.eval() mode
@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_forward_with_patches(device):
    """
    Test the forward method of the SongUNetPosLtEmbd for different architectures
    with patches (i.e. only a subset of the global image). Uses AMP, Apex GN,
    and compile (for small models only). Also test backward propagation through
    the model.
    """
    torch._dynamo.reset()

    # Common parameters
    P, B, C_x, H_p, W_p, lt_steps = 4, 3, 4, 32, 64, 4

    # DDM++ model with square global shape (no compile because model too large)
    H = W = 128
    N_pos, lt_channels = 4, 8
    model = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(*generate_data_with_patches(H_p, W_p, device))
    assert output_image.shape == (P * B, C_x, H_p, W_p)
    loss = output_image.sum()
    loss.backward()
    # TODO: add non-regression test

    # NCSN++ model with square global shape (no compile because model too large)
    H = W = 128
    N_pos, lt_channels = 4, 8
    model = (
        setup_model_ncsn_plus_plus(H, lt_steps, lt_channels)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(*generate_data_with_patches(H_p, W_p, device))
    assert output_image.shape == (P * B, C_x, H_p, W_p)
    loss = output_image.sum()
    loss.backward()
    # TODO: add non-regression test

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_channels = 6, 8
    model = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    # Compile model
    model = common.torch_compile_model(model)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(*generate_data_with_patches(H_p, W_p, device))
    assert output_image.shape == (P * B, C_x, H_p, W_p)
    loss = output_image.sum()
    loss.backward()
    # TODO: add non-regression test

    return


@pytest.mark.parametrize("device", ["cuda:0"])
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
        file_name="songunet_pos_lt_embd_pos_embed_indexing_no_patches_corrdiff.pth",
    )
    # TODO: add non-regression tests for other architectures


@pytest.mark.parametrize("device", ["cuda:0"])
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
        file_name="songunet_pos_lt_embd_pos_embed_indexing_with_patches_corrdiff.pth",
    )
    # TODO: add non-regression tests for other architectures


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_optims_no_patches(device):
    """Test SongUNetPosLtEmbd optimizations (CUDA graphs, JIT, AMP). Uses input
    data without patches (i.e. the entire global image)."""

    # NOTE: for now only test the corrdiff architecture
    def setup_model():
        H, W = 128, 112
        N_pos, lt_steps, lt_channels = 6, 4, 8
        model = (
            setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
            .to(device)
            .to(memory_format=torch.channels_last)
        )
        return model, generate_data_no_patches(H, W, device)

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # Check JIT
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_amp(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_optims_with_patches(device):
    """Test SongUNetPosLtEmbd optimizations (CUDA graphs, JIT, AMP). Uses input
    data with patches (i.e. input image is only a subset of the global image)."""

    # NOTE: for now only test the corrdiff architecture
    def setup_model():
        H, W = 128, 112
        H_p, W_p = 32, 64
        N_pos, lt_steps, lt_channels = 6, 4, 8
        model = (
            setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
            .to(device)
            .to(memory_format=torch.channels_last)
        )
        return model, generate_data_with_patches(H_p, W_p, device)

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # Check JIT
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_amp(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_checkpoint_no_patches(device):
    """Test SongUNetPosLtEmbd checkpoint save/load for different
    architectures. Uses input data without patches (i.e. input image is the
    entire global image)."""

    # Common parameters
    lt_steps = 4

    # DDM++ model with square global shape
    H = W = 128
    lt_steps, lt_channels = 4, 8
    model_1 = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels, seed=0)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    model_2 = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels, seed=1)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_checkpoint(
        model_1,
        model_2,
        generate_data_no_patches(H, W, device),
        enable_autocast=True,
    )

    # NCSN++ model with square global shape
    H = W = 128
    lt_steps, lt_channels = 4, 8
    model_1 = (
        setup_model_ncsn_plus_plus(H, lt_steps, lt_channels, seed=0)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    model_2 = (
        setup_model_ncsn_plus_plus(H, lt_steps, lt_channels, seed=1)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_checkpoint(
        model_1,
        model_2,
        generate_data_no_patches(H, W, device),
        enable_autocast=True,
    )

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_steps, lt_channels = 6, 4, 8
    model_1 = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos, seed=0)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    model_2 = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos, seed=1)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_checkpoint(
        model_1,
        model_2,
        generate_data_no_patches(H, W, device),
        enable_autocast=True,
    )

    return


@pytest.mark.parametrize("device", ["cuda:0"])
def test_song_unet_checkpoint_with_patches(device):
    """Test SongUNetPosLtEmbd checkpoint save/load for different
    architectures. Uses input data with patches (i.e. input image is only a
    subset of the global image)."""

    # Common parameters
    H_p, W_p, lt_steps = 32, 64, 4

    # DDM++ model with square global shape
    H = W = 128
    lt_steps, lt_channels = 4, 8
    model_1 = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels, seed=0)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    model_2 = (
        setup_model_ddm_plus_plus(H, lt_steps, lt_channels, seed=1)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_checkpoint(
        model_1,
        model_2,
        generate_data_with_patches(H_p, W_p, device),
        enable_autocast=True,
    )

    # NCSN++ model with square global shape
    H = W = 128
    lt_steps, lt_channels = 4, 8
    model_1 = (
        setup_model_ncsn_plus_plus(H, lt_steps, lt_channels, seed=0)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    model_2 = (
        setup_model_ncsn_plus_plus(H, lt_steps, lt_channels, seed=1)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_checkpoint(
        model_1,
        model_2,
        generate_data_with_patches(H_p, W_p, device),
        enable_autocast=True,
    )

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_steps, lt_channels = 6, 4, 8
    model_1 = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos, seed=0)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    model_2 = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos, seed=1)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_checkpoint(
        model_1,
        model_2,
        generate_data_with_patches(H_p, W_p, device),
        enable_autocast=True,
    )

    return


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0"])
def test_son_unet_deploy(device):
    """Test Song UNet deployment support"""

    # Common parameters
    H_p, W_p, lt_steps = 32, 64, 4

    # CorrDiff model with rectangular global shape
    H, W = 128, 112
    N_pos, lt_steps, lt_channels = 6, 4, 8
    model = (
        setup_model_learnable_embd([H, W], lt_steps, lt_channels, N_pos)
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert common.validate_onnx_export(
        model,
        generate_data_with_patches(H_p, W_p, device),
    )
    assert common.validate_onnx_runtime(
        model,
        generate_data_with_patches(H_p, W_p, device),
    )
