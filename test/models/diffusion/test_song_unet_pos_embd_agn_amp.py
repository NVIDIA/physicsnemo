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

from physicsnemo.models.diffusion_unets import SongUNetPosEmbd as UNet
from test import common
from test.conftest import requires_module


@requires_module("apex")
def test_song_unet_global_indexing(apex_device):
    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    torch.manual_seed(0)
    N_pos = 2
    batch_shape_x = 32
    batch_shape_y = 64
    # Construct the DDM++ UNet model

    model = (
        UNet(
            img_resolution=128,
            in_channels=2 + N_pos,
            out_channels=2,
            gridtype="test",
            N_grid_channels=N_pos,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )
    input_image = torch.ones([1, 2, batch_shape_x, batch_shape_y]).to(apex_device)
    noise_labels = noise_labels = torch.randn([1]).to(apex_device)
    class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
    idx_x = torch.arange(45, 45 + batch_shape_x)
    idx_y = torch.arange(12, 12 + batch_shape_y)
    mesh_x, mesh_y = torch.meshgrid(idx_x, idx_y)
    global_index = torch.stack((mesh_x, mesh_y), dim=0)[None].to(apex_device)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels, global_index)

    pos_embed = model.positional_embedding_indexing(input_image, global_index)
    assert output_image.shape == (1, 2, batch_shape_x, batch_shape_y)
    assert torch.equal(pos_embed, global_index)


@requires_module("apex")
def test_song_unet_constructor(apex_device):
    """Test the Song UNet constructor options"""

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    # DDM++
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    N_pos = 4
    model = (
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels + N_pos,
            out_channels=out_channels,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )
    noise_labels = torch.randn([1]).to(apex_device)
    class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
    input_image = torch.ones([1, 2, 16, 16]).to(apex_device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)

    # test rectangular shape
    model = (
        UNet(
            img_resolution=[img_resolution, img_resolution * 2],
            in_channels=in_channels + N_pos,
            out_channels=out_channels,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )
    noise_labels = torch.randn([1]).to(apex_device)
    class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
    input_image = torch.ones([1, out_channels, img_resolution, img_resolution * 2]).to(
        apex_device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution * 2)


@requires_module("apex")
def test_song_unet_position_embedding(apex_device):
    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    # build unet
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    # NCSN++
    N_pos = 100
    model = (
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels + N_pos,
            out_channels=out_channels,
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            resample_filter=[1, 3, 3, 1],
            gridtype="learnable",
            N_grid_channels=N_pos,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )
    noise_labels = torch.randn([1]).to(apex_device)
    class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
    input_image = torch.ones([1, 2, 16, 16]).to(apex_device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)

    model = (
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            N_grid_channels=40,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )
    assert model.pos_embd.shape == (40, img_resolution, img_resolution)


@requires_module("apex")
def test_fails_if_grid_is_invalid(apex_device):
    """Test the positional embedding options. "linear" gridtype only support 2 channels, and N_grid_channels in "sinusoidal" should be a factor of 4"""
    img_resolution = 16
    in_channels = 2
    out_channels = 2

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    with pytest.raises(ValueError):
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            gridtype="linear",
            N_grid_channels=20,
            use_apex_gn=True,
            amp_mode=True,
        ).to(memory_format=torch.channels_last)

    with pytest.raises(ValueError):
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            gridtype="sinusoidal",
            N_grid_channels=11,
            use_apex_gn=True,
            amp_mode=True,
        ).to(memory_format=torch.channels_last)


@requires_module("apex")
def test_song_unet_optims(apex_device):
    """Test Song UNet optimizations"""

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    def setup_model():
        model = (
            UNet(
                img_resolution=16,
                in_channels=6,
                out_channels=2,
                embedding_type="fourier",
                channel_mult_noise=2,
                encoder_type="residual",
                resample_filter=[1, 3, 3, 1],
                use_apex_gn=True,
                amp_mode=True,
            )
            .to(apex_device)
            .to(memory_format=torch.channels_last)
        )
        noise_labels = torch.randn([1]).to(apex_device)
        class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
        input_image = torch.ones([1, 2, 16, 16]).to(apex_device)

        return model, [input_image, noise_labels, class_labels]

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
    # Check Combo
    model, invar = setup_model()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        assert common.validate_combo_optims(model, (*invar,))


@requires_module("apex")
def test_song_unet_checkpoint(apex_device):
    """Test Song UNet checkpoint save/load"""

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    # Construct FNO models
    model_1 = (
        UNet(
            img_resolution=16,
            in_channels=6,
            out_channels=2,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )

    model_2 = (
        UNet(
            img_resolution=16,
            in_channels=6,
            out_channels=2,
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )

    noise_labels = torch.randn([1]).to(apex_device)
    class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
    input_image = torch.ones([1, 2, 16, 16]).to(apex_device)
    assert common.validate_checkpoint(
        model_1,
        model_2,
        (*[input_image, noise_labels, class_labels],),
        enable_autocast=True,
    )


@requires_module("apex")
@common.check_ort_version()
def test_son_unet_deploy(apex_device):
    """Test Song UNet deployment support"""

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    model = (
        UNet(
            img_resolution=16,
            in_channels=6,
            out_channels=2,
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            resample_filter=[1, 3, 3, 1],
            use_apex_gn=True,
            amp_mode=True,
        )
        .to(apex_device)
        .to(memory_format=torch.channels_last)
    )

    noise_labels = torch.randn([1]).to(apex_device)
    class_labels = torch.randint(0, 1, (1, 1)).to(apex_device)
    input_image = torch.ones([1, 2, 16, 16]).to(apex_device)

    assert common.validate_onnx_export(
        model, (*[input_image, noise_labels, class_labels],)
    )
    assert common.validate_onnx_runtime(
        model, (*[input_image, noise_labels, class_labels],)
    )
