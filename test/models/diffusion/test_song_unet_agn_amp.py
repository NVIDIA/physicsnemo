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
from test.conftest import requires_module


@requires_module("apex")
def test_song_unet_constructor(apex_device):
    """Test the Song UNet constructor options"""

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    # DDM++
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    model = (
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
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

    # DDM++ with additive pos embed
    model_channels = 64
    model = (
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            additive_pos_embed=True,
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
    assert model.spatial_emb.shape == (
        1,
        model_channels,
        img_resolution,
        img_resolution,
    )

    # NCSN++
    model = (
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
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
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)

    # test rectangular shape
    model = (
        UNet(
            img_resolution=[img_resolution, img_resolution * 2],
            in_channels=in_channels,
            out_channels=out_channels,
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
    input_image = torch.ones([1, out_channels, img_resolution, img_resolution * 2]).to(
        apex_device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution * 2)

    # Also test failure cases
    try:
        model = (
            UNet(
                img_resolution=img_resolution,
                in_channels=in_channels,
                out_channels=out_channels,
                embedding_type=None,
                use_apex_gn=True,
                amp_mode=True,
            )
            .to(apex_device)
            .to(memory_format=torch.channels_last)
        )
        raise AssertionError("Failed to error for invalid argument")
    except ValueError:
        pass

    try:
        model = (
            UNet(
                img_resolution=img_resolution,
                in_channels=in_channels,
                out_channels=out_channels,
                encoder_type=None,
                use_apex_gn=True,
                amp_mode=True,
            )
            .to(apex_device)
            .to(memory_format=torch.channels_last)
        )
        raise AssertionError("Failed to error for invalid argument")
    except ValueError:
        pass

    try:
        model = (
            UNet(
                img_resolution=img_resolution,
                in_channels=in_channels,
                out_channels=out_channels,
                decoder_type=None,
                use_apex_gn=True,
                amp_mode=True,
            )
            .to(apex_device)
            .to(memory_format=torch.channels_last)
        )
        raise AssertionError("Failed to error for invalid argument")
    except ValueError:
        pass


@requires_module("apex")
def test_song_unet_optims(apex_device):
    """Test Song UNet optimizations"""

    if "cpu" in apex_device:
        pytest.skip("Apex GN is not supported on CPU")

    def setup_model():
        model = (
            UNet(
                img_resolution=16,
                in_channels=2,
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
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
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

    model_1 = (
        UNet(
            img_resolution=16,
            in_channels=2,
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
            in_channels=2,
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
    model = (
        UNet(
            img_resolution=16,
            in_channels=2,
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
