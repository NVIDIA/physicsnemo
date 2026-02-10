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

import numpy as np
import pytest
import torch

from physicsnemo.nn.module.hpx import (
    HEALPixAvgPool,
    HEALPixLayer,
    HEALPixMaxPool,
    HEALPixPadding,
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)
from physicsnemo.nn.module.hpx.padding import (
    HEALPixFoldFaces,
    HEALPixUnfoldFaces,
)
from physicsnemo.nn.module.hpx.tokenizer import (
    CalendarEmbedding,
)
from test import common
from test.conftest import requires_module


class MulX(torch.nn.Module):
    """Helper class that just multiplies the values of an input tensor."""

    def __init__(self, multiplier: int = 1):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x * self.multiplier


@pytest.fixture
def test_data():
    def generate_test_data(faces=12, channels=2, img_size=16, device="cpu"):
        test = torch.eye(img_size, device=device)
        test = test[(None,) * 2]
        return test.expand([faces, channels, -1, -1])

    return generate_test_data


@requires_module("earth2grid")
def test_HEALPixFoldFaces_initialization(device, pytestconfig):
    fold_func = HEALPixFoldFaces()
    assert isinstance(fold_func, HEALPixFoldFaces)


@requires_module("earth2grid")
def test_HEALPixFoldFaces_forward(device, pytestconfig):
    fold_func = HEALPixFoldFaces()

    tensor_size = torch.randint(low=2, high=4, size=(5,)).tolist()
    output_size = (tensor_size[0] * tensor_size[1], *tensor_size[2:])
    invar = torch.ones(*tensor_size, device=device)

    outvar = fold_func(invar)
    assert outvar.shape == output_size

    fold_func = HEALPixFoldFaces(enable_nhwc=True)
    assert fold_func(invar).shape == outvar.shape
    assert fold_func(invar).stride() != outvar.stride()


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_initialization(device, pytestconfig):
    unfold_func = HEALPixUnfoldFaces()
    assert isinstance(unfold_func, HEALPixUnfoldFaces)


@requires_module("earth2grid")
def test_HEALPixUnfoldFaces_forward(device, pytestconfig):
    num_faces = 12
    unfold_func = HEALPixUnfoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(4,)).tolist()
    output_size = (tensor_size[0], num_faces, *tensor_size[1:])

    tensor_size[0] *= num_faces
    invar = torch.ones(*tensor_size, device=device)

    outvar = unfold_func(invar)
    assert outvar.shape == output_size


@requires_module("earth2grid")
@pytest.mark.parametrize("padding", [2, 3, 4])
def test_HEALPixPadding_initialization(device, padding, pytestconfig):
    pad_func = HEALPixPadding(padding)
    assert isinstance(pad_func, HEALPixPadding)


@requires_module("earth2grid")
@pytest.mark.parametrize("padding", [2, 3, 4])
def test_HEALPixPadding_forward(device, padding, pytestconfig):
    num_faces = 12
    batch_size = 2
    pad_func = HEALPixPadding(padding)

    with pytest.raises(
        ValueError, match=("invalid value for 'padding', expected int > 0 but got 0")
    ):
        HEALPixPadding(0)

    hw_size = torch.randint(low=4, high=24, size=(1,)).tolist()
    c_size = torch.randint(low=3, high=7, size=(1,)).tolist()
    hw_size = np.asarray(hw_size + hw_size)

    tensor_size = (batch_size * num_faces, *c_size, *hw_size)
    invar = torch.rand(tensor_size, device=device)

    hw_padded_size = hw_size + (2 * padding)
    out_size = (batch_size * num_faces, *c_size, *hw_padded_size)

    outvar = pad_func(invar)
    assert outvar.shape == out_size


@requires_module("earth2grid")
@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_initialization(device, multiplier, pytestconfig):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)
    assert isinstance(layer, HEALPixLayer)


@requires_module("earth2grid")
@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_forward(device, multiplier, pytestconfig):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)

    kernel_size = 3
    dilation = 2
    in_channels = 4
    out_channels = 8

    tensor_size = torch.randint(low=2, high=4, size=(1,)).tolist()
    tensor_size = [24, in_channels, *tensor_size, *tensor_size]
    invar = torch.rand(tensor_size, device=device)
    outvar = layer(invar)

    assert common.compare_output(outvar, invar * multiplier)

    layer = HEALPixLayer(
        layer=torch.nn.Conv2d,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        device=device,
        dilation=dilation,
        enable_healpixpad=True,
        enable_nhwc=True,
    )

    expected_shape = [24, out_channels, tensor_size[-1], tensor_size[-1]]
    expected_shape = torch.Size(expected_shape)

    assert expected_shape == layer(invar).shape


@requires_module("earth2grid")
def test_MaxPool_initialization(device, pytestconfig):
    pooling = 2
    maxpool_block = HEALPixMaxPool(pooling=pooling).to(device)
    assert isinstance(maxpool_block, HEALPixMaxPool)


@requires_module("earth2grid")
def test_MaxPool_forward(device, test_data, pytestconfig):
    pooling = 2
    size = 16
    channels = 4
    maxpool_block = HEALPixMaxPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    assert common.compare_output(outvar, maxpool_block(invar))


@requires_module("earth2grid")
def test_AvgPool_initialization(device, pytestconfig):
    pooling = 2
    avgpool_block = HEALPixAvgPool(pooling=pooling).to(device)
    assert isinstance(avgpool_block, HEALPixAvgPool)


@requires_module("earth2grid")
def test_AvgPool_forward(device, test_data, pytestconfig):
    pooling = 2
    size = 32
    channels = 4
    avgpool_block = HEALPixAvgPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    outvar = outvar * 0.5

    assert common.compare_output(outvar, avgpool_block(invar))


@requires_module("earth2grid")
def test_hpx_patch_tokenizer_forward(device):
    """Test HEALPixPatchTokenizer forward pass."""
    torch.manual_seed(0)

    in_channels = 5
    hidden_size = 8
    level_fine = 2
    level_coarse = 1

    model = HEALPixPatchTokenizer(
        in_channels=in_channels,
        hidden_size=hidden_size,
        level_fine=level_fine,
        level_coarse=level_coarse,
    ).to(device)
    model.eval()

    b, t = 2, 1
    npix = 12 * 4**level_fine
    x = torch.randn(b, in_channels, t, npix).to(device)
    second_of_day = torch.tensor([[43200], [21600]], device=device)
    day_of_year = torch.tensor([[100], [200]], device=device)
    # Manually track device since not psn Module
    model.device = device

    assert common.validate_forward_accuracy(
        model,
        (x, second_of_day, day_of_year),
        file_name="nn/module/data/hpx_tokenizer_output.pth",
        atol=1e-3,  # Data on order of [1 to 0.1]
    )


@requires_module("earth2grid")
def test_calendar_embedding_shape_mismatch():
    """Test CalendarEmbedding raises on shape mismatch."""
    lon = torch.linspace(-180, 180, 10)
    model = CalendarEmbedding(lon=lon, embed_channels=4)

    day_of_year = torch.tensor([[100, 101]])
    second_of_day = torch.tensor([[43200]])

    with pytest.raises(ValueError):
        model(day_of_year=day_of_year, second_of_day=second_of_day)


# HealDA tokenizers
@requires_module("earth2grid")
def test_hpx_patch_detokenizer_forward(device):
    """Test HEALPixPatchDetokenizer forward pass."""
    torch.manual_seed(0)

    hidden_size = 8
    out_channels = 2
    level_coarse = 1
    level_fine = 2
    time_length = 1

    model = HEALPixPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=time_length,
    ).to(device)
    model.eval()

    b = 2
    L = time_length * 12 * 4**level_coarse
    x = torch.randn(b, L, hidden_size).to(device)
    c = torch.randn(b, hidden_size).to(device)
    # Manually track device since not psn Module
    model.device = device

    assert common.validate_forward_accuracy(
        model,
        (x, c),
        file_name="nn/module/data/hpx_detokenizer_output.pth",
        atol=1e-3,  # Data on order of [1 to 0.1]
    )
