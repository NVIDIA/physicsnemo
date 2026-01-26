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

import numpy as np
import pytest
import torch

from physicsnemo.nn import (
    HEALPixAvgPool,
    HEALPixFoldFaces,
    HEALPixLayer,
    HEALPixMaxPool,
    HEALPixPadding,
    HEALPixUnfoldFaces,
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


@requires_module("hydra")
def test_HEALPixFoldFaces_initialization(device, pytestconfig):
    fold_func = HEALPixFoldFaces()
    assert isinstance(fold_func, HEALPixFoldFaces)


@requires_module("hydra")
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


@requires_module("hydra")
def test_HEALPixUnfoldFaces_initialization(device, pytestconfig):
    unfold_func = HEALPixUnfoldFaces()
    assert isinstance(unfold_func, HEALPixUnfoldFaces)


@requires_module("hydra")
def test_HEALPixUnfoldFaces_forward(device, pytestconfig):
    num_faces = 12
    unfold_func = HEALPixUnfoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(4,)).tolist()
    output_size = (tensor_size[0], num_faces, *tensor_size[1:])

    tensor_size[0] *= num_faces
    invar = torch.ones(*tensor_size, device=device)

    outvar = unfold_func(invar)
    assert outvar.shape == output_size


@requires_module("hydra")
@pytest.mark.parametrize("padding", [2, 3, 4])
def test_HEALPixPadding_initialization(device, padding, pytestconfig):
    pad_func = HEALPixPadding(padding)
    assert isinstance(pad_func, HEALPixPadding)


@requires_module("hydra")
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


@requires_module("hydra")
@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_initialization(device, multiplier, pytestconfig):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)
    assert isinstance(layer, HEALPixLayer)


@requires_module("hydra")
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


@requires_module("hydra")
def test_MaxPool_initialization(device, pytestconfig):
    pooling = 2
    maxpool_block = HEALPixMaxPool(pooling=pooling).to(device)
    assert isinstance(maxpool_block, HEALPixMaxPool)


@requires_module("hydra")
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


@requires_module("hydra")
def test_AvgPool_initialization(device, pytestconfig):
    pooling = 2
    avgpool_block = HEALPixAvgPool(pooling=pooling).to(device)
    assert isinstance(avgpool_block, HEALPixAvgPool)


@requires_module("hydra")
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
