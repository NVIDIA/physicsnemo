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

from test import common


class MulX(torch.nn.Module):
    """Helper class that just multiplies the values of an input tensor"""

    def __init__(self, multiplier: int = 1):
        super(MulX, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x * self.multiplier


HEALPixLayer_testdata = [
    ("cuda:0", 2),
    ("cuda:0", 3),
    ("cuda:0", 4),
    ("cpu", 2),
    ("cpu", 3),
    ("cpu", 4),
]


@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_initialization(device, multiplier, pytestconfig):
    from physicsnemo.nn.module.hpx import (
        HEALPixLayer,
    )

    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)
    assert isinstance(layer, HEALPixLayer)


@pytest.mark.parametrize("multiplier", [2, 3, 4])
def test_HEALPixLayer_forward(device, multiplier, pytestconfig):
    from physicsnemo.nn.module.hpx import (
        HEALPixLayer,
    )

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

    # size of the padding added byu HEALPixLayer
    expected_shape = [24, out_channels, tensor_size[-1], tensor_size[-1]]
    expected_shape = torch.Size(expected_shape)

    assert expected_shape == layer(invar).shape

    del layer, outvar, invar
    torch.cuda.empty_cache()
