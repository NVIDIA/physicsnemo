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
from test.nn.module.healpix_helpers import MulX


class KwargCapture(torch.nn.Module):
    """Helper layer that records the kwargs it was instantiated with, so
    tests can verify which kwargs HEALPixLayer strips before forwarding."""

    captured_kwargs = None

    def __init__(self, **kwargs):
        super().__init__()
        KwargCapture.captured_kwargs = kwargs

    def forward(self, x):
        return x


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


def test_HEALPixLayer_strips_wrapper_kwargs(device, pytestconfig):
    """`enable_nhwc` and `enable_healpixpad` are HEALPixLayer-only knobs and
    must not be forwarded to the wrapped layer's constructor."""
    from physicsnemo.nn.module.hpx import HEALPixLayer

    HEALPixLayer(
        layer=KwargCapture,
        multiplier=5,
        enable_nhwc=False,
        enable_healpixpad=False,
    )

    assert "enable_nhwc" not in KwargCapture.captured_kwargs
    assert "enable_healpixpad" not in KwargCapture.captured_kwargs
    assert KwargCapture.captured_kwargs == {"multiplier": 5}


def test_HEALPixLayer_no_padding_when_kernel_size_small(device, pytestconfig):
    """A conv layer with kernel_size == 1 has no spatial neighborhood to
    stitch, so HEALPixLayer must not insert a HEALPixPadding submodule."""
    from physicsnemo.nn.module.hpx import HEALPixLayer
    from physicsnemo.nn.module.hpx.padding import HEALPixPadding

    in_channels = 4
    out_channels = 3
    kernel_size = 1

    layer = HEALPixLayer(
        layer=torch.nn.Conv2d,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        device=device,
    )

    assert not any(isinstance(m, HEALPixPadding) for m in layer.layers)

    size = 4
    invar = torch.rand(24, in_channels, size, size, device=device)
    outvar = layer(invar)

    assert outvar.shape == (24, out_channels, size, size)


def test_HEALPixLayer_conv_disables_native_padding(device, pytestconfig):
    """When HEALPixLayer inserts its own HEALPixPadding submodule for a
    kernel_size > 1 conv, the wrapped Conv2d's native padding must be
    disabled (0) since padding is already applied upstream."""
    from physicsnemo.nn.module.hpx import HEALPixLayer
    from physicsnemo.nn.module.hpx.padding import HEALPixPadding

    kernel_size = 3
    in_channels = 4
    out_channels = 3

    layer = HEALPixLayer(
        layer=torch.nn.Conv2d,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        device=device,
    )

    conv_submodules = [m for m in layer.layers if isinstance(m, torch.nn.Conv2d)]
    padding_submodules = [m for m in layer.layers if isinstance(m, HEALPixPadding)]

    assert len(conv_submodules) == 1
    assert len(padding_submodules) == 1
    assert conv_submodules[0].padding == (0, 0)

    # kernel_size=3 with dilation=1 needs 1 pixel of context on each side.
    assert padding_submodules[0].p == 1

    size = 4
    invar = torch.rand(24, in_channels, size, size, device=device)
    outvar = layer(invar)
    # HEALPixPadding restores the spatial size that Conv2d's kernel consumes.
    assert outvar.shape == (24, out_channels, size, size)
