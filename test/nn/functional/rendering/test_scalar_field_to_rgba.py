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

import pytest
import torch

from physicsnemo.nn.functional import scalar_field_to_rgba
from physicsnemo.nn.functional.rendering import ScalarFieldToRGBA
from test.conftest import requires_module


@requires_module("warp")
def test_scalar_field_to_rgba_warp(device: str):
    field = torch.linspace(0.0, 1.0, 16, device=device).reshape(4, 4, 1)
    field = field.expand(4, 4, 4).contiguous()

    rgba_volume = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.5,
        opacity_threshold=0.25,
        implementation="warp",
    )

    assert rgba_volume.shape == (4, 4, 4, 4)
    assert rgba_volume.dtype == torch.uint8
    assert int(rgba_volume[..., 3].min()) == 0
    assert int(rgba_volume[..., 3].max()) <= 128
    assert int(rgba_volume[..., :3].max()) > 0
    assert "warp" in ScalarFieldToRGBA.available_implementations()
    assert "torch" in ScalarFieldToRGBA.available_implementations()


def test_scalar_field_to_rgba_torch():
    field = torch.linspace(0.0, 1.0, 16).reshape(4, 4, 1)
    field = field.expand(4, 4, 4).contiguous()

    rgba_volume = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.5,
        opacity_threshold=0.25,
        implementation="torch",
    )

    assert rgba_volume.shape == (4, 4, 4, 4)
    assert rgba_volume.dtype == torch.uint8
    assert int(rgba_volume[..., 3].min()) == 0
    assert int(rgba_volume[..., 3].max()) <= 128


@requires_module("warp")
def test_scalar_field_to_rgba_backend_forward_parity(device: str):
    field = torch.linspace(-0.2, 1.2, 5 * 6 * 7, device=device).reshape(5, 6, 7)

    rgba_warp = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.7,
        opacity_threshold=0.2,
        implementation="warp",
    )
    rgba_torch = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.7,
        opacity_threshold=0.2,
        implementation="torch",
    )

    ScalarFieldToRGBA.compare_forward(rgba_warp, rgba_torch)


def test_scalar_field_to_rgba_make_inputs_forward():
    label, args, kwargs = next(iter(ScalarFieldToRGBA.make_inputs_forward("cpu")))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = ScalarFieldToRGBA.dispatch(*args, implementation="torch", **kwargs)
    assert output is not None


def test_scalar_field_to_rgba_compare_forward_contract():
    reference = torch.zeros(2, 2, 2, 4, dtype=torch.uint8)
    output = reference.clone()
    ScalarFieldToRGBA.compare_forward(output, reference)


def test_scalar_field_to_rgba_error_handling():
    field = torch.zeros(4, 4, 4)
    with pytest.raises(ValueError, match="vmax"):
        scalar_field_to_rgba(field, 1.0, 1.0, implementation="torch")
