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

from physicsnemo.nn.functional import volume_render
from physicsnemo.nn.functional.rendering import VolumeRender
from test.conftest import requires_module


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.5], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_volume_render_warp(device: str):
    rgba_volume = torch.zeros(16, 16, 16, 4, device=device, dtype=torch.uint8)
    rgba_volume[5:11, 5:11, 5:11, 0] = 255
    rgba_volume[5:11, 5:11, 5:11, 3] = 128
    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    eye, center, up = _camera(device)

    rgba, depth = volume_render(
        rgba_volume,
        25,
        25,
        eye,
        center,
        up,
        35.0,
        bounds_min,
        bounds_max,
        step_size=0.08,
        max_steps=80,
        implementation="warp",
    )

    assert rgba.shape == (25, 25, 4)
    assert depth.shape == (25, 25)
    assert float(rgba[..., 3].sum()) > 0.0
    assert float(rgba[..., 0].max()) > 0.8
    assert torch.isfinite(depth).any()
    assert torch.isinf(depth[0, 0])
    assert "warp" in VolumeRender.available_implementations()


@requires_module("warp")
def test_volume_render_accepts_sequence_camera_inputs(device: str):
    rgba_volume = torch.zeros(8, 8, 8, 4, device=device, dtype=torch.uint8)
    rgba_volume[2:6, 2:6, 2:6, 1] = 255
    rgba_volume[2:6, 2:6, 2:6, 3] = 128

    rgba, depth = volume_render(
        rgba_volume,
        11,
        11,
        [0.0, 0.0, -2.5],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        35.0,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        step_size=0.12,
        max_steps=48,
        implementation="warp",
    )

    assert rgba.shape == (11, 11, 4)
    assert depth.shape == (11, 11)
    assert float(rgba[..., 3].sum()) > 0.0


@requires_module("warp")
def test_volume_render_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(VolumeRender.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = VolumeRender.dispatch(*args, implementation="warp", **kwargs)
    assert output is not None


@requires_module("warp")
def test_volume_render_error_handling(device: str):
    rgba_volume = torch.zeros(8, 8, 8, 4, device=device, dtype=torch.uint8)
    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    eye, center, up = _camera(device)

    with pytest.raises(ValueError, match="step_size"):
        volume_render(
            rgba_volume,
            11,
            11,
            eye,
            center,
            up,
            35.0,
            bounds_min,
            bounds_max,
            step_size=0.0,
            implementation="warp",
        )
