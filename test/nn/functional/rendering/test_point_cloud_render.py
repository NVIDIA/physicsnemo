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

from physicsnemo.nn.functional import point_cloud_render
from physicsnemo.nn.functional.rendering import PointCloudRender
from test.conftest import requires_module


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.0], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_point_cloud_render_warp(device: str):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.35, 0.2, 0.0]], device=device, dtype=torch.float32
    )
    colors = torch.tensor(
        [[255, 0, 0], [0, 128, 255]], device=device, dtype=torch.uint8
    )
    eye, center, up = _camera(device)

    rgba, depth = point_cloud_render(
        points,
        21,
        21,
        eye,
        center,
        up,
        45.0,
        point_colors=colors,
        point_size=1,
        implementation="warp",
    )

    assert rgba.shape == (21, 21, 4)
    assert depth.shape == (21, 21)
    assert float(rgba[..., 3].sum()) == pytest.approx(2.0)
    assert torch.isfinite(depth).any()
    assert "warp" in PointCloudRender.available_implementations()


@requires_module("warp")
def test_point_cloud_render_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(PointCloudRender.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = PointCloudRender.dispatch(*args, implementation="warp", **kwargs)
    assert output is not None


@requires_module("warp")
def test_point_cloud_render_error_handling(device: str):
    eye, center, up = _camera(device)

    with pytest.raises(ValueError, match="point_size"):
        point_cloud_render(
            torch.zeros(1, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            point_size=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="either point_colors or point_color"):
        point_cloud_render(
            torch.zeros(1, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            point_colors=torch.zeros(1, 3, device=device),
            point_color=torch.ones(3, device=device),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="at least one point"):
        point_cloud_render(
            torch.zeros(0, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            implementation="warp",
        )
