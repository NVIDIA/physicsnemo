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

from physicsnemo.nn.functional import wireframe_render
from physicsnemo.nn.functional.rendering import WireframeRender
from test.conftest import requires_module


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.0], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_wireframe_render_warp(device: str):
    edges = torch.tensor(
        [[[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    eye, center, up = _camera(device)

    rgba, depth = wireframe_render(
        edges,
        21,
        21,
        eye,
        center,
        up,
        45.0,
        line_color=torch.tensor([0.8, 0.7, 0.2], device=device),
        implementation="warp",
    )

    assert rgba.shape == (21, 21, 4)
    assert depth.shape == (21, 21)
    assert float(rgba[..., 3].sum()) > 0.0
    assert torch.isfinite(depth).any()
    assert "warp" in WireframeRender.available_implementations()


@requires_module("warp")
def test_wireframe_render_clips_depth_range(device: str):
    edges = torch.tensor(
        [[[0.0, -0.5, -2.2], [0.0, 0.5, 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    eye, center, up = _camera(device)

    rgba, depth = wireframe_render(
        edges,
        21,
        21,
        eye,
        center,
        up,
        45.0,
        near=0.1,
        far=10.0,
        implementation="warp",
    )

    assert float(rgba[..., 3].sum()) > 0.0
    assert torch.isfinite(depth).any()


@requires_module("warp")
def test_wireframe_render_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(WireframeRender.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = WireframeRender.dispatch(*args, implementation="warp", **kwargs)
    assert output is not None


@requires_module("warp")
def test_wireframe_render_error_handling(device: str):
    eye, center, up = _camera(device)

    with pytest.raises(ValueError, match="line_thickness"):
        wireframe_render(
            torch.zeros(1, 2, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            line_thickness=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="at least one edge"):
        wireframe_render(
            torch.zeros(0, 2, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="up must not be parallel"):
        wireframe_render(
            torch.zeros(1, 2, 3, device=device),
            16,
            16,
            eye,
            center,
            center - eye,
            45.0,
            implementation="warp",
        )
