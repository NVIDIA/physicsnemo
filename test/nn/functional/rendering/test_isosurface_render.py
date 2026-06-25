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

from physicsnemo.nn.functional import isosurface_render
from physicsnemo.nn.functional.rendering import IsosurfaceRender
from test.conftest import requires_module


def _sphere_field(grid_n: int, device: str, radius: float = 0.5) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, grid_n, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.sqrt(x * x + y * y + z * z) - radius


def _sphere_color_field(grid_n: int, device: str) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, grid_n, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    rgb = torch.stack([x, y, 0.35 + 0.65 * z], dim=-1)
    return (rgb * 255).to(torch.uint8)


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.5], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_isosurface_render_warp(device: str):
    field = _sphere_field(48, device)
    color_field = _sphere_color_field(48, device)
    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    eye, center, up = _camera(device)

    rgba, depth, normal = isosurface_render(
        field,
        33,
        33,
        eye,
        center,
        up,
        35.0,
        bounds_min,
        bounds_max,
        threshold=0.0,
        step_size=0.05,
        max_steps=128,
        color_field=color_field,
        light_direction=torch.tensor([0.0, 0.0, -1.0], device=device),
        implementation="warp",
    )

    assert rgba.shape == (33, 33, 4)
    assert depth.shape == (33, 33)
    assert normal.shape == (33, 33, 3)
    assert float(rgba[..., 3].sum()) > 0.0
    torch.testing.assert_close(
        depth[16, 16], torch.tensor(2.0, device=device), atol=6.0e-2, rtol=0.0
    )
    assert float(normal[16, 16, 2]) < -0.9
    hit_luminance = rgba[..., :3].mean(dim=-1)[rgba[..., 3] > 0.0]
    assert float(hit_luminance.max() - hit_luminance.min()) > 0.15
    assert "warp" in IsosurfaceRender.available_implementations()


@requires_module("warp")
def test_isosurface_render_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(IsosurfaceRender.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    rgba, depth, normal = IsosurfaceRender.dispatch(
        *args, implementation="warp", **kwargs
    )
    assert rgba.shape[-1] == 4
    assert normal.shape[-1] == 3
    assert depth.shape == rgba.shape[:2]


@requires_module("warp")
def test_isosurface_render_error_handling(device: str):
    field = _sphere_field(16, device)
    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    eye, center, up = _camera(device)

    with pytest.raises(ValueError, match="color_field spatial shape"):
        isosurface_render(
            field,
            16,
            16,
            eye,
            center,
            up,
            45.0,
            bounds_min,
            bounds_max,
            color_field=torch.zeros(15, 16, 16, 3, device=device),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="step_size"):
        isosurface_render(
            field,
            16,
            16,
            eye,
            center,
            up,
            45.0,
            bounds_min,
            bounds_max,
            step_size=0.0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="eye and center"):
        isosurface_render(
            field,
            16,
            16,
            eye,
            eye,
            up,
            45.0,
            bounds_min,
            bounds_max,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="up must not be parallel"):
        isosurface_render(
            field,
            16,
            16,
            eye,
            center,
            center - eye,
            45.0,
            bounds_min,
            bounds_max,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="bounds_max"):
        isosurface_render(
            field,
            16,
            16,
            eye,
            center,
            up,
            45.0,
            bounds_max,
            bounds_min,
            implementation="warp",
        )
