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

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._warp_impl import isosurface_render_warp


def _sphere_field(
    grid_n: int, device: torch.device, radius: float = 0.5
) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, grid_n, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.sqrt(x * x + y * y + z * z) - radius


class IsosurfaceRender(FunctionSpec):
    """Render a threshold isosurface from a scalar volume.

    This is a fused image-space renderer: one Warp thread computes one output
    pixel, generates the camera ray, intersects the volume bounds, marches the
    scalar field, samples optional RGB/RGBA color data, shades the hit, and
    writes ``(rgba, depth, normal)`` buffers.

    Args:
        field: Scalar volume with shape ``(nx, ny, nz)``.
        image_height: Output image height.
        image_width: Output image width.
        eye: Camera position with shape ``(3,)``.
        center: Camera look-at point with shape ``(3,)``.
        up: Camera up direction with shape ``(3,)``.
        fov_y_degrees: Vertical field of view in degrees.
        bounds_min: Minimum world-space volume bound with shape ``(3,)``.
        bounds_max: Maximum world-space volume bound with shape ``(3,)``.
        threshold: Isosurface scalar threshold. Defaults to ``0.0``.
        step_size: Ray-marching step size in world units. Defaults to ``0.01``.
        max_steps: Maximum number of march steps per pixel. Defaults to ``512``.
        color_field: Optional RGB/RGBA volume with shape ``(nx, ny, nz, 3|4)``.
            ``uint8`` colors are normalized to ``[0, 1]``.
        surface_color: Optional uniform RGB/RGBA color used when ``color_field``
            is omitted.
        light_direction: Optional surface-to-light direction with shape ``(3,)``.
        ambient: Ambient lighting coefficient in ``[0, 1]``.
        implementation: Explicit implementation name. Currently only ``"warp"``
            is registered.

    Returns:
        Tuple of ``(rgba, depth, normal)`` image tensors. Missed pixels have zero
        alpha, infinite depth, and zero normal.
    """

    _BENCHMARK_CASES = (
        ("small-grid32-img32", 32, 32, 0.05, 96),
        ("medium-grid48-img64", 48, 64, 0.035, 160),
    )

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=1.0.0",), rank=0, baseline=True
    )
    def warp_forward(
        field: torch.Tensor,
        image_height: int,
        image_width: int,
        eye: torch.Tensor | Sequence[float],
        center: torch.Tensor | Sequence[float],
        up: torch.Tensor | Sequence[float],
        fov_y_degrees: float,
        bounds_min: torch.Tensor | Sequence[float],
        bounds_max: torch.Tensor | Sequence[float],
        threshold: float = 0.0,
        step_size: float = 0.01,
        max_steps: int = 512,
        color_field: torch.Tensor | None = None,
        surface_color: torch.Tensor | None = None,
        light_direction: torch.Tensor | None = None,
        ambient: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the Warp implementation for ``isosurface_render``."""
        return isosurface_render_warp(
            field=field,
            image_height=image_height,
            image_width=image_width,
            eye=eye,
            center=center,
            up=up,
            fov_y_degrees=fov_y_degrees,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            threshold=threshold,
            step_size=step_size,
            max_steps=max_steps,
            color_field=color_field,
            surface_color=surface_color,
            light_direction=light_direction,
            ambient=ambient,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for the isosurface renderer."""
        device = torch.device(device)
        bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
        bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
        eye = torch.tensor([0.0, 0.0, -2.5], device=device)
        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        color = torch.tensor([0.2, 0.7, 1.0], device=device)
        for label, grid_n, image_n, step_size, max_steps in cls._BENCHMARK_CASES:
            field = _sphere_field(grid_n, device)
            yield (
                label,
                (
                    field,
                    image_n,
                    image_n,
                    eye,
                    center,
                    up,
                    45.0,
                    bounds_min,
                    bounds_max,
                ),
                {
                    "threshold": 0.0,
                    "step_size": step_size,
                    "max_steps": max_steps,
                    "surface_color": color,
                },
            )


isosurface_render = IsosurfaceRender.make_function("isosurface_render")

__all__ = ["IsosurfaceRender", "isosurface_render"]
