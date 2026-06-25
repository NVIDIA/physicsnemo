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

from ._warp_impl import volume_render_warp


class VolumeRender(FunctionSpec):
    """Render an RGBA volume with front-to-back ray marching.

    Args:
        rgba_volume: RGBA volume with shape ``(nx, ny, nz, 4)``. ``uint8`` input
            is normalized to ``[0, 1]`` internally.
        image_height: Output image height.
        image_width: Output image width.
        eye: Camera position with shape ``(3,)``.
        center: Camera look-at point with shape ``(3,)``.
        up: Camera up direction with shape ``(3,)``.
        fov_y_degrees: Vertical field of view in degrees.
        bounds_min: Minimum world-space volume bound with shape ``(3,)``.
        bounds_max: Maximum world-space volume bound with shape ``(3,)``.
        step_size: Ray-marching step size in world units.
        max_steps: Maximum number of march steps per pixel.
        opacity_threshold: Stop marching after this accumulated opacity.
        depth_threshold: Accumulated opacity needed before depth is recorded.
        implementation: Explicit implementation name. Currently only ``"warp"``
            is registered.

    Returns:
        Tuple of ``(rgba, depth)`` image tensors.
    """

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=1.0.0",), rank=0, baseline=True
    )
    def warp_forward(
        rgba_volume: torch.Tensor,
        image_height: int,
        image_width: int,
        eye: torch.Tensor | Sequence[float],
        center: torch.Tensor | Sequence[float],
        up: torch.Tensor | Sequence[float],
        fov_y_degrees: float,
        bounds_min: torch.Tensor | Sequence[float],
        bounds_max: torch.Tensor | Sequence[float],
        step_size: float = 0.01,
        max_steps: int = 512,
        opacity_threshold: float = 0.95,
        depth_threshold: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Warp implementation for ``volume_render``."""
        return volume_render_warp(
            rgba_volume=rgba_volume,
            image_height=image_height,
            image_width=image_width,
            eye=eye,
            center=center,
            up=up,
            fov_y_degrees=fov_y_degrees,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            step_size=step_size,
            max_steps=max_steps,
            opacity_threshold=opacity_threshold,
            depth_threshold=depth_threshold,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for volume rendering."""
        device = torch.device(device)
        rgba_volume = torch.zeros((24, 24, 24, 4), device=device, dtype=torch.uint8)
        rgba_volume[..., 0] = 96
        rgba_volume[..., 1] = 192
        rgba_volume[..., 2] = 255
        rgba_volume[7:17, 7:17, 7:17, 3] = 96
        eye = torch.tensor([0.0, 0.0, -2.5], device=device)
        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
        bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
        yield (
            "cube24-img32",
            (rgba_volume, 32, 32, eye, center, up, 45.0, bounds_min, bounds_max),
            {"step_size": 0.08, "max_steps": 80},
        )


volume_render = VolumeRender.make_function("volume_render")

__all__ = ["VolumeRender", "volume_render"]
