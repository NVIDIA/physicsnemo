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

from ._warp_impl import point_cloud_render_warp


class PointCloudRender(FunctionSpec):
    """Rasterize a 3D point cloud into RGBA and depth images.

    The Warp implementation uses one pass to atomically select the nearest
    point per covered pixel and a second pass to resolve the winning color and
    depth.

    Args:
        points: Point positions with shape ``(num_points, 3)``.
        image_height: Output image height.
        image_width: Output image width.
        eye: Camera position with shape ``(3,)``.
        center: Camera look-at point with shape ``(3,)``.
        up: Camera up direction with shape ``(3,)``.
        fov_y_degrees: Vertical field of view in degrees.
        point_colors: Optional RGB/RGBA colors with one color per point.
        point_color: Optional uniform RGB/RGBA point color.
        point_size: Square point size in pixels.
        near: Near clip distance.
        far: Far clip distance.
        implementation: Explicit implementation name. Currently only ``"warp"``
            is registered.

    Returns:
        Tuple of ``(rgba, depth)`` image tensors.
    """

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=1.0.0",), rank=0, baseline=True
    )
    def warp_forward(
        points: torch.Tensor,
        image_height: int,
        image_width: int,
        eye: torch.Tensor | Sequence[float],
        center: torch.Tensor | Sequence[float],
        up: torch.Tensor | Sequence[float],
        fov_y_degrees: float,
        point_colors: torch.Tensor | None = None,
        point_color: torch.Tensor | None = None,
        point_size: int = 1,
        near: float = 0.01,
        far: float = 1.0e8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Warp implementation for ``point_cloud_render``."""
        return point_cloud_render_warp(
            points=points,
            image_height=image_height,
            image_width=image_width,
            eye=eye,
            center=center,
            up=up,
            fov_y_degrees=fov_y_degrees,
            point_colors=point_colors,
            point_color=point_color,
            point_size=point_size,
            near=near,
            far=far,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for point-cloud rendering."""
        device = torch.device(device)
        points = torch.tensor(
            [[-0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.4, 0.0, 0.0]],
            device=device,
        )
        colors = torch.tensor(
            [[255, 32, 32], [32, 255, 32], [32, 128, 255]],
            device=device,
            dtype=torch.uint8,
        )
        eye = torch.tensor([0.0, 0.0, -2.0], device=device)
        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        yield (
            "points3-img32",
            (points, 32, 32, eye, center, up, 45.0),
            {"point_colors": colors},
        )


point_cloud_render = PointCloudRender.make_function("point_cloud_render")

__all__ = ["PointCloudRender", "point_cloud_render"]
