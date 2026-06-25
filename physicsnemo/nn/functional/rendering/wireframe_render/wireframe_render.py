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

from ._warp_impl import wireframe_render_warp


class WireframeRender(FunctionSpec):
    """Rasterize 3D line segments into RGBA and depth images.

    One Warp thread projects and rasterizes each segment. Depth writes are
    resolved atomically and all segments use the same line color.

    Args:
        edges: Line segments with shape ``(num_edges, 2, 3)`` or
            ``(num_edges, 6)``.
        image_height: Output image height.
        image_width: Output image width.
        eye: Camera position with shape ``(3,)``.
        center: Camera look-at point with shape ``(3,)``.
        up: Camera up direction with shape ``(3,)``.
        fov_y_degrees: Vertical field of view in degrees.
        line_color: Optional uniform RGB/RGBA line color.
        line_thickness: Line thickness in pixels.
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
        edges: torch.Tensor,
        image_height: int,
        image_width: int,
        eye: torch.Tensor | Sequence[float],
        center: torch.Tensor | Sequence[float],
        up: torch.Tensor | Sequence[float],
        fov_y_degrees: float,
        line_color: torch.Tensor | None = None,
        line_thickness: int = 1,
        near: float = 0.01,
        far: float = 1.0e8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Warp implementation for ``wireframe_render``."""
        return wireframe_render_warp(
            edges=edges,
            image_height=image_height,
            image_width=image_width,
            eye=eye,
            center=center,
            up=up,
            fov_y_degrees=fov_y_degrees,
            line_color=line_color,
            line_thickness=line_thickness,
            near=near,
            far=far,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for wireframe rendering."""
        device = torch.device(device)
        edges = torch.tensor(
            [
                [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]],
                [[0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            ],
            device=device,
        )
        eye = torch.tensor([0.0, 0.0, -2.0], device=device)
        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        line_color = torch.tensor([1.0, 0.9, 0.2], device=device)
        yield (
            "edges2-img32",
            (edges, 32, 32, eye, center, up, 45.0),
            {"line_color": line_color},
        )


wireframe_render = WireframeRender.make_function("wireframe_render")

__all__ = ["WireframeRender", "wireframe_render"]
