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

from ._warp_impl import mesh_raycast_warp


class MeshRaycast(FunctionSpec):
    """Render a triangle mesh with Warp ray queries.

    ``mesh_raycast`` builds a Warp ``Mesh`` acceleration structure from triangle
    vertices and indices, casts one camera ray per output pixel, and returns
    image-space ``(rgba, depth, normal)`` buffers. Mesh color may be uniform,
    per vertex, or per face. ``uint8`` colors are accepted and normalized to
    ``[0, 1]`` internally.

    Args:
        mesh_vertices: Vertex positions with shape ``(num_vertices, 3)``.
        mesh_indices: Triangle connectivity with shape ``(num_faces, 3)`` or a
            flattened equivalent.
        image_height: Output image height.
        image_width: Output image width.
        eye: Camera position with shape ``(3,)``.
        center: Camera look-at point with shape ``(3,)``.
        up: Camera up direction with shape ``(3,)``.
        fov_y_degrees: Vertical field of view in degrees.
        vertex_colors: Optional RGB/RGBA colors with one color per vertex.
        face_colors: Optional RGB/RGBA colors with one color per triangle.
        surface_color: Optional uniform RGB/RGBA color used when per-element
            color arrays are omitted.
        light_direction: Optional surface-to-light direction with shape ``(3,)``.
        ambient: Ambient lighting coefficient in ``[0, 1]``.
        max_distance: Maximum ray distance.
        implementation: Explicit implementation name. Currently only ``"warp"``
            is registered.

    Returns:
        Tuple of ``(rgba, depth, normal)`` image tensors. Missed pixels have zero
        alpha, infinite depth, and zero normal.
    """

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=1.0.0",), rank=0, baseline=True
    )
    def warp_forward(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        image_height: int,
        image_width: int,
        eye: torch.Tensor | Sequence[float],
        center: torch.Tensor | Sequence[float],
        up: torch.Tensor | Sequence[float],
        fov_y_degrees: float,
        vertex_colors: torch.Tensor | None = None,
        face_colors: torch.Tensor | None = None,
        surface_color: torch.Tensor | None = None,
        light_direction: torch.Tensor | None = None,
        ambient: float = 0.2,
        max_distance: float = 1.0e8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the Warp implementation for ``mesh_raycast``."""
        return mesh_raycast_warp(
            mesh_vertices=mesh_vertices,
            mesh_indices=mesh_indices,
            image_height=image_height,
            image_width=image_width,
            eye=eye,
            center=center,
            up=up,
            fov_y_degrees=fov_y_degrees,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
            surface_color=surface_color,
            light_direction=light_direction,
            ambient=ambient,
            max_distance=max_distance,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield benchmark inputs for mesh raycasting."""
        device = torch.device(device)
        mesh_vertices = torch.tensor(
            [
                [-0.9, -0.7, 0.0],
                [0.9, -0.7, 0.0],
                [0.0, 0.85, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        mesh_indices = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
        vertex_colors = torch.tensor(
            [[255, 64, 64], [64, 255, 64], [64, 128, 255]],
            device=device,
            dtype=torch.uint8,
        )
        eye = torch.tensor([0.0, 0.0, -2.5], device=device)
        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        yield (
            "triangle-img64",
            (mesh_vertices, mesh_indices, 64, 64, eye, center, up, 45.0),
            {"vertex_colors": vertex_colors},
        )


mesh_raycast = MeshRaycast.make_function("mesh_raycast")

__all__ = ["MeshRaycast", "mesh_raycast"]
