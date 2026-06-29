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

from physicsnemo.nn.functional import mesh_raycast
from physicsnemo.nn.functional.rendering import MeshRaycast
from test.conftest import requires_module


def _triangle_mesh(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[-0.9, -0.8, 0.0], [0.9, -0.8, 0.0], [0.0, 0.9, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
    )


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.0], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_mesh_raycast_warp(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    vertex_colors = torch.tensor(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        device=device,
        dtype=torch.uint8,
    )
    eye, center, up = _camera(device)

    rgba, depth, normal = mesh_raycast(
        mesh_vertices,
        mesh_indices,
        31,
        31,
        eye,
        center,
        up,
        45.0,
        vertex_colors=vertex_colors,
        light_direction=torch.tensor([0.0, 0.0, -1.0], device=device),
        implementation="warp",
    )

    assert rgba.shape == (31, 31, 4)
    assert depth.shape == (31, 31)
    assert normal.shape == (31, 31, 3)
    assert float(rgba[15, 15, 3]) == 1.0
    torch.testing.assert_close(
        depth[15, 15], torch.tensor(2.0, device=device), atol=1.0e-5, rtol=0.0
    )
    assert float(normal[15, 15, 2]) < -0.9
    assert float(rgba[15, 15, :3].max()) > 0.0
    assert torch.isinf(depth[0, 0])
    assert "warp" in MeshRaycast.available_implementations()


@requires_module("warp")
def test_mesh_raycast_face_colors_and_flat_indices(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    face_colors = torch.tensor([[64, 192, 255, 255]], device=device, dtype=torch.uint8)
    eye, center, up = _camera(device)

    rgba_faces, depth_faces, normal_faces = mesh_raycast(
        mesh_vertices,
        mesh_indices,
        17,
        17,
        eye,
        center,
        up,
        45.0,
        face_colors=face_colors,
        light_direction=torch.tensor([0.0, 0.0, -1.0], device=device),
        implementation="warp",
    )
    rgba_flat, depth_flat, normal_flat = mesh_raycast(
        mesh_vertices,
        mesh_indices.reshape(-1),
        17,
        17,
        eye,
        center,
        up,
        45.0,
        face_colors=face_colors,
        light_direction=torch.tensor([0.0, 0.0, -1.0], device=device),
        implementation="warp",
    )

    torch.testing.assert_close(rgba_faces, rgba_flat)
    torch.testing.assert_close(depth_faces, depth_flat)
    torch.testing.assert_close(normal_faces, normal_flat)


@requires_module("warp")
def test_mesh_raycast_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(MeshRaycast.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    rgba, depth, normal = MeshRaycast.dispatch(*args, implementation="warp", **kwargs)
    assert rgba.shape[-1] == 4
    assert normal.shape[-1] == 3
    assert depth.shape == rgba.shape[:2]


@requires_module("warp")
def test_mesh_raycast_error_handling(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    eye, center, up = _camera(device)

    with pytest.raises(ValueError, match="either vertex_colors or face_colors"):
        mesh_raycast(
            mesh_vertices,
            mesh_indices,
            16,
            16,
            eye,
            center,
            up,
            45.0,
            vertex_colors=torch.zeros(3, 3, device=device),
            face_colors=torch.zeros(1, 3, device=device),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="complete triangle faces"):
        mesh_raycast(
            mesh_vertices,
            torch.tensor([0, 1], device=device, dtype=torch.int32),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="fov_y_degrees"):
        mesh_raycast(
            mesh_vertices,
            mesh_indices,
            16,
            16,
            eye,
            center,
            up,
            180.0,
            implementation="warp",
        )
