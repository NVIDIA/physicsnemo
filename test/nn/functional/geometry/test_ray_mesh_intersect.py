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

from physicsnemo.nn.functional import ray_mesh_intersect
from physicsnemo.nn.functional.geometry import RayMeshIntersect
from physicsnemo.nn.functional.geometry.ray_mesh_intersect._warp_impl import (
    op as ray_mesh_intersect_op,
)
from test.conftest import requires_module


def _triangle_mesh(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
        torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32),
    )


@requires_module("warp")
def test_ray_mesh_intersect_default_dispatch(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.tensor([[0.0, 0.0, -1.0]], device=device)
    ray_directions = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    hit_mask, hit_distance, hit_points, face_ids, hit_normals = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
    )

    assert hit_mask.tolist() == [True]
    torch.testing.assert_close(
        hit_distance,
        torch.tensor([1.0], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hit_points,
        torch.tensor([[0.0, 0.0, 0.0]], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert face_ids.tolist() == [0]
    torch.testing.assert_close(
        hit_normals,
        torch.tensor([[0.0, 0.0, 1.0]], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )


@requires_module("warp")
def test_ray_mesh_intersect_warp_hits_and_misses(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.tensor(
        [[0.0, 0.0, -1.0], [2.0, 2.0, -1.0], [0.25, -0.25, -2.0]],
        device=device,
        dtype=torch.float32,
    )
    ray_directions = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    hit_mask, hit_distance, hit_points, face_ids, hit_normals = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=10.0,
        implementation="warp",
    )

    assert hit_mask.dtype == torch.bool
    assert face_ids.dtype == torch.int64
    assert hit_mask.tolist() == [True, False, True]
    torch.testing.assert_close(
        hit_distance,
        torch.tensor([1.0, torch.inf, 2.0], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hit_points[0],
        torch.tensor([0.0, 0.0, 0.0], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hit_points[2],
        torch.tensor([0.25, -0.25, 0.0], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert face_ids.tolist() == [0, -1, 0]
    torch.testing.assert_close(
        hit_normals[0],
        torch.tensor([0.0, 0.0, 1.0], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hit_normals[1],
        torch.zeros(3, device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert "warp" in RayMeshIntersect.available_implementations()


@requires_module("warp")
def test_ray_mesh_intersect_returns_closest_stacked_face(device: str):
    mesh_vertices = torch.tensor(
        [
            [-1.0, -1.0, 0.5],
            [1.0, -1.0, 0.5],
            [0.0, 1.0, 0.5],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    mesh_indices = torch.tensor(
        [[0, 1, 2], [3, 4, 5]],
        device=device,
        dtype=torch.int32,
    )
    ray_origins = torch.tensor([[0.0, 0.0, -1.0]], device=device)
    ray_directions = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    hit_mask, hit_distance, hit_points, face_ids, _ = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=10.0,
        implementation="warp",
    )

    assert hit_mask.tolist() == [True]
    assert face_ids.tolist() == [1]
    torch.testing.assert_close(
        hit_distance,
        torch.tensor([1.0], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hit_points,
        torch.tensor([[0.0, 0.0, 0.0]], device=device),
        atol=1.0e-6,
        rtol=0.0,
    )


@requires_module("warp")
def test_ray_mesh_intersect_returned_warp_mesh_matches_tensor_path(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.tensor(
        [[0.0, 0.0, -1.0], [2.0, 2.0, -1.0], [0.25, -0.25, -2.0]],
        device=device,
        dtype=torch.float32,
    )
    ray_directions = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    tensor_outputs = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=10.0,
        implementation="warp",
    )
    *returned_outputs, warp_mesh = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=10.0,
        return_warp_mesh=True,
        implementation="warp",
    )
    reused_outputs = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=10.0,
        warp_mesh=warp_mesh,
        implementation="warp",
    )

    assert isinstance(warp_mesh, ray_mesh_intersect_op.wp.Mesh)

    for tensor_output, returned_output, reused_output in zip(
        tensor_outputs,
        returned_outputs,
        reused_outputs,
    ):
        torch.testing.assert_close(returned_output, tensor_output)
        torch.testing.assert_close(reused_output, tensor_output)


@requires_module("warp")
def test_ray_mesh_intersect_reuses_returned_warp_mesh(device: str, monkeypatch):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.tensor([[0.0, 0.0, -1.0]], device=device)
    ray_directions = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    mesh_build_count = 0
    original_build_warp_mesh = ray_mesh_intersect_op._build_warp_mesh

    def counting_build_warp_mesh(*args, **kwargs):
        nonlocal mesh_build_count
        mesh_build_count += 1
        return original_build_warp_mesh(*args, **kwargs)

    monkeypatch.setattr(
        ray_mesh_intersect_op,
        "_build_warp_mesh",
        counting_build_warp_mesh,
    )

    *_, warp_mesh = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        return_warp_mesh=True,
        implementation="warp",
    )
    assert mesh_build_count == 1

    for _ in range(2):
        ray_mesh_intersect(
            mesh_vertices,
            mesh_indices,
            ray_origins,
            ray_directions,
            warp_mesh=warp_mesh,
            implementation="warp",
        )

    assert mesh_build_count == 1


@requires_module("warp")
def test_ray_mesh_intersect_shape_and_flat_index_compatibility(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.tensor(
        [
            [[0.0, 0.0, -1.0], [2.0, 2.0, -1.0]],
            [[0.25, -0.25, -2.0], [0.0, 0.0, 1.0]],
        ],
        device=device,
        dtype=torch.float64,
    )
    ray_directions = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        ],
        device=device,
        dtype=torch.float64,
    )

    outputs_faces = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=10.0,
        implementation="warp",
    )
    outputs_flat = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices.reshape(-1),
        ray_origins,
        ray_directions,
        max_distance=10.0,
        implementation="warp",
    )

    hit_mask, hit_distance, hit_points, face_ids, hit_normals = outputs_faces
    assert hit_mask.shape == ray_origins.shape[:-1]
    assert hit_distance.shape == ray_origins.shape[:-1]
    assert hit_points.shape == ray_origins.shape
    assert face_ids.shape == ray_origins.shape[:-1]
    assert hit_normals.shape == ray_origins.shape
    assert hit_distance.dtype == torch.float64
    assert hit_points.dtype == torch.float64
    assert hit_normals.dtype == torch.float64

    for output_faces, output_flat in zip(outputs_faces, outputs_flat):
        torch.testing.assert_close(output_faces, output_flat)


@requires_module("warp")
def test_ray_mesh_intersect_max_distance_and_zero_direction(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        device=device,
        dtype=torch.float32,
    )
    ray_directions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )

    hit_mask, hit_distance, _, face_ids, _ = ray_mesh_intersect(
        mesh_vertices,
        mesh_indices,
        ray_origins,
        ray_directions,
        max_distance=0.5,
        implementation="warp",
    )

    assert hit_mask.tolist() == [False, False]
    assert torch.isinf(hit_distance).all()
    assert face_ids.tolist() == [-1, -1]


@requires_module("warp")
def test_ray_mesh_intersect_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(RayMeshIntersect.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    hit_mask, hit_distance, hit_points, face_ids, hit_normals = (
        RayMeshIntersect.dispatch(*args, implementation="warp", **kwargs)
    )

    assert hit_mask.ndim == 1
    assert hit_distance.shape == hit_mask.shape
    assert hit_points.shape[-1] == 3
    assert face_ids.shape == hit_mask.shape
    assert hit_normals.shape[-1] == 3


@requires_module("warp")
def test_ray_mesh_intersect_error_handling(device: str):
    mesh_vertices, mesh_indices = _triangle_mesh(device)
    ray_origins = torch.zeros((2, 3), device=device)
    ray_directions = torch.ones((2, 3), device=device)

    with pytest.raises(ValueError, match="same shape"):
        ray_mesh_intersect(
            mesh_vertices,
            mesh_indices,
            ray_origins,
            torch.ones((3, 3), device=device),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="triangle-triplet"):
        ray_mesh_intersect(
            mesh_vertices,
            torch.tensor([0, 1], device=device, dtype=torch.int32),
            ray_origins,
            ray_directions,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        ray_mesh_intersect(
            mesh_vertices,
            mesh_indices,
            ray_origins,
            ray_directions,
            max_distance=0.0,
            implementation="warp",
        )

    with pytest.raises(TypeError, match="wp.Mesh"):
        ray_mesh_intersect(
            mesh_vertices,
            mesh_indices,
            ray_origins,
            ray_directions,
            warp_mesh=object(),
            implementation="warp",
        )
