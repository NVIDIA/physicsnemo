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

from physicsnemo.nn.functional import signed_distance_field
from physicsnemo.nn.functional.geometry import SignedDistanceField
from test.conftest import requires_module


# Build a simple tetrahedron surface mesh as four triangles.
def _tetrahedron_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


# Validate the warp-backed SDF implementation on a deterministic tetrahedron setup.
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_distance_field_warp(dtype: torch.dtype, device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=dtype)
    mesh_indices_flat = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
        dtype=torch.int32,
    )
    query_points = torch.tensor(
        [[1.0, 1.0, 1.0], [0.05, 0.1, 0.1]],
        device=device,
        dtype=dtype,
    )

    sdf_out, hit_points = signed_distance_field(
        mesh_vertices=mesh_vertices,
        mesh_indices=mesh_indices_flat,
        input_points=query_points,
        use_sign_winding_number=False,
    )

    torch.testing.assert_close(
        sdf_out,
        torch.tensor([1.1547, -0.05], device=device, dtype=dtype),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        hit_points,
        torch.tensor(
            [[0.33333322, 0.33333334, 0.3333334], [0.0, 0.10, 0.10]],
            device=device,
            dtype=dtype,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


# Validate SDF index-shape compatibility paths.
@requires_module("warp")
def test_signed_distance_field_index_layout_compatibility(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices_flat = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
        dtype=torch.int32,
    )
    mesh_indices_faces = mesh_indices_flat.reshape(-1, 3)
    query_points = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    # Accept both flattened and (n_faces, 3) connectivity layouts.
    sdf_flat, hit_flat = signed_distance_field(
        mesh_vertices, mesh_indices_flat, query_points
    )
    sdf_faces, hit_faces = signed_distance_field(
        mesh_vertices, mesh_indices_faces, query_points
    )
    torch.testing.assert_close(sdf_flat, sdf_faces)
    torch.testing.assert_close(hit_flat, hit_faces)


# Validate benchmark input generation contract for SDF.
@requires_module("warp")
def test_signed_distance_field_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(SignedDistanceField.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    sdf_out, hit_points = SignedDistanceField.dispatch(
        *args,
        implementation="warp",
        **kwargs,
    )
    assert sdf_out.ndim == 1
    assert hit_points.ndim == 2
    assert hit_points.shape[1] == 3


# Validate SDF input and shape error handling paths.
@requires_module("warp")
def test_signed_distance_field_error_handling(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices_flat = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
        dtype=torch.int32,
    )
    query_points = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    # Query points must have xyz in the last dimension.
    bad_queries = torch.randn(4, 2, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        signed_distance_field(mesh_vertices, mesh_indices_flat, bad_queries)

    # 2D mesh indices must be shaped as (n_faces, 3).
    bad_connectivity_shape = torch.zeros(4, 4, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="shape \\(n_faces, 3\\)"):
        signed_distance_field(mesh_vertices, bad_connectivity_shape, query_points)

    # Connectivity may be 1D flattened or 2D triangular faces only.
    bad_connectivity_rank = torch.zeros(1, 2, 3, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="1D flattened indices or 2D"):
        signed_distance_field(mesh_vertices, bad_connectivity_rank, query_points)
