# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
# ruff: noqa: E402


import pytest
import torch
from pytest_utils import import_or_fail


def tet_verts(flip_x=1):
    tet = torch.tensor(
        [
            flip_x * 0,
            0,
            0,  # bottom
            flip_x * 0,
            1,
            0,
            flip_x * 1,
            0,
            0,
            flip_x * 0,
            0,
            0,  # front
            flip_x * 1,
            0,
            0,
            flip_x * 0,
            0,
            1,
            flip_x * 0,
            0,
            0,  # left
            flip_x * 0,
            0,
            1,
            flip_x * 0,
            1,
            0,
            flip_x * 1,
            0,
            0,  # "top"
            flip_x * 0,
            1,
            0,
            flip_x * 0,
            0,
            1,
        ],
        dtype=torch.float64,
    )

    return tet


@import_or_fail("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sdf(pytestconfig, dtype, device):
    from physicsnemo.utils.sdf import signed_distance_field

    mesh_vertices = tet_verts().reshape(-1, 3)

    if device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mesh_indices = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int32
    )
    input_points = torch.tensor([[1, 1, 1], [0.05, 0.1, 0.1]], dtype=torch.float64)

    mesh_vertices = mesh_vertices.to(dtype)
    input_points = input_points.to(dtype)

    sdf_tet, sdf_hit_point = signed_distance_field(
        mesh_vertices,
        mesh_indices,
        input_points,
        use_sign_winding_number=False,
    )
    expected_sdf = torch.tensor([1.1547, -0.05], dtype=dtype)

    print(f"Input shape: {input_points.shape}")
    print(f"sdf_tet shape: {sdf_tet.shape}")
    print(f"expected_sdf shape: {expected_sdf.shape}")
    print(f"sdf_tet: {sdf_tet}")
    print(f"expected_sdf: {expected_sdf}")
    assert torch.allclose(sdf_tet, expected_sdf, atol=1e-7)

    assert torch.allclose(
        sdf_hit_point,
        torch.tensor(
            [[0.33333322, 0.33333334, 0.3333334], [0.0, 0.10, 0.10]], dtype=dtype
        ),
        atol=1e-7,
    )
