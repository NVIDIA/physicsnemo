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

from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters


def _assert_equal_vertex_distances(
    vertices: torch.Tensor,
    centers: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    distances = torch.linalg.vector_norm(vertices - centers[:, None, :], dim=-1)
    torch.testing.assert_close(
        distances, distances[:, :1].expand_as(distances), atol=atol, rtol=0
    )


def test_right_triangle_circumcenter_3d() -> None:
    vertices = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )

    centers = compute_circumcenters(vertices)

    torch.testing.assert_close(centers, torch.tensor([[1.0, 1.0, 0.0]]))
    _assert_equal_vertex_distances(vertices, centers)


def test_random_triangle_circumcenters_have_equal_vertex_distances() -> None:
    generator = torch.Generator().manual_seed(123)
    vertices = torch.randn((64, 3, 3), generator=generator)
    vertices[:, 2, :] += torch.tensor([0.0, 0.0, 2.0])

    centers = compute_circumcenters(vertices)

    _assert_equal_vertex_distances(vertices, centers, atol=1e-5)


def test_reversed_triangle_orientation_has_same_circumcenter() -> None:
    vertices = torch.tensor(
        [[[0.1, 0.2, 0.3], [1.5, -0.1, 0.4], [0.2, 1.7, -0.2]]],
        dtype=torch.float32,
    )

    center = compute_circumcenters(vertices)
    reversed_center = compute_circumcenters(vertices[:, [0, 2, 1], :])

    torch.testing.assert_close(reversed_center, center, atol=1e-6, rtol=1e-6)


def test_primitive_sphere_triangle_circumcenters() -> None:
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)
    vertices = mesh.points[mesh.cells]

    centers = compute_circumcenters(vertices)

    assert torch.isfinite(centers).all()
    _assert_equal_vertex_distances(vertices, centers, atol=1e-5)


def test_near_degenerate_triangle_circumcenter_is_finite() -> None:
    vertices = torch.tensor(
        [[[0.0, 0.0, 0.0], [1e-8, 0.0, 0.0], [2e-8, 1e-12, 0.0]]],
        dtype=torch.float32,
    )

    centers = compute_circumcenters(vertices)

    assert torch.isfinite(centers).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_triangle_circumcenters_match_cpu() -> None:
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)
    vertices = mesh.points[mesh.cells]

    cpu_centers = compute_circumcenters(vertices)
    cuda_centers = compute_circumcenters(vertices.cuda()).cpu()

    torch.testing.assert_close(cuda_centers, cpu_centers, atol=1e-6, rtol=1e-6)
