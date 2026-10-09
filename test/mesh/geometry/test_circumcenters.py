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


def _circumcenters_solve(vertices: torch.Tensor) -> torch.Tensor:
    """The former square-system formulation (batched solve), as reference."""
    v0 = vertices[:, 0, :]
    relative_vecs = vertices[:, 1:, :] - v0.unsqueeze(1)
    rhs = (relative_vecs**2).sum(dim=-1).unsqueeze(-1)
    return v0 + torch.linalg.solve(2 * relative_vecs, rhs).squeeze(-1)


@pytest.mark.parametrize("n_dims", [2, 3, 4])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_square_circumcenters_match_solve(device, n_dims, scale, dtype) -> None:
    """The closed form matches the former batched solve on regular and thin
    full-dimensional cells (triangles, tetrahedra, 4-simplices) at any scale."""
    generator = torch.Generator().manual_seed(n_dims)
    vertices = torch.randn(
        256, n_dims + 1, n_dims, generator=generator, dtype=torch.float64
    )
    # Thin cells: the last vertex 1e-2 away from the facet spanned by the others.
    weights = torch.rand(128, n_dims, generator=generator, dtype=torch.float64)
    weights = weights / weights.sum(dim=1, keepdim=True)
    vertices[:128, -1] = (weights[:, :, None] * vertices[:128, :-1]).sum(dim=1)
    vertices[:128, -1] += 1e-2 * torch.randn(
        128, n_dims, generator=generator, dtype=torch.float64
    )
    vertices = (scale * (vertices + 3.0)).to(device=device, dtype=dtype)

    centers = compute_circumcenters(vertices)

    # Reference in float64 on the same (rounded) inputs.
    reference = _circumcenters_solve(vertices.double())
    radius = (vertices.double() - reference[:, None, :]).norm(dim=-1).mean(dim=-1)
    error = (centers.double() - reference).norm(dim=-1) / radius
    assert float(error.max()) < (3e-2 if dtype == torch.float32 else 1e-9)
    old_error = (_circumcenters_solve(vertices).double() - reference).norm(
        dim=-1
    ) / radius
    eps = torch.finfo(dtype).eps
    assert float(error.median()) < 4 * float(old_error.median()) + 16 * eps


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_degenerate_square_cells_return_centroid(device, dtype) -> None:
    """Numerically singular full-dimensional cells have no circumcenter; they
    return their centroid (the former lstsq fallback gave dtype-dependent
    points, e.g. (0.3, 0.6, 0) in float64 but (0.5, 0.5, 0) in float32 for
    the coplanar square below)."""
    triangles = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # collinear
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],  # two coincident vertices
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],  # a point
        ],
        dtype=dtype,
        device=device,
    )
    tetrahedra = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=dtype,
        device=device,
    )
    for vertices in (triangles, tetrahedra):
        vertices = vertices.clone().requires_grad_(True)
        centers = compute_circumcenters(vertices)
        torch.testing.assert_close(centers, vertices.mean(dim=1))
        centers.sum().backward()
        assert torch.isfinite(vertices.grad).all()


def test_near_degenerate_square_cell_keeps_distant_circumcenter(device) -> None:
    """Only numerically singular cells take the centroid: a nearly collinear
    triangle keeps its exact, distant circumcenter."""
    vertices = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 1e-10]]], dtype=torch.float64, device=device
    )

    centers = compute_circumcenters(vertices)

    torch.testing.assert_close(
        centers, _circumcenters_solve(vertices), rtol=1e-9, atol=0
    )
    assert float(centers[0, 1]) > 1e9


@pytest.mark.parametrize(("n_manifold_dims", "n_dims"), [(2, 4), (3, 4), (3, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_embedded_circumcenters_match_lstsq(
    device, n_manifold_dims, n_dims, dtype
) -> None:
    """Cells embedded in a higher-dimensional space get the minimum-norm
    solution, which is the circumcenter in their affine hull, also when thin."""
    generator = torch.Generator().manual_seed(n_dims)
    vertices = torch.randn(
        256, n_manifold_dims + 1, n_dims, generator=generator, dtype=torch.float64
    )
    # Thin cells: the last vertex 1e-2 away from the facet spanned by the others.
    vertices[:128, -1] = vertices[:128, :-1].mean(dim=1) + 1e-2 * vertices[:128, -1]
    vertices = vertices.to(device=device, dtype=dtype)

    centers = compute_circumcenters(vertices)

    # Reference: CPU float64 lstsq (minimum norm) on the same (rounded) inputs.
    v0 = vertices.double().cpu()[:, 0, :]
    relative_vecs = vertices.double().cpu()[:, 1:, :] - v0[:, None, :]
    rhs = (relative_vecs**2).sum(dim=-1, keepdim=True)
    reference = v0 + torch.linalg.lstsq(2 * relative_vecs, rhs).solution.squeeze(-1)
    radius = (vertices.double().cpu() - reference[:, None, :]).norm(dim=-1).mean(dim=-1)
    error = (centers.double().cpu() - reference).norm(dim=-1) / radius
    assert float(error.max()) < (1e-2 if dtype == torch.float32 else 1e-10)
