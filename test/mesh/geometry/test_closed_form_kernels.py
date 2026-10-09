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

"""Closed-form geometry kernels agree with the batched linear algebra they replace.

Each reference below is the batched ``torch.linalg`` formulation that the
closed forms replace, kept here verbatim in spirit so that any drift in the
fast paths shows up as a numerical mismatch.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry._angles import compute_vertex_angles
from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
from physicsnemo.mesh.utilities._tolerances import safe_eps
from physicsnemo.mesh.utilities._topology import extract_unique_edges
from physicsnemo.utils._small_linalg import small_det, small_inverse
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget


def _random_simplices(
    n_cells: int,
    n_manifold_dims: int,
    n_spatial_dims: int,
    device,
    scale: float = 1.0,
    sliver_thickness: float = 1e-6,
) -> Mesh:
    """Disjoint random simplices, a third of them squashed into slivers."""
    generator = torch.Generator().manual_seed(100 * n_manifold_dims + n_spatial_dims)
    n_verts = n_manifold_dims + 1
    points = torch.randn(
        n_cells, n_verts, n_spatial_dims, generator=generator, dtype=torch.float64
    )
    # Squash every third cell towards the hyperplane of its first n vertices
    sliver = torch.arange(n_cells) % 3 == 0
    points[sliver, -1] = (
        points[sliver, :-1].mean(dim=1) + sliver_thickness * points[sliver, -1]
    )
    return Mesh(
        points=(scale * points.reshape(-1, n_spatial_dims)).to(device),
        cells=torch.arange(n_cells * n_verts, device=device).reshape(n_cells, n_verts),
    )


def _reference_vertex_angles(mesh: Mesh) -> torch.Tensor:
    """Generalized vertex angles from batched determinants of correlation matrices."""
    vertices = mesh.points[mesh.cells].double()
    n_edges = mesh.n_manifold_dims
    edges = torch.stack(
        [vertices.roll(-(i + 1), dims=1) - vertices for i in range(n_edges)], dim=2
    )
    unit = edges / edges.norm(dim=-1, keepdim=True).clamp(min=safe_eps(torch.float64))
    correlation = torch.einsum("cvid,cvjd->cvij", unit, unit)
    rows, cols = torch.triu_indices(n_edges, n_edges, offset=1)
    angles = 2.0 * torch.atan2(
        torch.linalg.det(correlation).abs().sqrt(),
        1.0 + correlation[:, :, rows, cols].sum(dim=-1),
    )
    return angles.to(mesh.points.dtype)


def _reference_cotan_weights(mesh: Mesh) -> torch.Tensor:
    """FEM cotangent weights via batched Gram inverses and C = H G^-1 H^T."""
    n = mesh.n_manifold_dims
    dtype = mesh.points.dtype
    unique_edges, inverse_indices = extract_unique_edges(mesh)
    vertices = mesh.points[mesh.cells]
    E = vertices[:, 1:, :] - vertices[:, :1, :]
    G = E @ E.transpose(-1, -2)
    length_scale = E.norm(dim=-1).mean(dim=-1)
    has_extent = length_scale > torch.finfo(dtype).tiny ** 0.5
    g_scale = torch.where(has_extent, length_scale, 1.0).square()
    is_degenerate = ~has_extent | (
        torch.linalg.det(G / g_scale[:, None, None]).abs() < 1e-12
    )
    eye = torch.eye(n, dtype=dtype, device=G.device)
    G = torch.where(is_degenerate[:, None, None], g_scale[:, None, None] * eye, G)
    H = torch.cat([-torch.ones(1, n, dtype=dtype), torch.eye(n, dtype=dtype)]).to(
        G.device
    )
    C = H @ torch.linalg.inv(G) @ H.T
    rows, cols = torch.triu_indices(n + 1, n + 1, offset=1)
    per_cell = -mesh.cell_areas[:, None] * C[:, rows, cols]
    weights = torch.zeros(len(unique_edges), dtype=dtype, device=G.device)
    return weights.scatter_add_(0, inverse_indices, per_cell.reshape(-1))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_shape", [(64,), (3, 5)])
def test_small_det_and_inverse_match_linalg(n, batch_shape, device):
    generator = torch.Generator().manual_seed(n)
    matrices = torch.randn(*batch_shape, n, n, generator=generator, dtype=torch.float64)
    matrices = (matrices + 3.0 * torch.eye(n, dtype=torch.float64)).to(device)

    torch.testing.assert_close(small_det(matrices), torch.linalg.det(matrices))
    torch.testing.assert_close(small_inverse(matrices), torch.linalg.inv(matrices))


@pytest.mark.parametrize(
    "n_manifold_dims, n_spatial_dims",
    [(1, 2), (2, 2), (2, 4), (3, 3), (3, 4), (4, 4)],
)
def test_vertex_angles_match_determinant_formula(
    n_manifold_dims, n_spatial_dims, device
):
    mesh = _random_simplices(300, n_manifold_dims, n_spatial_dims, device)
    torch.testing.assert_close(
        compute_vertex_angles(mesh),
        _reference_vertex_angles(mesh),
        rtol=1e-6,
        atol=1e-6,
    )


def test_vertex_angles_of_nearly_flat_cells(device):
    """Accurate where det(C) cancels: a cap triangle and a flat tetrahedron."""
    # Cap triangle with an angle of pi - 1e-9 at the origin: the same angles in
    # 2D as in 3D, summing to pi
    cap = torch.tensor([[0.0, 0.0], [-1.0, 0.0], [1.0, 1e-9]], device=device)
    cells = torch.tensor([[0, 1, 2]], device=device)
    angles_2d = compute_vertex_angles(Mesh(points=cap, cells=cells))
    angles_3d = compute_vertex_angles(
        Mesh(points=torch.nn.functional.pad(cap, (0, 1)), cells=cells)
    )
    torch.testing.assert_close(angles_2d, angles_3d)
    torch.testing.assert_close(angles_2d.sum(), torch.tensor(torch.pi, device=device))

    # Tetrahedron of height 1e-9, against Van Oosterom-Strackee on the raw edges
    tet = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.25, 0.25, 1e-9]],
        dtype=torch.float64,
        device=device,
    )
    a, b, c = (tet.roll(-i, dims=0) - tet for i in (1, 2, 3))
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    triple = (a * torch.linalg.cross(b, c)).sum(dim=-1).abs()
    dots = (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
    expected = 2.0 * torch.atan2(triple, la * lb * lc + dots)
    mesh = Mesh(points=tet, cells=torch.tensor([[0, 1, 2, 3]], device=device))
    torch.testing.assert_close(
        compute_vertex_angles(mesh)[0], expected, rtol=1e-6, atol=0
    )


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize(
    "n_manifold_dims, n_spatial_dims",
    [(1, 3), (2, 2), (2, 3), (3, 3), (4, 4)],
)
def test_cotan_weights_match_gram_inverse(
    n_manifold_dims, n_spatial_dims, scale, device
):
    """Equal across scales, and for degenerate cells, to the batched inverse."""
    from physicsnemo.mesh.primitives.planar import structured_grid
    from physicsnemo.mesh.primitives.volumes import cube_volume

    if (n_manifold_dims, n_spatial_dims) == (2, 2):
        mesh = structured_grid.load(n_x=6, n_y=5, device=device)
    elif (n_manifold_dims, n_spatial_dims) == (3, 3):
        mesh = cube_volume.load(subdivisions=3, device=device)
    else:
        # Slivers this thin are far below the degeneracy threshold
        mesh = _random_simplices(
            60, n_manifold_dims, n_spatial_dims, device, sliver_thickness=1e-9
        )
    # Append a cell whose vertices coincide: degenerate at every scale
    n_verts = mesh.cells.shape[1]
    collapsed = torch.arange(n_verts, device=device) + mesh.n_points
    mesh = Mesh(
        points=scale
        * torch.cat([mesh.points, mesh.points[:1].expand(n_verts, -1)]).double(),
        cells=torch.cat([mesh.cells, collapsed[None]]),
    )

    weights, edges = compute_cotan_weights_fem(mesh)
    expected = _reference_cotan_weights(mesh)

    torch.testing.assert_close(edges, extract_unique_edges(mesh)[0])
    assert weights.isfinite().all()
    torch.testing.assert_close(
        weights, expected, rtol=1e-7, atol=1e-12 * expected.abs().max()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_manifold_dims, n_spatial_dims", [(2, 2), (3, 3)])
def test_vertex_angles_are_sync_free(n_manifold_dims, n_spatial_dims):
    mesh = _random_simplices(1000, n_manifold_dims, n_spatial_dims, "cuda")
    compute_vertex_angles(mesh)  # warm up lazy CUDA initialization
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        compute_vertex_angles(mesh)
    torch.cuda.synchronize()
