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

import itertools

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry._angles import compute_vertex_angles
from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
from physicsnemo.mesh.utilities._tolerances import safe_eps
from physicsnemo.mesh.utilities._topology import extract_unique_edges
from physicsnemo.utils._small_linalg import (
    best_base_first,
    normalized_gram_inverse,
    normalized_inverse,
    small_det,
    small_inverse,
    undo_base_rotation,
)
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


def _thin_simplices(
    n_cells: int,
    n_manifold_dims: int,
    n_spatial_dims: int,
    n_thin: int,
    thickness: float,
    device,
) -> Mesh:
    """Disjoint random float32 simplices, ``thickness`` thin in ``n_thin`` directions.

    One thin direction makes slivers; two make needles.
    """
    generator = torch.Generator().manual_seed(10 * n_manifold_dims + n_thin)
    extent = torch.ones(n_manifold_dims, dtype=torch.float64)
    extent[n_manifold_dims - n_thin :] = thickness
    shape = (n_cells, n_manifold_dims + 1, n_manifold_dims)
    local = extent * torch.randn(*shape, generator=generator, dtype=torch.float64)
    frames = torch.linalg.qr(
        torch.randn(
            n_cells,
            n_spatial_dims,
            n_manifold_dims,
            generator=generator,
            dtype=torch.float64,
        )
    ).Q
    points = local @ frames.transpose(-1, -2) + torch.randn(
        n_cells, 1, n_spatial_dims, generator=generator, dtype=torch.float64
    )
    return Mesh(
        points=points.reshape(-1, n_spatial_dims).float().to(device),
        cells=torch.arange(n_cells * shape[1], device=device).reshape(n_cells, -1),
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
    """FEM cotangent weights via batched Gram inverses and C = H G^-1 H^T.

    With the edges from the best base vertex (best_base_first) at unit length,
    G = D G_hat D. Degenerate cells, with det(G_hat) < 1e-12 or an edge of no
    length, invert the identity in place of G_hat, as the closed form does.
    """
    n = mesh.n_manifold_dims
    dtype = mesh.points.dtype
    unique_edges, inverse_indices = extract_unique_edges(mesh)
    vertices, base = best_base_first(mesh.points[mesh.cells])
    E = vertices[:, 1:, :] - vertices[:, :1, :]
    lengths = E.norm(dim=-1)
    has_length = lengths > torch.finfo(dtype).tiny ** 0.5
    lengths = torch.where(has_length, lengths, 1.0)
    E_hat = E / lengths[..., None]
    G_hat = E_hat @ E_hat.transpose(-1, -2)
    is_degenerate = ~has_length.all(dim=-1) | (torch.linalg.det(G_hat) < 1e-12)
    eye = torch.eye(n, dtype=dtype, device=G_hat.device)
    G_hat = torch.where(is_degenerate[:, None, None], eye, G_hat)
    H = torch.cat([-torch.ones(1, n, dtype=dtype), torch.eye(n, dtype=dtype)]).to(
        G_hat.device
    )
    # Unchecked, as main inverted G: LAPACK may meet an exactly zero pivot in float32
    G_inv = torch.linalg.inv_ex(G_hat, check_errors=False).inverse / (
        lengths[:, :, None] * lengths[:, None, :]
    )
    C = H @ G_inv @ H.T
    # Back from the rotated vertex order: vertex a of the cell is at (a - base) % (n + 1)
    position = (torch.arange(n + 1, device=C.device) - base[:, None]) % (n + 1)
    C = C.gather(1, position[:, :, None].expand_as(C)).gather(
        2, position[:, None, :].expand_as(C)
    )
    rows, cols = torch.triu_indices(n + 1, n + 1, offset=1)
    per_cell = -mesh.cell_areas[:, None] * C[:, rows, cols]
    weights = torch.zeros(len(unique_edges), dtype=dtype, device=G_hat.device)
    return weights.scatter_add_(0, inverse_indices, per_cell.reshape(-1))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_shape", [(64,), (3, 5)])
def test_small_det_and_inverse_match_linalg(n, batch_shape, device):
    generator = torch.Generator().manual_seed(n)
    matrices = torch.randn(*batch_shape, n, n, generator=generator, dtype=torch.float64)
    matrices = (matrices + 3.0 * torch.eye(n, dtype=torch.float64)).to(device)

    torch.testing.assert_close(small_det(matrices), torch.linalg.det(matrices))
    torch.testing.assert_close(small_inverse(matrices), torch.linalg.inv(matrices))


def test_best_base_first_avoids_the_tip_of_a_needle(device):
    """The base vertex is one of the three at the short edges, never the tip,
    and undo_base_rotation restores the input order."""
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2000.0]],
        device=device,
    )
    orders = torch.tensor(list(itertools.permutations(range(4))), device=device)

    rotated, base = best_base_first(vertices[orders])

    assert not (rotated[:, 0, 2] == 2000.0).any()
    order = (torch.arange(4, device=device) + base[:, None]) % 4
    torch.testing.assert_close(
        rotated, vertices[orders].gather(1, order[:, :, None].expand(-1, -1, 3))
    )
    torch.testing.assert_close(
        undo_base_rotation(rotated[..., 2], base), vertices[orders][..., 2]
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_normalized_inverse_ignores_row_lengths(n, device):
    """Rows of very different lengths (a stretched cell) are not singular."""
    generator = torch.Generator().manual_seed(n)
    directions = torch.randn(64, n, n, generator=generator, dtype=torch.float64)
    directions = directions + 3.0 * torch.eye(n, dtype=torch.float64)
    lengths = 10.0 ** (
        16 * torch.rand(64, n, generator=generator, dtype=torch.float64) - 8
    )
    matrices = (lengths[..., None] * directions).to(device)

    inverse, row_lengths, is_singular = normalized_inverse(matrices)

    assert not is_singular.any()
    # inv(M) = inv(M_hat) D^-1
    identity = (inverse / row_lengths[..., None, :]) @ matrices
    torch.testing.assert_close(
        identity, torch.eye(n, dtype=torch.float64, device=device).expand_as(identity)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalized_inverse_flags_singular_matrices(dtype, device):
    """Zero rows and rows dependent up to rounding are singular; gradients are finite."""
    e1 = torch.tensor([0.3, -0.7, 0.2], dtype=dtype)
    e2 = torch.tensor([0.9, 0.4, -0.1], dtype=dtype)
    matrices = torch.stack(
        [
            torch.stack([e1, e2, torch.zeros(3, dtype=dtype)]),  # zero row
            torch.stack([e1, e2, 2 * e1]),  # repeated direction
            torch.stack([e1, e2, e1 + e2]),  # dependent up to rounding
            torch.stack([e1, e2, torch.tensor([0.0, 0.0, 1e4], dtype=dtype)]),  # fine
        ]
    ).to(device)
    matrices.requires_grad_(True)

    inverse, _, is_singular = normalized_inverse(matrices)

    assert is_singular.tolist() == [True, True, True, False]
    inverse.sum().backward()
    assert torch.isfinite(matrices.grad).all()


@pytest.mark.parametrize("n", [1, 2, 3])
def test_normalized_gram_inverse_matches_gram_inverse(n, device):
    """inv(G) = D^-1 inv(G_hat) D^-1 for rows of very different lengths."""
    generator = torch.Generator().manual_seed(n)
    rows = torch.randn(64, n, n + 1, generator=generator, dtype=torch.float64)
    lengths = 10.0 ** (
        8 * torch.rand(64, n, generator=generator, dtype=torch.float64) - 4
    )
    rows = (lengths[..., None] * rows / rows.norm(dim=-1, keepdim=True)).to(device)

    inverse, row_lengths, is_singular = normalized_gram_inverse(rows)

    assert not is_singular.any()
    gram = rows @ rows.transpose(-1, -2)
    torch.testing.assert_close(
        inverse / (row_lengths[..., :, None] * row_lengths[..., None, :]),
        torch.linalg.inv(gram),
    )


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


@pytest.mark.parametrize(
    "n_manifold_dims, n_spatial_dims, n_thin",
    [(2, 2, 1), (2, 3, 1), (3, 3, 1), (3, 3, 2)],
)
def test_float32_cotan_weights_of_thin_cells(
    n_manifold_dims, n_spatial_dims, n_thin, device
):
    """In float32, thin cells above the degeneracy cutoff are no less accurate.

    Each implementation is compared with a float64 truth. Full-dimensional cells
    invert E itself rather than G = E E^T, so they must come out much more
    accurate than the batched inverse of G. The closed-form inverse of G is far
    less accurate than the batched one for needles (two thin directions).
    """
    mesh = _thin_simplices(
        2000, n_manifold_dims, n_spatial_dims, n_thin, thickness=1e-2, device=device
    )
    truth = _reference_cotan_weights(
        Mesh(points=mesh.points.double(), cells=mesh.cells)
    )

    def error(weights: torch.Tensor) -> float:
        """90th percentile of the relative error, floored at a typical weight."""
        floor = truth.abs().median()
        relative = (weights.double() - truth).abs() / truth.abs().clamp_min(floor)
        # A weight that failed (NaN) counts as an infinitely large error
        return relative.nan_to_num(nan=float("inf")).quantile(0.9).item()

    fast = error(compute_cotan_weights_fem(mesh)[0])
    batched = error(_reference_cotan_weights(mesh))
    assert fast <= 2 * batched
    if n_manifold_dims == n_spatial_dims:
        assert fast <= batched / 10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [1e4, 1e6])
def test_cotan_weights_of_stretched_cells(dtype, length, device):
    """A long, thin but well-shaped tetrahedron is not treated as degenerate,
    in any vertex order: listed from its tip, its edges are nearly parallel."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, length]],
        dtype=torch.float64,
    )

    # Float64 FEM weights without any degeneracy cutoff; for one cell the unique
    # edges are in itertools.combinations order
    E = points[1:] - points[:1]
    H = torch.cat(
        [-torch.ones(1, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)]
    )
    C = H @ torch.linalg.inv(E @ E.T) @ H.T
    rows, cols = torch.triu_indices(4, 4, offset=1)
    expected = -(length / 6) * C[rows, cols]
    for order in itertools.permutations(range(4)):
        mesh = Mesh(
            points=points.to(device=device, dtype=dtype),
            cells=torch.tensor([order], device=device),
        )
        weights, edges = compute_cotan_weights_fem(mesh)
        assert edges.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        torch.testing.assert_close(
            weights.double().cpu(),
            expected,
            rtol=1e-5 if dtype == torch.float32 else 1e-9,
            atol=1e-6 * expected.abs().max(),
        )


def test_float16_cotan_weights(device):
    """16-bit meshes run in float32, so the degeneracy cutoff still works."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    cells = torch.tensor([[0, 1, 2], [0, 1, 3]])  # cell 1 is collinear
    half = compute_cotan_weights_fem(
        Mesh(points=points.half().to(device), cells=cells.to(device))
    )[0]
    full = compute_cotan_weights_fem(
        Mesh(points=points.to(device), cells=cells.to(device))
    )[0]

    assert half.dtype == torch.float16
    torch.testing.assert_close(half.float(), full, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_manifold_dims, n_spatial_dims", [(2, 2), (3, 3)])
def test_vertex_angles_are_sync_free(n_manifold_dims, n_spatial_dims):
    mesh = _random_simplices(1000, n_manifold_dims, n_spatial_dims, "cuda")
    compute_vertex_angles(mesh)  # warm up lazy CUDA initialization
    torch.cuda.synchronize()

    with _cuda_sync_budget(0):
        compute_vertex_angles(mesh)
    torch.cuda.synchronize()
