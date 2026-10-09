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

"""Tests the closed-form barycentric solver against the torch.linalg one it replaced."""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.sampling import find_all_containing_cells, sample_data_at_points
from physicsnemo.mesh.sampling.sample_data import _solve_barycentric_system

### (n_manifold_dims, n_spatial_dims): triangles and tetrahedra filling their
### space, edges in 2D and 3D, triangles in 3D, and cells of codimension 2.
CELL_TYPES = [(2, 2), (3, 3), (1, 2), (1, 3), (2, 3), (2, 4)]


def _solve_barycentric_system_reference(
    relative_vectors: torch.Tensor, query_relative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The torch.linalg formulation that ``_solve_barycentric_system`` replaced."""
    n_manifold_dims = relative_vectors.shape[-2]
    n_spatial_dims = relative_vectors.shape[-1]
    A = relative_vectors.transpose(-2, -1)
    b = query_relative.unsqueeze(-1)
    if n_spatial_dims == n_manifold_dims:
        try:
            weights_1_to_n = torch.linalg.solve(A, b).squeeze(-1)
        except torch.linalg.LinAlgError:
            weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)
        reconstruction_error = torch.zeros(
            weights_1_to_n.shape[:-1],
            dtype=query_relative.dtype,
            device=query_relative.device,
        )
    else:
        weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)
        reconstructed = torch.einsum(
            "...m,...ms->...s", weights_1_to_n, relative_vectors
        )
        reconstruction_error = torch.linalg.vector_norm(
            query_relative - reconstructed, dim=-1
        )
    w_0 = 1.0 - weights_1_to_n.sum(dim=-1, keepdim=True)
    return torch.cat([w_0, weights_1_to_n], dim=-1), reconstruction_error


def _random_systems(
    n_cells: int,
    n_manifold_dims: int,
    n_spatial_dims: int,
    thinness: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 edge vectors and queries near the cells, some off the affine hull.

    For ``thinness < 1`` the last edge is nearly parallel to the first, so the
    cells are slivers whose condition number grows like ``1 / thinness``.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (n_cells, n_manifold_dims, n_spatial_dims)
    relative_vectors = torch.randn(shape, generator=generator, dtype=torch.float64)
    if n_manifold_dims > 1:
        relative_vectors[:, -1] = (
            relative_vectors[:, 0] + thinness * relative_vectors[:, -1]
        )
    weights = (
        1.4
        * torch.rand(n_cells, n_manifold_dims, generator=generator, dtype=torch.float64)
        - 0.2
    )
    query_relative = (weights.unsqueeze(-1) * relative_vectors).sum(dim=-2)
    if n_manifold_dims < n_spatial_dims:
        query_relative = query_relative + 0.1 * torch.randn(
            n_cells, n_spatial_dims, generator=generator, dtype=torch.float64
        )
    return relative_vectors, query_relative


@pytest.mark.parametrize("n_manifold_dims, n_spatial_dims", CELL_TYPES)
@pytest.mark.parametrize("thinness", [1.0, 1e-3])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_linalg_solver(
    n_manifold_dims, n_spatial_dims, thinness, scale, dtype, device
):
    """Closed form agrees with solve/lstsq to within the conditioning of each cell."""
    relative_vectors, query_relative = _random_systems(
        500, n_manifold_dims, n_spatial_dims, thinness, seed=n_spatial_dims
    )
    relative_vectors = (scale * relative_vectors).to(dtype=dtype, device=device)
    query_relative = (scale * query_relative).to(dtype=dtype, device=device)

    bary, reconstruction_error = _solve_barycentric_system(
        relative_vectors, query_relative
    )
    assert bary.dtype == dtype and reconstruction_error.dtype == dtype

    ### The float64 reference solves the same (rounded) inputs, so the difference
    ### is the closed form's own error, bounded by the least-squares perturbation
    ### bound eps * (cond * |w| + cond^2 * |residual|) in units of the cell size.
    relative_vectors_64 = relative_vectors.double()
    query_relative_64 = query_relative.double()
    bary_ref, reconstruction_error_ref = _solve_barycentric_system_reference(
        relative_vectors_64, query_relative_64
    )
    length_scale = relative_vectors_64.norm(dim=-1).mean(dim=-1)
    cond = torch.linalg.cond(relative_vectors_64)
    error_bound = (
        20.0
        * torch.finfo(dtype).eps
        * (
            cond
            * (
                1.0
                + bary_ref.abs().amax(dim=-1)
                + query_relative_64.norm(dim=-1) / length_scale
            )
            + cond.square() * reconstruction_error_ref / length_scale
        )
    )
    assert ((bary.double() - bary_ref).abs().amax(dim=-1) <= error_bound).all()
    assert (
        (reconstruction_error.double() - reconstruction_error_ref).abs()
        <= error_bound * length_scale
    ).all()


@pytest.mark.parametrize("n_manifold_dims, n_spatial_dims", CELL_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_degenerate_cells_get_nan(n_manifold_dims, n_spatial_dims, dtype, device):
    """Exactly degenerate cells get NaN coordinates; the others are unaffected."""
    relative_vectors, query_relative = _random_systems(
        4, n_manifold_dims, n_spatial_dims, thinness=1.0, seed=0
    )
    relative_vectors = relative_vectors.to(dtype=dtype, device=device)
    query_relative = query_relative.to(dtype=dtype, device=device)
    relative_vectors[0] = 0.0  # all vertices coincide
    if n_manifold_dims > 1:
        relative_vectors[1, -1] = 2.0 * relative_vectors[1, 0]  # collinear
    is_degenerate = torch.tensor(
        [True, n_manifold_dims > 1, False, False], device=device
    )

    bary, reconstruction_error = _solve_barycentric_system(
        relative_vectors, query_relative
    )

    assert bary[is_degenerate].isnan().all()
    if n_manifold_dims < n_spatial_dims:
        assert reconstruction_error[is_degenerate].isnan().all()
    else:
        assert (reconstruction_error == 0).all()
    bary_ref, reconstruction_error_ref = _solve_barycentric_system_reference(
        relative_vectors[2:], query_relative[2:]
    )
    torch.testing.assert_close(bary[2:], bary_ref)
    torch.testing.assert_close(reconstruction_error[2:], reconstruction_error_ref)


@pytest.mark.parametrize("n_manifold_dims, n_spatial_dims", CELL_TYPES)
def test_gradients_match_linalg_solver(n_manifold_dims, n_spatial_dims, device):
    """Gradients agree with solve/lstsq, and degenerate cells give finite zeros."""
    relative_vectors, query_relative = _random_systems(
        8, n_manifold_dims, n_spatial_dims, thinness=1.0, seed=1
    )
    relative_vectors = relative_vectors.to(device)
    query_relative = query_relative.to(device)
    relative_vectors[0] = 0.0
    if n_manifold_dims > 1:
        relative_vectors[1, -1] = 2.0 * relative_vectors[1, 0]
    regular = slice(2, None)
    cotangent = torch.randn(
        8, n_manifold_dims + 1, generator=torch.Generator().manual_seed(2)
    ).to(device=device, dtype=torch.float64)

    def gradients(solver, relative_vectors, query_relative):
        relative_vectors = relative_vectors.clone().requires_grad_()
        query_relative = query_relative.clone().requires_grad_()
        bary, reconstruction_error = solver(relative_vectors, query_relative)
        # Both solvers see the regular rows last.
        loss = (bary * cotangent[-len(bary) :]).sum() + reconstruction_error.sum()
        return torch.autograd.grad(loss, (relative_vectors, query_relative))

    ### Downstream, containment tests drop the NaN rows of degenerate cells.
    def solve_regular_rows(relative_vectors, query_relative):
        bary, reconstruction_error = _solve_barycentric_system(
            relative_vectors, query_relative
        )
        return bary[regular], reconstruction_error[regular]

    grads = gradients(solve_regular_rows, relative_vectors, query_relative)
    grads_ref = gradients(
        _solve_barycentric_system_reference,
        relative_vectors[regular],
        query_relative[regular],
    )
    for grad, grad_ref in zip(grads, grads_ref):
        assert (grad[:2] == 0).all()
        torch.testing.assert_close(grad[regular], grad_ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_point_cells_match_linalg_solver(dtype, device):
    """Point cells (n_manifold_dims == 0) have weight 1 and the distance as error."""
    relative_vectors = torch.zeros(5, 0, 3, dtype=dtype, device=device)
    query_relative = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
    query_relative = query_relative.to(dtype=dtype, device=device)

    bary, reconstruction_error = _solve_barycentric_system(
        relative_vectors, query_relative
    )

    bary_ref, reconstruction_error_ref = _solve_barycentric_system_reference(
        relative_vectors, query_relative
    )
    torch.testing.assert_close(bary, bary_ref)
    torch.testing.assert_close(reconstruction_error, reconstruction_error_ref)


def _degenerate_cell_next_to_regular_cell(
    n_manifold_dims: int, n_spatial_dims: int, device: str
) -> Mesh:
    """Cell 0 is degenerate on the x axis; cell 1 is a regular neighbor."""
    if n_manifold_dims == 2:
        # Collinear triangle (0,0)-(1,0)-(2,0), and a triangle sharing its edge 0-1.
        points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
        cells = [[0, 1, 2], [0, 1, 3]]
    else:
        # Zero-length edge at (2,0), and the edge (0,0)-(1,0).
        points = [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
        cells = [[0, 1], [2, 3]]
    points = torch.tensor(points, device=device)
    if n_spatial_dims == 3:
        points = torch.nn.functional.pad(points, (0, 1))
    return Mesh(points=points, cells=torch.tensor(cells, device=device))


@pytest.mark.parametrize(
    "n_manifold_dims, n_spatial_dims", [(2, 2), (2, 3), (1, 2), (1, 3)]
)
def test_degenerate_cells_contain_no_points(n_manifold_dims, n_spatial_dims, device):
    """Degenerate cells contain nothing, on every device; their neighbors still do.

    Previously, on CPU, ``solve`` raised on the singular cell and the whole batch
    fell back to the minimum-norm ``lstsq`` solution, so zero-area triangles
    contained the points of their segment and zero-length edges their point.
    """
    mesh = _degenerate_cell_next_to_regular_cell(
        n_manifold_dims, n_spatial_dims, device
    )
    queries = torch.tensor([[0.5, 0.0], [1.5, 0.0], [2.0, 0.0]], device=device)
    if n_spatial_dims == 3:
        queries = torch.nn.functional.pad(queries, (0, 1))

    containing = find_all_containing_cells(mesh, queries)

    assert containing.offsets.tolist() == [0, 1, 1, 1]
    assert containing.indices.tolist() == [1]


@pytest.mark.parametrize(
    "n_manifold_dims, n_spatial_dims", [(2, 2), (2, 3), (1, 2), (1, 3)]
)
def test_sampling_gradients_finite_next_to_degenerate_cell(
    n_manifold_dims, n_spatial_dims, device
):
    """Interpolated point data differentiates to finite point gradients."""
    mesh = _degenerate_cell_next_to_regular_cell(
        n_manifold_dims, n_spatial_dims, device
    )
    points = mesh.points.clone().requires_grad_()
    mesh = Mesh(
        points=points,
        cells=mesh.cells,
        point_data={"u": torch.arange(4.0, device=device)},
    )
    queries = torch.tensor([[0.25, 0.0], [0.5, 0.0]], device=device)
    if n_spatial_dims == 3:
        queries = torch.nn.functional.pad(queries, (0, 1))

    sampled = sample_data_at_points(mesh, queries, data_source="points")
    sampled["u"].sum().backward()

    assert sampled["u"].isfinite().all()
    assert points.grad.isfinite().all()
