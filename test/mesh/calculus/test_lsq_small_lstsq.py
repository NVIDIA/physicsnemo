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

"""LSQ gradients on mesh stencils: closed-form small_lstsq against torch.linalg.lstsq,
and the systems assembled at once against those solved by neighbor count."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import _lsq_intrinsic
from physicsnemo.mesh.calculus._lsq_intrinsic import (
    compute_point_gradient_lsq_intrinsic,
)
from physicsnemo.mesh.calculus._lsq_reconstruction import (
    compute_cell_gradient_lsq,
    compute_point_gradient_lsq,
)
from physicsnemo.mesh.generate import mesh_implicit_domain, sdf_sphere
from physicsnemo.mesh.primitives.curves import circle_2d, helix_3d
from physicsnemo.mesh.primitives.planar import unit_square
from physicsnemo.mesh.primitives.surfaces import plane, sphere_icosahedral, torus
from physicsnemo.mesh.primitives.volumes import cube_volume
from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient import _torch_impl


def _lstsq_reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Former formulation: CPU ``lstsq``, which gives minimum-norm solutions.

    The driver is ``gelsd`` rather than the default ``gelsy``, which
    intermittently returns zero for some exactly rank-deficient systems.
    """
    solution = torch.linalg.lstsq(A.cpu(), B.cpu(), rcond=None, driver="gelsd").solution
    return solution.to(A.device)


def _scaled_mesh(mesh: Mesh, length_scale: float, dtype: torch.dtype) -> Mesh:
    return Mesh(points=(mesh.points * length_scale).to(dtype), cells=mesh.cells)


def _fields(points: torch.Tensor, length_scale: float) -> torch.Tensor:
    """A scalar and a vector field, of order one at every length scale."""
    p = points / length_scale
    scalar = torch.sin(3 * p[:, 0]) + p[:, 1] ** 2 + 0.5 * p[:, -1]
    return torch.stack([scalar, p.sum(-1), torch.cos(2 * p).prod(-1)], dim=-1)


def _assert_close_to_reference(
    output: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype
) -> None:
    assert torch.isfinite(output).all()
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        output, reference, rtol=tolerance, atol=tolerance * reference.abs().max()
    )


_INTRINSIC_MESHES = {
    "sphere": lambda device: sphere_icosahedral.load(subdivisions=3, device=device),
    "plane": lambda device: plane.load(subdivisions=8, device=device),
    "helix": lambda device: helix_3d.load(n_points=64, device=device),
    "circle": lambda device: circle_2d.load(n_points=48, device=device),
}


@pytest.mark.parametrize("mesh_name", list(_INTRINSIC_MESHES))
@pytest.mark.parametrize("length_scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_point_gradient_lsq_intrinsic_matches_lstsq(
    device: str, mesh_name: str, length_scale: float, dtype: torch.dtype, monkeypatch
):
    """Tangent-space LSQ gradients of surfaces and curves match lstsq."""
    mesh = _scaled_mesh(_INTRINSIC_MESHES[mesh_name](device), length_scale, dtype)
    values = _fields(mesh.points, length_scale)

    for point_values in (values[:, 0], values):
        output = compute_point_gradient_lsq_intrinsic(mesh, point_values)
        with monkeypatch.context() as patch:
            patch.setattr(_lsq_intrinsic, "_uses_normal_equations", lambda *_: False)
            patch.setattr(_lsq_intrinsic, "batched_lstsq", _lstsq_reference)
            reference = compute_point_gradient_lsq_intrinsic(mesh, point_values)
        _assert_close_to_reference(output, reference, dtype)


_EXTRINSIC_CASES = {
    # Nearly flat 3D stencils.
    "sphere_points": lambda device: (
        sphere_icosahedral.load(subdivisions=3, device=device),
        compute_point_gradient_lsq,
    ),
    "torus_points": lambda device: (
        torus.load(n_major=32, n_minor=16, device=device),
        compute_point_gradient_lsq,
    ),
    # Exactly flat 3D stencils: rank 2.
    "plane_points": lambda device: (
        plane.load(subdivisions=8, device=device),
        compute_point_gradient_lsq,
    ),
    "tet_cube_points": lambda device: (
        cube_volume.load(subdivisions=3, device=device),
        compute_point_gradient_lsq,
    ),
    # Boundary cells have one to three face neighbors: underdetermined.
    "tet_cube_cells": lambda device: (
        cube_volume.load(subdivisions=3, device=device),
        compute_cell_gradient_lsq,
    ),
}


@pytest.mark.parametrize("case", list(_EXTRINSIC_CASES))
@pytest.mark.parametrize("length_scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_lsq_matches_lstsq(
    device: str, case: str, length_scale: float, dtype: torch.dtype, monkeypatch
):
    """Ambient-space LSQ gradients at points and cells match lstsq."""
    mesh, gradient_lsq = _EXTRINSIC_CASES[case](device)
    mesh = _scaled_mesh(mesh, length_scale, dtype)
    locations = mesh.points if case.endswith("_points") else mesh.cell_centroids
    values = _fields(locations, length_scale)

    output = gradient_lsq(mesh, values)
    with monkeypatch.context() as patch:
        patch.setattr(_torch_impl, "_uses_normal_equations", lambda *_: False)
        patch.setattr(_torch_impl, "batched_lstsq", _lstsq_reference)
        reference = gradient_lsq(mesh, values)
    _assert_close_to_reference(output, reference, dtype)


def _unstructured_ball(device: str) -> Mesh:
    """An unstructured tetrahedral ball, with 4 to 16 neighbors per point."""
    mesh = mesh_implicit_domain(
        sdf_sphere([0.0] * 3, 1.0), ((-1.1,) * 3, (1.1,) * 3), 0.25, dtype=torch.float32
    )
    return mesh.to(device)  # generated on the CPU, where it is reproducible


def _with_isolated_point(mesh: Mesh) -> Mesh:
    """The mesh and a point in no cell, which has no neighbors."""
    points = torch.cat([mesh.points, mesh.points[:1] + 0.1])
    return Mesh(points=points, cells=mesh.cells)


_PARITY_MESHES = {
    "kuhn_cube": lambda device: cube_volume.load(subdivisions=4, device=device),
    "unstructured_ball": _unstructured_ball,
    # Rank-deficient 3D stencils, and the interior points with no neighbors
    "ball_surface": lambda device: _unstructured_ball(device).get_boundary_mesh(),
    "square": lambda device: _with_isolated_point(
        unit_square.load(subdivisions=6, device=device)
    ),
    "helix": lambda device: helix_3d.load(n_points=64, device=device),
    "circle": lambda device: circle_2d.load(n_points=48, device=device),
}


def _parity_fields(locations: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    """A scalar, a vector and a rank-2 tensor field."""
    g = torch.Generator().manual_seed(0)
    p = locations.detach().float()
    scalar = torch.sin(3 * p[:, 0]) + p[:, -1] ** 2
    tensor = torch.randn(len(p), 2, 3, generator=g).to(p.device)
    return [field.to(dtype) for field in (scalar, torch.cos(2 * p), tensor)]


def _assert_matches_grouped(assembled: torch.Tensor, grouped: torch.Tensor) -> None:
    """Float32 gradients are mostly identical: the moments are summed in
    float64 in another order, which changes a few float32 roundings, by less
    than float32 eps times the largest gradient. 16-bit systems differ in the
    last bit of some 16-bit weights, whose sums of three squares the CPU rounds
    differently for tensors of different shapes."""
    if assembled.dtype == torch.float32:
        scale = grouped.abs().max().item()
        torch.testing.assert_close(assembled, grouped, rtol=0, atol=1e-7 * scale)
    else:
        scale = grouped.float().abs().max()
        torch.testing.assert_close(
            assembled, grouped, atol=1e-2 * scale.item(), rtol=1e-2
        )


@pytest.mark.parametrize("mesh_name", list(_PARITY_MESHES))
@pytest.mark.parametrize("location", ["points", "cells"])
@pytest.mark.parametrize("min_neighbors", [0, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_assembled_lsq_gradients_match_grouped(
    device: str, mesh_name: str, location: str, min_neighbors: int, dtype
):
    """Solving every LSQ system at once gives the gradients of solving them in
    groups of one neighbor count, up to rounding; too few neighbors give zero."""
    mesh = _PARITY_MESHES[mesh_name](device)
    if location == "points":
        locations, adjacency = mesh.points, mesh.get_point_to_points_adjacency()
    else:
        locations = mesh.cell_centroids
        adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
    fields = _parity_fields(locations, dtype)

    gradients = [
        _torch_impl._mesh_lsq_gradients_torch(
            locations,
            fields,
            adjacency.offsets,
            adjacency.indices,
            min_neighbors=min_neighbors,
            grouped=grouped,
        )
        for grouped in (False, True)
    ]

    unsolved = adjacency.counts < max(min_neighbors, 1)
    for assembled, grouped in zip(*gradients):
        _assert_matches_grouped(assembled, grouped)
        assert (assembled[unsolved] == 0).all()


@pytest.mark.parametrize("mesh_name", ["ball_surface", "helix", "circle"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_assembled_intrinsic_lsq_gradients_match_grouped(
    device: str, mesh_name: str, dtype: torch.dtype
):
    """The same for the tangent-space systems of surfaces and curves."""
    if mesh_name == "helix" and dtype == torch.bfloat16:
        pytest.skip("the tangents of curves in 3D need eigh, with no 16-bit kernel")
    mesh = _PARITY_MESHES[mesh_name](device)
    mesh = Mesh(points=mesh.points.to(dtype), cells=mesh.cells)
    fields = _parity_fields(mesh.points, dtype)

    gradients = [
        _lsq_intrinsic._compute_point_gradients_lsq_intrinsic(
            mesh, fields, grouped=grouped
        )
        for grouped in (False, True)
    ]

    unsolved = mesh.get_point_to_points_adjacency().counts < 2
    for assembled, grouped in zip(*gradients):
        _assert_matches_grouped(assembled, grouped)
        assert (assembled[unsolved] == 0).all()


def _backward_of_both_paths(
    mesh: Mesh, gradient_type: str, seed: int = 0
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """First and second derivatives, with respect to the points and the
    values, of the assembled and the grouped LSQ gradients of one vector field.

    Both paths differentiate one ``Mesh``, so they share its point normals and
    tangent bases. Recomputed, those differ by rounding on CUDA, whose sums
    use atomics. Where the face normals around a point nearly cancel, as at a
    fold of a coarse boundary surface, that rounding turns the tangent plane
    there, which changes the intrinsic gradient and its derivatives by far
    more than the solves do.
    """
    generator = torch.Generator().manual_seed(seed)
    values = _parity_fields(mesh.points, torch.float32)[1]
    weights = torch.randn(
        mesh.n_points, mesh.n_spatial_dims, values.shape[1], generator=generator
    ).to(mesh.points.device)
    points = mesh.points.clone().requires_grad_()
    field = values.clone().requires_grad_()
    shared = Mesh(points=points, cells=mesh.cells)

    def derivatives(grouped: bool) -> list[torch.Tensor]:
        if gradient_type == "extrinsic":
            adjacency = shared.get_point_to_points_adjacency()
            gradient = _torch_impl._mesh_lsq_gradients_torch(
                points, [field], adjacency.offsets, adjacency.indices, grouped=grouped
            )[0]
        else:
            gradient = _lsq_intrinsic._compute_point_gradients_lsq_intrinsic(
                shared, [field], grouped=grouped
            )[0]
        first = torch.autograd.grad(
            (weights * gradient).sum(), (points, field), create_graph=True
        )
        # The graph of the shared normals serves the other path too
        second = torch.autograd.grad(
            sum(d.square().sum() for d in first), (points, field), retain_graph=True
        )
        return [*first, *second]

    return list(zip(derivatives(False), derivatives(True)))


@pytest.mark.parametrize("mesh_name", ["unstructured_ball", "ball_surface"])
@pytest.mark.parametrize("gradient_type", ["extrinsic", "intrinsic"])
def test_assembled_lsq_gradients_backward_match_grouped(
    device: str, mesh_name: str, gradient_type: str
):
    """First and second derivatives with respect to the points and the values
    match those of the groups."""
    mesh = _PARITY_MESHES[mesh_name](device)
    for assembled, grouped in _backward_of_both_paths(mesh, gradient_type):
        assert torch.isfinite(assembled).all()
        torch.testing.assert_close(
            assembled, grouped, rtol=1e-4, atol=1e-4 * grouped.abs().max().item()
        )
