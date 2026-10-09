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

"""LSQ gradients on mesh stencils: closed-form small_lstsq against torch.linalg.lstsq."""

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
from physicsnemo.mesh.primitives.curves import circle_2d, helix_3d
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
        patch.setattr(_torch_impl, "batched_lstsq", _lstsq_reference)
        reference = gradient_lsq(mesh, values)
    _assert_close_to_reference(output, reference, dtype)
