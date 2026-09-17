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

"""Tests for mesh-aware deformation-energy wrappers."""

import importlib

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.deformation import (
    closed_surface_volume_energy,
    simplex_inversion_energy,
    simplex_measure_energy,
    simplex_strain_energy,
    surface_bending_energy,
    total_measure_energy,
)
from physicsnemo.mesh.utilities._topology import extract_triangle_hinges
from test.conftest import requires_module


def _triangle_mesh_2d() -> Mesh:
    return Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )


def _tetrahedron_surface(*, inward: bool = False) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    if inward:
        cells = cells[:, (0, 2, 1)]
    return Mesh(points=points, cells=cells)


def test_deformation_namespace_exports_mesh_energy_wrappers():
    module = importlib.import_module("physicsnemo.mesh.deformation")
    mesh_module = importlib.import_module("physicsnemo.mesh")

    assert module.simplex_strain_energy is simplex_strain_energy
    assert module.simplex_measure_energy is simplex_measure_energy
    assert module.total_measure_energy is total_measure_energy
    assert module.simplex_inversion_energy is simplex_inversion_energy
    assert module.surface_bending_energy is surface_bending_energy
    assert module.closed_surface_volume_energy is closed_surface_volume_energy
    assert not hasattr(mesh_module, "simplex_strain_energy")
    assert not hasattr(mesh_module, "surface_bending_energy")


def test_simplex_wrapper_forwards_reference_geometry_and_options(monkeypatch):
    module = importlib.import_module("physicsnemo.mesh.deformation._energies")
    mesh = _triangle_mesh_2d()
    points = mesh.points + 0.1
    sentinel = torch.tensor(3.0)
    received = {}

    def fake_energy(current, reference, cells, **kwargs):
        received.update(
            current=current,
            reference=reference,
            cells=cells,
            kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(module, "_simplex_strain_energy", fake_energy)

    result = simplex_strain_energy(
        mesh,
        points,
        lame_lambda=2.0,
        shear_modulus=3.0,
        reduction="mean",
        implementation="torch",
    )

    assert result is sentinel
    assert received["current"] is points
    assert received["reference"] is mesh.points
    assert received["cells"] is mesh.cells
    assert received["kwargs"] == {
        "lame_lambda": 2.0,
        "shear_modulus": 3.0,
        "reduction": "mean",
        "implementation": "torch",
    }


def test_simplex_mesh_wrappers_are_zero_at_reference_and_differentiable():
    mesh = _triangle_mesh_2d()
    points = mesh.points.detach().clone().requires_grad_()

    energies = (
        simplex_strain_energy(mesh, points, implementation="torch"),
        simplex_measure_energy(mesh, points, implementation="torch"),
        total_measure_energy(mesh, points, implementation="torch"),
        simplex_inversion_energy(mesh, points, implementation="torch"),
    )
    for energy in energies:
        torch.testing.assert_close(energy, torch.zeros_like(energy))

    sum(energies).backward()
    assert points.grad is not None
    torch.testing.assert_close(points.grad, torch.zeros_like(points))


@pytest.mark.parametrize(
    "function",
    [simplex_strain_energy, simplex_measure_energy, total_measure_energy],
)
def test_simplex_wrappers_reject_batched_current_points(function):
    mesh = _triangle_mesh_2d()

    with pytest.raises(ValueError, match="unbatched points"):
        function(mesh, mesh.points.unsqueeze(0), implementation="torch")


def test_mesh_wrappers_require_exact_shape_dtype_and_device():
    mesh = _triangle_mesh_2d()

    with pytest.raises(TypeError, match="reference_mesh must be a Mesh"):
        simplex_measure_energy(mesh.points, mesh.points, implementation="torch")
    with pytest.raises(ValueError, match="same shape"):
        simplex_measure_energy(mesh, mesh.points[:-1], implementation="torch")
    with pytest.raises(TypeError, match="same dtype"):
        simplex_measure_energy(mesh, mesh.points.float(), implementation="torch")


@pytest.mark.parametrize(
    "connectivity_dtype",
    [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ],
)
def test_mesh_wrapper_caches_normalized_integer_connectivity(
    monkeypatch, connectivity_dtype
):
    module = importlib.import_module("physicsnemo.mesh.deformation._energies")
    mesh = _triangle_mesh_2d()
    mesh = Mesh(points=mesh.points, cells=mesh.cells.to(connectivity_dtype))
    received = []

    def fake_energy(current, reference, cells, **kwargs):
        received.append(cells)
        return current.new_zeros(())

    monkeypatch.setattr(module, "_simplex_measure_energy", fake_energy)

    simplex_measure_energy(mesh, mesh.points, implementation="torch")
    simplex_measure_energy(mesh, mesh.points, implementation="torch")

    assert received[0].dtype == torch.int64
    assert received[0].data_ptr() == received[1].data_ptr()


def test_inversion_wrapper_rejects_embedded_surface():
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )

    with pytest.raises(ValueError, match="full-dimensional simplices"):
        simplex_inversion_energy(mesh, mesh.points, implementation="torch")


def test_bending_wrapper_extracts_and_reuses_cached_hinges(monkeypatch):
    module = importlib.import_module("physicsnemo.mesh.deformation._energies")
    mesh_2d = _triangle_mesh_2d()
    mesh = Mesh(
        points=torch.nn.functional.pad(mesh_2d.points, (0, 1)),
        cells=mesh_2d.cells,
    )
    captured_hinges = []

    def fake_energy(current, reference, hinges, **kwargs):
        captured_hinges.append(hinges)
        return current.new_zeros(())

    monkeypatch.setattr(module, "_surface_bending_energy", fake_energy)

    surface_bending_energy(mesh, mesh.points, implementation="torch")
    surface_bending_energy(mesh, mesh.points, implementation="torch")

    torch.testing.assert_close(captured_hinges[0], torch.tensor([[0, 2, 1, 3]]))
    assert captured_hinges[0].data_ptr() == captured_hinges[1].data_ptr()


def test_bending_mesh_wrapper_is_zero_at_reference():
    mesh_2d = _triangle_mesh_2d()
    mesh = Mesh(
        points=torch.nn.functional.pad(mesh_2d.points, (0, 1)),
        cells=mesh_2d.cells,
    )

    energy = surface_bending_energy(mesh, mesh.points, implementation="torch")

    torch.testing.assert_close(energy, torch.zeros_like(energy))


def test_bending_mesh_wrapper_is_independent_of_input_face_winding():
    source = _triangle_mesh_2d()
    reference_points = torch.nn.functional.pad(source.points, (0, 1))
    points = reference_points.clone()
    points[3, 2] = 0.4
    cell_variants = (
        source.cells,
        torch.stack((source.cells[0, (0, 2, 1)], source.cells[1])),
        torch.stack((source.cells[0], source.cells[1, (0, 2, 1)])),
        source.cells[:, (0, 2, 1)],
    )

    energies = [
        surface_bending_energy(
            Mesh(points=reference_points, cells=cells),
            points,
            implementation="torch",
        )
        for cells in cell_variants
    ]
    assert energies[0] > 0
    for energy in energies[1:]:
        torch.testing.assert_close(energy, energies[0])


@pytest.mark.parametrize("inward", [False, True])
def test_closed_volume_wrapper_accepts_consistent_global_orientation(
    monkeypatch, inward
):
    module = importlib.import_module("physicsnemo.mesh.deformation._energies")
    mesh = _tetrahedron_surface(inward=inward)
    captured = {}

    def fake_energy(current, reference, triangles, **kwargs):
        captured["triangles"] = triangles
        captured["kwargs"] = kwargs
        return current.new_zeros(())

    monkeypatch.setattr(module, "_closed_surface_volume_energy", fake_energy)

    result = closed_surface_volume_energy(
        mesh,
        mesh.points,
        target_ratio=0.9,
        implementation="torch",
    )

    torch.testing.assert_close(result, torch.tensor(0.0, dtype=mesh.points.dtype))
    assert captured["triangles"] is mesh.cells
    assert captured["kwargs"]["target_ratio"] == 0.9


def test_closed_volume_wrapper_rejects_open_surface_before_dispatch(monkeypatch):
    module = importlib.import_module("physicsnemo.mesh.deformation._energies")
    mesh = _tetrahedron_surface()
    mesh = Mesh(points=mesh.points, cells=mesh.cells[:-1])

    def must_not_run(*args, **kwargs):
        raise AssertionError("tensor energy must not run for invalid topology")

    monkeypatch.setattr(module, "_closed_surface_volume_energy", must_not_run)

    with pytest.raises(ValueError, match="edge-closed surface"):
        closed_surface_volume_energy(mesh, mesh.points, implementation="torch")


def test_closed_volume_mesh_wrapper_is_zero_at_reference():
    mesh = _tetrahedron_surface(inward=True)

    energy = closed_surface_volume_energy(mesh, mesh.points, implementation="torch")

    torch.testing.assert_close(energy, torch.zeros_like(energy))


@pytest.mark.cuda
@requires_module("warp")
def test_cuda_surface_energy_wrappers_preprocess_cache_and_differentiate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mesh-level Warp coverage")
    mesh = _tetrahedron_surface().to("cuda:0")
    points = mesh.points.detach().clone()
    points[3, 2] += 0.12
    points.requires_grad_()

    energy = surface_bending_energy(
        mesh, points, implementation="warp"
    ) + closed_surface_volume_energy(mesh, points, implementation="warp")
    energy.backward()

    assert torch.isfinite(energy)
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    first_hinges = extract_triangle_hinges(mesh)
    second_hinges = extract_triangle_hinges(mesh)
    assert first_hinges.is_cuda
    assert first_hinges.data_ptr() == second_hinges.data_ptr()
    assert mesh._cache.get(("topology", "closed_oriented_triangle_surface")) is not None
