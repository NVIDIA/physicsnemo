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

"""Mesh and DomainMesh integration tests for lattice free-form deformation."""

import importlib
import inspect
from typing import Literal, get_type_hints

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.transformations.deform import free_form_deform

_EXPECTED_FFD_BASIS = Literal[
    "bernstein", "bspline", "linear", "cubic_hermite", "quintic_hermite"
]


def test_free_form_deform_namespace_is_canonical():
    """Only the deformation namespace exports ``free_form_deform``."""

    transformations = importlib.import_module("physicsnemo.mesh.transformations")
    deform_module = importlib.import_module("physicsnemo.mesh.transformations.deform")
    implementation_module = importlib.import_module(
        "physicsnemo.mesh.transformations.deform.ffd"
    )

    assert deform_module.free_form_deform is free_form_deform
    assert implementation_module.free_form_deform is free_form_deform
    assert not hasattr(implementation_module, "ffd")
    assert free_form_deform.__name__ == "free_form_deform"
    assert free_form_deform.__qualname__ == "free_form_deform"
    assert free_form_deform.__module__ == (
        "physicsnemo.mesh.transformations.deform.ffd"
    )
    assert "ffd" not in deform_module.__all__
    assert not hasattr(transformations, "free_form_deform")
    assert hasattr(Mesh, "free_form_deform")
    assert hasattr(DomainMesh, "free_form_deform")
    assert not hasattr(Mesh, "ffd")
    assert not hasattr(DomainMesh, "ffd")


def test_mesh_free_form_deform_signatures_and_annotations_are_introspectable():
    """Public deformation methods expose resolvable annotations."""

    assert (
        get_type_hints(free_form_deform, localns={"Mesh": Mesh})["basis"]
        == _EXPECTED_FFD_BASIS
    )
    for owner, deform_method in (
        (Mesh, Mesh.free_form_deform),
        (DomainMesh, DomainMesh.free_form_deform),
    ):
        signature = inspect.signature(deform_method)
        assert signature.parameters["basis"].default == "bernstein"
        assert signature.parameters["origin"].default is None
        assert signature.parameters["extent"].default is None
        assert get_type_hints(deform_method)["basis"] == _EXPECTED_FFD_BASIS
        assert get_type_hints(deform_method)["return"] is owner


def _triangle_mesh(*, requires_grad: bool = False) -> Mesh:
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=requires_grad
    )
    cells = torch.tensor([[0, 1, 2]])
    return Mesh(
        points=points,
        cells=cells,
        point_data={"temperature": torch.tensor([10.0, 20.0, 30.0])},
        cell_data={"material": torch.tensor([7])},
        global_data={"case_id": torch.tensor(12)},
    )


def test_mesh_ffd_default_box_spans_mesh_bounds():
    """With no explicit box, a constant lattice translates the whole mesh."""

    mesh = _triangle_mesh()
    translation = torch.tensor([0.25, -0.5])
    control_displacements = translation.expand(4, 4, 2).clone()
    source_points = mesh.points.clone()

    output = mesh.free_form_deform(control_displacements, implementation="torch")

    torch.testing.assert_close(output.points, source_points + translation)
    torch.testing.assert_close(mesh.points, source_points)
    assert output is not mesh
    assert torch.equal(output.cells, mesh.cells)
    assert torch.equal(output.point_data["temperature"], mesh.point_data["temperature"])
    assert torch.equal(output.cell_data["material"], mesh.cell_data["material"])
    assert torch.equal(output.global_data["case_id"], mesh.global_data["case_id"])


@pytest.mark.parametrize("basis", ["linear", "cubic_hermite", "quintic_hermite"])
def test_mesh_ffd_interpolating_bases_reproduce_control_nodes(basis):
    mesh = _triangle_mesh()
    control_displacements = 0.1 * torch.arange(8.0).reshape(2, 2, 2)

    output = mesh.free_form_deform(
        control_displacements,
        origin=[0.0, 0.0],
        extent=[1.0, 1.0],
        basis=basis,
        implementation="torch",
    )

    expected_displacements = torch.stack(
        (
            control_displacements[0, 0],
            control_displacements[1, 0],
            control_displacements[0, 1],
        )
    )
    torch.testing.assert_close(output.points, mesh.points + expected_displacements)


def test_mesh_ffd_explicit_box_leaves_outside_points_unchanged():
    mesh = _triangle_mesh()
    control_displacements = torch.full((4, 4, 2), 0.5)

    output = mesh.free_form_deform(
        control_displacements,
        origin=[-0.25, -0.25],
        extent=[0.5, 0.5],
        implementation="torch",
    )

    # Only the origin vertex lies inside the box.
    torch.testing.assert_close(output.points[0], torch.tensor([0.5, 0.5]))
    assert torch.equal(output.points[1:], mesh.points[1:])


def test_mesh_ffd_default_extent_rejects_degenerate_bounds():
    flat = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        cells=torch.tensor([[0, 1], [1, 2]]),
    )
    with pytest.raises(ValueError, match="must be finite and strictly positive"):
        flat.free_form_deform(torch.zeros(4, 4, 2), implementation="torch")


def test_mesh_ffd_defaults_support_empty_meshes_and_domains():
    mesh = Mesh(points=torch.empty((0, 2)))
    control_displacements = torch.zeros(4, 4, 2)

    mesh_output = mesh.free_form_deform(control_displacements, implementation="torch")
    domain_output = DomainMesh(interior=mesh).free_form_deform(
        control_displacements, implementation="torch"
    )

    assert mesh_output.points.shape == (0, 2)
    assert domain_output.interior.points.shape == (0, 2)


@pytest.mark.parametrize(
    ("origin", "error", "match"),
    [
        (0.0, TypeError, "origin must be a torch.Tensor or a sequence"),
        ([0.0, 0.0, 0.0], TypeError, "origin must contain exactly 2 real values"),
    ],
)
def test_mesh_ffd_validates_origin_before_deriving_extent(origin, error, match):
    mesh = _triangle_mesh()
    with pytest.raises(error, match=match):
        mesh.free_form_deform(
            torch.zeros(4, 4, 2),
            origin=origin,
            implementation="torch",
        )


def test_mesh_ffd_rejects_nonfinite_derived_extent():
    mesh = Mesh(points=torch.tensor([[-2.0e38, 0.0], [2.0e38, 1.0]]))
    with pytest.raises(ValueError, match="derived lattice extent must be finite"):
        mesh.free_form_deform(torch.zeros(4, 4, 2), implementation="torch")


def test_mesh_ffd_validates_point_dtype_before_bounds_reduction():
    mesh = Mesh(points=torch.zeros((2, 2), dtype=torch.complex64))
    with pytest.raises(TypeError, match="points must have dtype torch.float32"):
        mesh.free_form_deform(
            torch.zeros((4, 4, 2), dtype=torch.complex64), implementation="torch"
        )


def test_mesh_ffd_point_weights_key_and_cache_invalidation():
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]),
        cells=torch.tensor([[0, 1, 2]]),
        point_data={"weight": torch.tensor([1.0, 0.0, 1.0])},
    )
    original_area = mesh.cell_areas.clone()
    _ = mesh.cell_centroids
    topology = mesh.get_point_to_points_adjacency()

    translation = torch.tensor([0.0, 0.0, 0.25])
    control_displacements = translation.expand(4, 4, 4, 3).clone()
    output = mesh.free_form_deform(
        control_displacements, point_weights="weight", implementation="torch"
    )

    expected = mesh.points + translation * mesh.point_data["weight"].unsqueeze(-1)
    torch.testing.assert_close(output.points, expected)

    assert list(output._cache["cell"].keys()) == []
    assert list(output._cache["point"].keys()) == []
    cached_topology = output._cache.get(("topology", "point_to_points"))
    assert cached_topology is not None
    assert output.get_point_to_points_adjacency().to_list() == topology.to_list()

    # The source remains unchanged and keeps its already-computed geometry.
    torch.testing.assert_close(mesh.cell_areas, original_area)
    assert mesh._cache.get(("cell", "areas")) is not None
    assert mesh._cache.get(("cell", "centroids")) is not None


def test_mesh_ffd_preserves_autograd_through_returned_points():
    mesh = _triangle_mesh(requires_grad=True)
    control_displacements = 0.1 * torch.sin(torch.arange(4 * 4 * 2.0)).reshape(4, 4, 2)
    control_displacements.requires_grad_(True)
    output = mesh.free_form_deform(
        control_displacements,
        origin=[-0.5, -0.5],
        extent=[2.0, 2.0],
        implementation="torch",
    )
    loss = output.points.square().sum()
    loss.backward()

    assert mesh.points.grad is not None
    assert control_displacements.grad is not None
    assert torch.isfinite(mesh.points.grad).all()
    assert torch.isfinite(control_displacements.grad).all()


def test_mesh_ffd_missing_point_data_key_has_actionable_diagnostic():
    mesh = _triangle_mesh()
    with pytest.raises(KeyError, match="point_weights field 'missing'.*Available keys"):
        mesh.free_form_deform(
            torch.zeros(4, 4, 2),
            point_weights="missing",
            implementation="torch",
        )


def test_mesh_ffd_rejects_non_tensor_control_displacements():
    mesh = _triangle_mesh()
    with pytest.raises(TypeError, match="control_displacements"):
        mesh.free_form_deform("lattice", implementation="torch")


def _domain_with_coincident_points() -> DomainMesh:
    interior = _triangle_mesh()
    wall = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        cells=torch.tensor([[0, 1]]),
        point_data={"marker": torch.tensor([1.0, 2.0])},
        cell_data={"boundary_id": torch.tensor([4])},
    )
    interior.point_data["marker"] = torch.tensor([1.0, 2.0, 3.0])
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall},
        global_data={"reynolds": torch.tensor(1.0e5)},
    )


@pytest.mark.parametrize("point_weights", [None, "marker"])
def test_domain_ffd_shared_lattice_and_common_point_weight_key(point_weights):
    domain = _domain_with_coincident_points()
    control_displacements = 0.1 * torch.sin(torch.arange(4 * 4 * 2.0)).reshape(4, 4, 2)
    output = domain.free_form_deform(
        control_displacements,
        origin=[-0.5, -0.5],
        extent=[2.0, 2.0],
        point_weights=point_weights,
        implementation="torch",
    )

    # Components sharing a point and weight move identically.
    torch.testing.assert_close(
        output.interior.points[:2], output.boundaries["wall"].points
    )
    assert output is not domain
    assert torch.equal(output.interior.cells, domain.interior.cells)
    assert torch.equal(
        output.boundaries["wall"].cell_data["boundary_id"],
        domain.boundaries["wall"].cell_data["boundary_id"],
    )
    assert torch.equal(output.global_data["reynolds"], domain.global_data["reynolds"])
    torch.testing.assert_close(domain.interior.points, _triangle_mesh().points)


def test_domain_ffd_clones_domain_global_data():
    domain = _domain_with_coincident_points()
    output = domain.free_form_deform(torch.zeros(4, 4, 2), implementation="torch")

    assert output.global_data is not domain.global_data
    output.global_data["reynolds"] = torch.tensor(2.0e5)
    torch.testing.assert_close(domain.global_data["reynolds"], torch.tensor(1.0e5))


def test_domain_ffd_default_box_spans_combined_components():
    domain = _domain_with_coincident_points()
    outlet = Mesh(
        points=torch.tensor([[2.0, 0.0], [2.0, 1.0]]),
        cells=torch.tensor([[0, 1]]),
        point_data={"marker": torch.tensor([0.5, 0.75])},
    )
    domain.boundaries["outlet"] = outlet

    translation = torch.tensor([0.1, 0.2])
    output = domain.free_form_deform(
        translation.expand(4, 4, 2).clone(),
        implementation="torch",
    )

    # The derived box covers every component, so all points translate exactly.
    torch.testing.assert_close(
        output.boundaries["outlet"].points, outlet.points + translation
    )
    torch.testing.assert_close(
        output.interior.points, domain.interior.points + translation
    )


def test_domain_ffd_rejects_raw_point_weight_tensor():
    domain = _domain_with_coincident_points()
    with pytest.raises(TypeError, match="common point_data key/path"):
        domain.free_form_deform(
            torch.zeros(4, 4, 2),
            point_weights=torch.ones(domain.interior.n_points),
            implementation="torch",
        )


def test_domain_ffd_evaluates_combined_components_once(monkeypatch):
    domain = _domain_with_coincident_points()

    deform_module = importlib.import_module("physicsnemo.nn.functional.geometry.deform")
    original = deform_module.free_form_deform_points
    calls: list[torch.Tensor] = []

    def counted_free_form_deform_points(points, *args, **kwargs):
        calls.append(points)
        return original(points, *args, **kwargs)

    monkeypatch.setattr(
        deform_module,
        "free_form_deform_points",
        counted_free_form_deform_points,
    )
    output = domain.free_form_deform(
        torch.zeros(4, 4, 2),
        point_weights="marker",
        implementation="torch",
    )

    assert len(calls) == 1
    assert calls[0].shape == (5, 2)
    assert output.interior.n_points == 3
    assert output.boundaries["wall"].n_points == 2


def test_domain_combined_ffd_preserves_component_autograd():
    domain = _domain_with_coincident_points()
    interior_points = domain.interior.points.requires_grad_()
    wall_points = domain.boundaries["wall"].points.requires_grad_()
    interior_point_weights = domain.interior.point_data["marker"].requires_grad_()
    wall_point_weights = domain.boundaries["wall"].point_data["marker"].requires_grad_()
    control_displacements = 0.1 * torch.sin(torch.arange(4 * 4 * 2.0)).reshape(4, 4, 2)
    control_displacements.requires_grad_(True)

    output = domain.free_form_deform(
        control_displacements,
        origin=[-0.5, -0.5],
        extent=[2.0, 2.0],
        point_weights="marker",
        implementation="torch",
    )
    loss = output.interior.points.square().sum()
    loss = loss + output.boundaries["wall"].points.square().sum()
    gradients = torch.autograd.grad(
        loss,
        (
            interior_points,
            wall_points,
            interior_point_weights,
            wall_point_weights,
            control_displacements,
        ),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
