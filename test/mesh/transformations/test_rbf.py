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

"""Mesh and DomainMesh tests for thin-plate-spline radial-basis deformation."""

import importlib
import inspect

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.transformations.deform import radial_basis_function_deform


def _triangle_mesh(*, requires_grad: bool = False) -> Mesh:
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2]]),
        point_data={"marker": torch.tensor([1.0, 0.5, 0.75], dtype=torch.float64)},
        cell_data={"material": torch.tensor([7])},
        global_data={"case_id": torch.tensor(12)},
    )


def _controls_and_displacements() -> tuple[torch.Tensor, torch.Tensor]:
    controls = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    displacements = torch.tensor(
        [[0.2, -0.1], [0.0, 0.35], [-0.25, 0.15]], dtype=torch.float64
    )
    return controls, displacements


def _domain_with_coincident_points() -> DomainMesh:
    interior = _triangle_mesh()
    wall = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        cells=torch.tensor([[0, 1]]),
        point_data={"marker": torch.tensor([1.0, 0.5], dtype=torch.float64)},
        cell_data={"boundary_id": torch.tensor([4])},
    )
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall},
        global_data={"reynolds": torch.tensor(1.0e5)},
    )


def test_radial_basis_function_deform_namespace_is_canonical():
    transformations = importlib.import_module("physicsnemo.mesh.transformations")
    deform_module = importlib.import_module("physicsnemo.mesh.transformations.deform")

    assert transformations.deform is deform_module
    assert deform_module.radial_basis_function_deform is radial_basis_function_deform


def test_mesh_and_domain_radial_basis_function_deform_signatures_are_introspectable():
    for method in (
        Mesh.radial_basis_function_deform,
        DomainMesh.radial_basis_function_deform,
    ):
        signature = inspect.signature(method)
        assert signature.parameters["smoothing"].annotation is float
        assert signature.parameters["polynomial"].annotation is bool
        assert signature.parameters["kernel"].default == "thin_plate_spline"
        assert signature.parameters["polynomial"].default is True


def test_mesh_radial_basis_function_deform_method_and_function_interpolate_controls_exactly():
    mesh = _triangle_mesh()
    controls, displacements = _controls_and_displacements()
    expected = mesh.points + displacements

    method_output = mesh.radial_basis_function_deform(
        controls,
        displacements,
        polynomial=True,
        smoothing=0.0,
        implementation="torch",
    )
    function_output = radial_basis_function_deform(
        mesh,
        controls,
        displacements,
        polynomial=True,
        smoothing=0.0,
        implementation="torch",
    )

    torch.testing.assert_close(method_output.points, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(function_output.points, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mesh.points, controls)


def test_mesh_radial_basis_function_deform_preserves_data_and_invalidates_only_geometry_cache():
    mesh = _triangle_mesh()
    source_points = mesh.points.clone()
    source_area = mesh.cell_areas.clone()
    _ = mesh.cell_centroids
    topology = mesh.get_point_to_points_adjacency()
    controls, displacements = _controls_and_displacements()

    output = mesh.radial_basis_function_deform(
        controls,
        displacements,
        implementation="torch",
    )

    assert output is not mesh
    assert torch.equal(output.cells, mesh.cells)
    assert torch.equal(output.point_data["marker"], mesh.point_data["marker"])
    assert torch.equal(output.cell_data["material"], mesh.cell_data["material"])
    assert torch.equal(output.global_data["case_id"], mesh.global_data["case_id"])
    assert list(output._cache["cell"].keys()) == []
    assert list(output._cache["point"].keys()) == []
    cached_topology = output._cache.get(("topology", "point_to_points"))
    assert cached_topology is not None
    assert output.get_point_to_points_adjacency().to_list() == topology.to_list()
    assert cached_topology.offsets.data_ptr() == topology.offsets.data_ptr()
    assert cached_topology.indices.data_ptr() == topology.indices.data_ptr()

    torch.testing.assert_close(mesh.points, source_points)
    torch.testing.assert_close(mesh.cell_areas, source_area)
    assert mesh._cache.get(("cell", "areas")) is not None
    assert mesh._cache.get(("cell", "centroids")) is not None


def test_mesh_radial_basis_function_deform_resolves_nested_point_weight_key():
    mesh = _triangle_mesh()
    weights = torch.tensor([1.0, -0.5, 0.0], dtype=torch.float64)
    mesh.point_data["motion"] = TensorDict(
        {"weight": weights},
        batch_size=[mesh.n_points],
        device=mesh.points.device,
    )
    controls, displacements = _controls_and_displacements()

    output = mesh.radial_basis_function_deform(
        controls,
        displacements,
        point_weights=("motion", "weight"),
        implementation="torch",
    )

    expected = mesh.points + weights.unsqueeze(-1) * displacements
    torch.testing.assert_close(output.points, expected, atol=1e-12, rtol=1e-12)


def test_mesh_radial_basis_function_deform_preserves_autograd_through_fit_and_evaluation():
    points = torch.tensor(
        [[0.15, 0.2], [0.75, 0.2], [0.2, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mesh = Mesh(points=points, cells=torch.tensor([[0, 1, 2]]))
    controls = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    displacements = torch.tensor(
        [[0.1, 0.0], [0.0, 0.3], [-0.2, 0.1], [0.25, -0.15]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.tensor([0.8, 1.1, 0.6], dtype=torch.float64, requires_grad=True)

    output = mesh.radial_basis_function_deform(
        controls,
        displacements,
        smoothing=0.01,
        point_weights=weights,
        implementation="torch",
    )
    gradients = torch.autograd.grad(
        output.points.square().sum(),
        (points, controls, displacements, weights),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_mesh_radial_basis_function_deform_has_actionable_boundary_validation():
    mesh = _triangle_mesh()
    controls, displacements = _controls_and_displacements()

    with pytest.raises(TypeError, match="control_points"):
        mesh.radial_basis_function_deform(
            "handles", displacements, implementation="torch"
        )
    with pytest.raises(TypeError, match="control_displacements"):
        mesh.radial_basis_function_deform(controls, "motions", implementation="torch")
    with pytest.raises(KeyError, match="point_weights field 'missing'.*Available keys"):
        mesh.radial_basis_function_deform(
            controls,
            displacements,
            point_weights="missing",
            implementation="torch",
        )

    mesh.point_data["motion"] = TensorDict(
        {
            "weight": torch.ones(mesh.n_points, dtype=torch.float64),
            "mask": torch.ones(mesh.n_points, dtype=torch.bool),
        },
        batch_size=[mesh.n_points],
    )
    with pytest.raises(TypeError, match="must resolve to a torch.Tensor"):
        mesh.radial_basis_function_deform(
            controls,
            displacements,
            point_weights="motion",
            implementation="torch",
        )
    with pytest.raises(ValueError, match="kernel must be 'thin_plate_spline'"):
        mesh.radial_basis_function_deform(
            controls,
            displacements,
            kernel="gaussian",
            implementation="torch",
        )
    with pytest.raises(ValueError, match=r"requires at least D \+ 1 controls"):
        mesh.radial_basis_function_deform(
            controls[:2],
            displacements[:2],
            polynomial=True,
            implementation="torch",
        )


@pytest.mark.parametrize("point_weights", [None, "marker"])
def test_domain_radial_basis_function_deform_shared_controls_and_common_point_weight_key(
    point_weights,
):
    domain = _domain_with_coincident_points()
    controls, displacements = _controls_and_displacements()

    output = domain.radial_basis_function_deform(
        controls,
        displacements,
        point_weights=point_weights,
        implementation="torch",
    )

    torch.testing.assert_close(
        output.interior.points[:2], output.boundaries["wall"].points
    )
    assert output is not domain
    assert torch.equal(output.interior.cells, domain.interior.cells)
    assert torch.equal(
        output.boundaries["wall"].cell_data["boundary_id"],
        domain.boundaries["wall"].cell_data["boundary_id"],
    )
    torch.testing.assert_close(domain.interior.points, _triangle_mesh().points)


def test_domain_radial_basis_function_deform_resolves_common_nested_point_weight_path():
    domain = _domain_with_coincident_points()
    for component in (domain.interior, domain.boundaries["wall"]):
        component.point_data["motion"] = TensorDict(
            {"weight": component.point_data["marker"]},
            batch_size=[component.n_points],
            device=component.points.device,
        )
    controls, displacements = _controls_and_displacements()

    nested = domain.radial_basis_function_deform(
        controls,
        displacements,
        point_weights=("motion", "weight"),
        implementation="torch",
    )
    flat = domain.radial_basis_function_deform(
        controls,
        displacements,
        point_weights="marker",
        implementation="torch",
    )

    torch.testing.assert_close(nested.interior.points, flat.interior.points)
    torch.testing.assert_close(
        nested.boundaries["wall"].points, flat.boundaries["wall"].points
    )


def test_domain_radial_basis_function_deform_missing_common_key_names_failing_component():
    domain = _domain_with_coincident_points()
    del domain.boundaries["wall"].point_data["marker"]
    controls, displacements = _controls_and_displacements()

    with pytest.raises(
        KeyError,
        match=(
            r"point_weights field 'marker' not found in "
            r"boundaries\['wall'\]\.point_data"
        ),
    ):
        domain.radial_basis_function_deform(
            controls,
            displacements,
            point_weights="marker",
            implementation="torch",
        )


def test_domain_radial_basis_function_deform_rejects_raw_point_weight_tensor():
    domain = _domain_with_coincident_points()
    controls, displacements = _controls_and_displacements()

    with pytest.raises(TypeError, match="common point_data key/path"):
        domain.radial_basis_function_deform(
            controls,
            displacements,
            point_weights=torch.ones(domain.interior.n_points),
            implementation="torch",
        )


def test_domain_radial_basis_function_deform_clones_domain_global_data():
    domain = _domain_with_coincident_points()
    controls, displacements = _controls_and_displacements()
    original_reynolds = domain.global_data["reynolds"].clone()

    output = domain.radial_basis_function_deform(
        controls, displacements, implementation="torch"
    )

    assert output.global_data is not domain.global_data
    assert (
        output.global_data["reynolds"].data_ptr()
        != domain.global_data["reynolds"].data_ptr()
    )
    output.global_data["reynolds"].fill_(2.0e5)
    torch.testing.assert_close(domain.global_data["reynolds"], original_reynolds)


def test_domain_radial_basis_function_deform_evaluates_combined_components_once(
    monkeypatch,
):
    domain = _domain_with_coincident_points()
    domain.boundaries["outlet"] = Mesh(
        points=torch.tensor([[2.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=torch.float64),
        cells=torch.tensor([[0, 1], [1, 2]]),
        point_data={"marker": torch.tensor([0.5, 0.75, 0.25], dtype=torch.float64)},
    )
    controls, displacements = _controls_and_displacements()
    expected = {
        "interior": domain.interior.radial_basis_function_deform(
            controls,
            displacements,
            point_weights="marker",
            implementation="torch",
        ),
        **{
            name: component.radial_basis_function_deform(
                controls,
                displacements,
                point_weights="marker",
                implementation="torch",
            )
            for name, component in domain.boundaries.items()
        },
    }
    deform_module = importlib.import_module("physicsnemo.nn.functional.geometry.deform")
    original = deform_module.radial_basis_function_deform_points
    calls: list[torch.Tensor] = []
    options: list[tuple[str, bool, float]] = []

    def counted_radial_basis_function_deform_points(points, *args, **kwargs):
        calls.append(points)
        options.append((kwargs["kernel"], kwargs["polynomial"], kwargs["smoothing"]))
        return original(points, *args, **kwargs)

    monkeypatch.setattr(
        deform_module,
        "radial_basis_function_deform_points",
        counted_radial_basis_function_deform_points,
    )
    output = domain.radial_basis_function_deform(
        controls,
        displacements,
        point_weights="marker",
        implementation="torch",
    )

    assert len(calls) == 1
    assert calls[0].shape == (8, 2)
    assert options == [("thin_plate_spline", True, 0.0)]
    torch.testing.assert_close(output.interior.points, expected["interior"].points)
    for name, boundary in output.boundaries.items():
        torch.testing.assert_close(boundary.points, expected[name].points)


def test_domain_radial_basis_function_deform_single_component_avoids_concatenation(
    monkeypatch,
):
    domain = DomainMesh(interior=_triangle_mesh())
    deform_module = importlib.import_module("physicsnemo.nn.functional.geometry.deform")
    original = deform_module.radial_basis_function_deform_points
    received_points: list[torch.Tensor] = []

    def inspect_radial_basis_function_deform_points(points, *args, **kwargs):
        received_points.append(points)
        return original(points, *args, **kwargs)

    monkeypatch.setattr(
        deform_module,
        "radial_basis_function_deform_points",
        inspect_radial_basis_function_deform_points,
    )
    controls, displacements = _controls_and_displacements()
    output = domain.radial_basis_function_deform(
        controls, displacements, implementation="torch"
    )

    assert len(received_points) == 1
    assert received_points[0] is domain.interior.points
    assert received_points[0].data_ptr() == domain.interior.points.data_ptr()
    assert output.interior.n_points == domain.interior.n_points


def test_domain_combined_radial_basis_function_deform_preserves_component_autograd():
    domain = _domain_with_coincident_points()
    interior_points = domain.interior.points.requires_grad_()
    wall_points = domain.boundaries["wall"].points.requires_grad_()
    interior_weights = domain.interior.point_data["marker"].requires_grad_()
    wall_weights = domain.boundaries["wall"].point_data["marker"].requires_grad_()
    controls = torch.tensor(
        [[-0.2, -0.1], [1.2, 0.0], [0.0, 1.2], [1.1, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    displacements = torch.tensor(
        [[0.1, 0.0], [0.0, 0.3], [-0.2, 0.1], [0.25, -0.15]],
        dtype=torch.float64,
        requires_grad=True,
    )

    output = domain.radial_basis_function_deform(
        controls,
        displacements,
        smoothing=0.01,
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
            interior_weights,
            wall_weights,
            controls,
            displacements,
        ),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
