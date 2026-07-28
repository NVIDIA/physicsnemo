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

"""Mesh and DomainMesh integration tests for nonlinear point deformation."""

import importlib
import inspect

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.transformations.deform import displace, morph


def test_deform_namespace_is_canonical():
    """Deformations are public only from their dedicated namespace."""

    transformations = importlib.import_module("physicsnemo.mesh.transformations")
    deform_module = importlib.import_module("physicsnemo.mesh.transformations.deform")
    geometric_module = importlib.import_module(
        "physicsnemo.mesh.transformations.geometric"
    )

    assert transformations.deform is deform_module
    assert deform_module.displace is displace
    assert deform_module.morph is morph
    assert not hasattr(transformations, "displace")
    assert not hasattr(transformations, "morph")
    assert not hasattr(geometric_module, "displace")
    assert not hasattr(geometric_module, "morph")


def test_mesh_morph_signatures_are_introspectable():
    """Generated ``.float()`` methods must not break public annotations."""

    for morph_method in (Mesh.morph, DomainMesh.morph):
        signature = inspect.signature(morph_method)
        assert signature.parameters["radius"].annotation == float | torch.Tensor
        assert signature.parameters["kernel"].default == "wendland_c2"


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


def test_mesh_displace_resolves_nested_keys_and_preserves_data():
    mesh = _triangle_mesh()
    displacement = torch.tensor([[0.0, 2.0], [1.0, 0.0], [-1.0, 1.0]])
    mesh.point_data["motion"] = TensorDict(
        {
            "delta": displacement,
            "weight": torch.tensor([0.5, -1.0, 0.0]),
        },
        batch_size=[mesh.n_points],
    )
    source_points = mesh.points.clone()

    output = mesh.displace(
        ("motion", "delta"),
        point_weights=("motion", "weight"),
        implementation="torch",
    )

    expected = source_points + torch.tensor([[0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(output.points, expected)
    torch.testing.assert_close(mesh.points, source_points)
    assert output is not mesh
    assert torch.equal(output.cells, mesh.cells)
    assert torch.equal(output.point_data["temperature"], mesh.point_data["temperature"])
    assert torch.equal(output.cell_data["material"], mesh.cell_data["material"])
    assert torch.equal(output.global_data["case_id"], mesh.global_data["case_id"])

    # The standalone transformation export follows the same key-resolution path.
    direct = displace(mesh, ("motion", "delta"), implementation="torch")
    torch.testing.assert_close(direct.points, source_points + displacement)


def test_mesh_morph_invalidates_geometry_but_reuses_topology_cache():
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )
    original_area = mesh.cell_areas.clone()
    _ = mesh.cell_centroids
    _ = mesh.point_normals
    topology = mesh.get_point_to_points_adjacency()

    controls = torch.tensor([[0.0, 0.0, 0.0]])
    control_displacements = torch.tensor([[0.0, 0.25, 0.0]])
    output = morph(
        mesh,
        controls,
        control_displacements,
        radius=2.0,
        implementation="torch",
    )

    assert list(output._cache["cell"].keys()) == []
    assert list(output._cache["point"].keys()) == []
    cached_topology = output._cache.get(("topology", "point_to_points"))
    assert cached_topology is not None
    assert output.get_point_to_points_adjacency().to_list() == topology.to_list()
    assert cached_topology.offsets.data_ptr() == topology.offsets.data_ptr()
    assert cached_topology.indices.data_ptr() == topology.indices.data_ptr()

    # The source remains unchanged and keeps its already-computed geometry.
    torch.testing.assert_close(mesh.cell_areas, original_area)
    assert mesh._cache.get(("cell", "areas")) is not None
    assert mesh._cache.get(("cell", "centroids")) is not None
    assert mesh._cache.get(("point", "normals")) is not None


def test_mesh_displace_preserves_autograd_through_returned_points():
    mesh = _triangle_mesh(requires_grad=True)
    displacement = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.7]], requires_grad=True
    )
    output = mesh.displace(displacement, implementation="torch")
    loss = output.points.square().sum()
    loss.backward()

    assert mesh.points.grad is not None
    assert displacement.grad is not None
    assert torch.isfinite(mesh.points.grad).all()
    assert torch.isfinite(displacement.grad).all()


def test_mesh_missing_point_data_key_has_actionable_diagnostic():
    mesh = _triangle_mesh()
    with pytest.raises(KeyError, match="displacement field 'missing'.*Available keys"):
        mesh.displace("missing", implementation="torch")
    with pytest.raises(KeyError, match="point_weights field 'missing'.*Available keys"):
        mesh.morph(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            radius=1.0,
            point_weights="missing",
            implementation="torch",
        )


def test_mesh_point_data_path_through_leaf_has_actionable_diagnostic():
    mesh = _triangle_mesh()

    # 'temperature' is a leaf tensor; descending into it must produce the same
    # actionable KeyError diagnostic as a missing key, not an internal
    # tensordict ValueError.
    with pytest.raises(KeyError, match="displacement field.*Available keys"):
        mesh.displace(("temperature", "x"), implementation="torch")
    with pytest.raises(KeyError, match="point_weights field.*Available keys"):
        mesh.morph(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            radius=1.0,
            point_weights=("temperature", "x"),
            implementation="torch",
        )


def test_mesh_displace_rejects_point_weights_resolving_to_tensordict():
    mesh = _triangle_mesh()
    mesh.point_data["motion"] = TensorDict(
        {"weight": torch.tensor([0.5, 1.0, 0.0]), "mask": torch.ones(3)},
        batch_size=[mesh.n_points],
        device=mesh.points.device,
    )
    displacement = torch.zeros_like(mesh.points)

    # A nested TensorDict is not a valid tensor-valued point field and must be
    # rejected at the API boundary.
    with pytest.raises(TypeError, match="must resolve to a torch.Tensor"):
        mesh.displace(displacement, point_weights="motion", implementation="torch")


def test_mesh_morph_rejects_non_tensor_control_points():
    mesh = _triangle_mesh()

    # DomainMesh.morph raises a descriptive TypeError for this identical
    # mistake; Mesh.morph must not die with AttributeError deep inside input
    # normalization.
    with pytest.raises(TypeError, match="control_points"):
        mesh.morph(
            "handles",
            torch.tensor([[0.0, 1.0]]),
            radius=0.5,
            implementation="torch",
        )


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
def test_domain_morph_shared_controls_and_common_point_weight_key(point_weights):
    domain = _domain_with_coincident_points()
    controls = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    control_displacements = torch.tensor([[0.0, 0.5], [0.25, 0.0]])
    output = domain.morph(
        controls,
        control_displacements,
        radius=torch.tensor([1.2, 1.2]),
        point_weights=point_weights,
        implementation="torch",
    )

    # Coincident component points with the same common-key values move identically.
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


def test_domain_morph_clones_domain_global_data():
    domain = _domain_with_coincident_points()
    output = domain.morph(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.5]]),
        radius=2.0,
        implementation="torch",
    )

    # Every other DomainMesh operation delegates to apply_to_meshes, which
    # always clones global_data; morph must not hand back an aliased
    # TensorDict whose mutation corrupts the source domain.
    assert output.global_data is not domain.global_data
    output.global_data["reynolds"] = torch.tensor(2.0e5)
    torch.testing.assert_close(domain.global_data["reynolds"], torch.tensor(1.0e5))


def test_domain_morph_missing_common_point_weight_names_failing_component():
    domain = _domain_with_coincident_points()
    del domain.boundaries["wall"].point_data["marker"]

    with pytest.raises(
        KeyError,
        match=(
            r"point_weights field 'marker' not found in "
            r"boundaries\['wall'\]\.point_data"
        ),
    ):
        domain.morph(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            radius=1.0,
            point_weights="marker",
            implementation="torch",
        )


def test_domain_morph_validates_common_point_weight_dtype_before_concatenation():
    domain = _domain_with_coincident_points()
    domain.boundaries["wall"].point_data["marker"] = torch.tensor([True, False])

    with pytest.raises(
        TypeError,
        match=r"one common dtype.*boundaries\['wall'\]\.point_data",
    ):
        domain.morph(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            radius=1.0,
            point_weights="marker",
            implementation="torch",
        )


def test_domain_morph_resolves_common_nested_point_weight_path():
    domain = _domain_with_coincident_points()
    for component in (domain.interior, domain.boundaries["wall"]):
        component.point_data["motion"] = TensorDict(
            {"weight": component.point_data["marker"]},
            batch_size=[component.n_points],
        )

    controls = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    displacements = torch.tensor([[0.0, 0.5], [0.25, 0.0]])
    nested = domain.morph(
        controls,
        displacements,
        radius=1.2,
        point_weights=("motion", "weight"),
        implementation="torch",
    )
    flat = domain.morph(
        controls,
        displacements,
        radius=1.2,
        point_weights="marker",
        implementation="torch",
    )

    torch.testing.assert_close(nested.interior.points, flat.interior.points)
    torch.testing.assert_close(
        nested.boundaries["wall"].points, flat.boundaries["wall"].points
    )


def test_domain_morph_rejects_raw_point_weight_tensor():
    domain = _domain_with_coincident_points()
    with pytest.raises(TypeError, match="common point_data key/path"):
        domain.morph(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            radius=1.0,
            point_weights=torch.ones(domain.interior.n_points),
            implementation="torch",
        )


def test_domain_morph_evaluates_combined_components_once(monkeypatch):
    domain = _domain_with_coincident_points()
    outlet = Mesh(
        points=torch.tensor([[2.0, 0.0], [2.0, 1.0], [3.0, 0.0]]),
        cells=torch.tensor([[0, 1], [1, 2]]),
        point_data={"marker": torch.tensor([0.5, 0.75, 0.25])},
    )
    domain.boundaries["outlet"] = outlet

    control_points = torch.tensor([[0.0, 0.0]])
    control_displacements = torch.tensor([[0.0, 0.5]])
    expected = {
        "interior": domain.interior.morph(
            control_points,
            control_displacements,
            radius=3.0,
            point_weights="marker",
            implementation="torch",
        ),
        **{
            name: component.morph(
                control_points,
                control_displacements,
                radius=3.0,
                point_weights="marker",
                implementation="torch",
            )
            for name, component in domain.boundaries.items()
        },
    }

    deform_module = importlib.import_module("physicsnemo.nn.functional.geometry.deform")
    original = deform_module.morph_points
    calls: list[torch.Tensor] = []
    kernels: list[str] = []

    def counted_morph_points(points, *args, **kwargs):
        calls.append(points)
        kernels.append(kwargs["kernel"])
        return original(points, *args, **kwargs)

    monkeypatch.setattr(deform_module, "morph_points", counted_morph_points)
    output = domain.morph(
        control_points,
        control_displacements,
        radius=3.0,
        point_weights="marker",
        implementation="torch",
    )

    assert len(calls) == 1
    assert kernels == ["wendland_c2"]
    assert calls[0].shape == (8, 2)
    torch.testing.assert_close(output.interior.points, expected["interior"].points)
    for name, boundary in output.boundaries.items():
        torch.testing.assert_close(boundary.points, expected[name].points)


def test_domain_morph_single_component_avoids_concatenation(monkeypatch):
    domain = DomainMesh(interior=_triangle_mesh())
    deform_module = importlib.import_module("physicsnemo.nn.functional.geometry.deform")
    original = deform_module.morph_points
    received_points: list[torch.Tensor] = []

    def inspect_morph_points(points, *args, **kwargs):
        received_points.append(points)
        return original(points, *args, **kwargs)

    monkeypatch.setattr(deform_module, "morph_points", inspect_morph_points)
    output = domain.morph(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.5]]),
        radius=2.0,
        implementation="torch",
    )

    assert len(received_points) == 1
    assert received_points[0] is domain.interior.points
    assert received_points[0].data_ptr() == domain.interior.points.data_ptr()
    assert output.interior.n_points == domain.interior.n_points


def test_domain_combined_morph_preserves_component_autograd():
    domain = _domain_with_coincident_points()
    interior_points = domain.interior.points.requires_grad_()
    wall_points = domain.boundaries["wall"].points.requires_grad_()
    interior_point_weights = domain.interior.point_data["marker"].requires_grad_()
    wall_point_weights = domain.boundaries["wall"].point_data["marker"].requires_grad_()
    control_displacements = torch.tensor([[0.2, 0.5], [-0.1, 0.25]], requires_grad=True)

    output = domain.morph(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        control_displacements,
        radius=1.5,
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
