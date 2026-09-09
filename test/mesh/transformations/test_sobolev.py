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

"""Mesh integration tests for Sobolev deformation."""

import importlib
import inspect

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.transformations.deform import sobolev_deform
from physicsnemo.nn.functional import sobolev_deform_points


def _surface_mesh(*, requires_grad: bool = False) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.2],
            [0.0, 1.0, 0.0],
        ],
        requires_grad=requires_grad,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return Mesh(
        points=points,
        cells=cells,
        point_data={
            "motion": torch.tensor(
                [
                    [0.0, 0.0, 0.1],
                    [0.1, 0.0, -0.2],
                    [0.0, -0.1, 0.3],
                    [-0.1, 0.0, -0.1],
                ]
            ),
            "fixed": torch.tensor([True, False, False, False]),
            "temperature": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        },
        cell_data={"material": torch.tensor([7, 8])},
        global_data={"case_id": torch.tensor(12)},
    )


def test_sobolev_deform_namespace_and_mesh_method():
    transformations = importlib.import_module("physicsnemo.mesh.transformations")
    deform = importlib.import_module("physicsnemo.mesh.transformations.deform")

    assert deform.sobolev_deform is sobolev_deform
    assert "sobolev_deform" in deform.__all__
    assert not hasattr(transformations, "sobolev_deform")
    assert Mesh.sobolev_deform is sobolev_deform

    signature = inspect.signature(Mesh.sobolev_deform)
    assert list(signature.parameters) == [
        "mesh",
        "displacement",
        "length_scale",
        "fixed_points",
        "max_iterations",
        "tolerance",
        "implementation",
    ]
    assert signature.parameters["length_scale"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "warp" in str(signature.parameters["implementation"].annotation)


def test_mesh_wrapper_resolves_fields_and_preserves_attached_data():
    mesh = _surface_mesh()
    source_points = mesh.points.clone()

    output = mesh.sobolev_deform(
        "motion",
        length_scale=0.4,
        fixed_points="fixed",
        max_iterations=64,
        tolerance=1.0e-7,
        implementation="torch",
    )
    expected_points = sobolev_deform_points(
        source_points,
        mesh.cells,
        mesh.point_data["motion"],
        length_scale=0.4,
        fixed_points=mesh.point_data["fixed"],
        max_iterations=64,
        tolerance=1.0e-7,
        implementation="torch",
    )

    torch.testing.assert_close(output.points, expected_points)
    torch.testing.assert_close(mesh.points, source_points)
    assert torch.equal(output.points[0], source_points[0])
    assert output is not mesh
    assert torch.equal(output.cells, mesh.cells)
    assert torch.equal(output.point_data["temperature"], mesh.point_data["temperature"])
    assert torch.equal(output.cell_data["material"], mesh.cell_data["material"])
    assert torch.equal(output.global_data["case_id"], mesh.global_data["case_id"])

    direct = sobolev_deform(
        mesh,
        "motion",
        length_scale=0.4,
        fixed_points="fixed",
        implementation="torch",
    )
    expected_direct = sobolev_deform_points(
        source_points,
        mesh.cells,
        mesh.point_data["motion"],
        length_scale=0.4,
        fixed_points=mesh.point_data["fixed"],
        implementation="torch",
    )
    torch.testing.assert_close(direct.points, expected_direct)


def test_mesh_sobolev_deform_invalidates_geometry_and_retains_topology():
    mesh = _surface_mesh()
    original_areas = mesh.cell_areas.clone()
    _ = mesh.cell_centroids
    _ = mesh.point_normals
    topology = mesh.get_point_to_points_adjacency()

    output = mesh.sobolev_deform(
        "motion",
        length_scale=0.25,
        implementation="torch",
    )

    assert list(output._cache["cell"].keys()) == []
    assert list(output._cache["point"].keys()) == []
    cached_topology = output._cache.get(("topology", "point_to_points"))
    assert cached_topology is not None
    assert output.get_point_to_points_adjacency().to_list() == topology.to_list()
    assert cached_topology.offsets.data_ptr() == topology.offsets.data_ptr()
    assert cached_topology.indices.data_ptr() == topology.indices.data_ptr()

    torch.testing.assert_close(mesh.cell_areas, original_areas)
    assert mesh._cache.get(("cell", "areas")) is not None
    assert mesh._cache.get(("cell", "centroids")) is not None
    assert mesh._cache.get(("point", "normals")) is not None


def test_mesh_sobolev_deform_preserves_autograd():
    mesh = _surface_mesh(requires_grad=True)
    displacement = mesh.point_data["motion"].detach().clone().requires_grad_(True)

    output = mesh.sobolev_deform(
        displacement,
        length_scale=0.3,
        max_iterations=64,
        tolerance=1.0e-7,
        implementation="torch",
    )
    output.points.square().sum().backward()

    assert mesh.points.grad is not None
    assert displacement.grad is not None
    assert torch.isfinite(mesh.points.grad).all()
    assert torch.isfinite(displacement.grad).all()


def test_mesh_sobolev_deform_reports_missing_fields():
    mesh = _surface_mesh()

    with pytest.raises(KeyError, match="displacement field 'missing'.*Available keys"):
        mesh.sobolev_deform(
            "missing",
            length_scale=0.2,
            implementation="torch",
        )
    with pytest.raises(KeyError, match="fixed_points field 'missing'.*Available keys"):
        mesh.sobolev_deform(
            "motion",
            length_scale=0.2,
            fixed_points="missing",
            implementation="torch",
        )
