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

"""Tests for the recipe-local surface mesh transforms."""

import pytest
import torch
from domain_transforms import ComputeFreestreamDirection, DropDegenerateCells
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh


def _two_triangles(second_degenerate: bool = False) -> Mesh:
    """Two triangles in the xy-plane; the second is optionally collapsed."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    if second_degenerate:
        points[5] = points[3]
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        cell_data={"pressure": torch.tensor([10.0, 20.0])},
        global_data={"U_inf": torch.tensor([3.0, 0.0, 4.0])},
    )


class TestComputeFreestreamDirection:
    """ComputeFreestreamDirection on meshes and domains."""

    def test_adds_unit_direction_and_keeps_velocity(self):
        """The unit direction is added and the physical velocity is untouched."""
        mesh = _two_triangles()
        out = ComputeFreestreamDirection()(mesh)

        torch.testing.assert_close(
            out.global_data["U_inf_dir"], torch.tensor([0.6, 0.0, 0.8])
        )
        torch.testing.assert_close(out.global_data["U_inf"], mesh.global_data["U_inf"])
        assert "U_inf_dir" not in mesh.global_data.keys()
        assert torch.equal(out.points, mesh.points)
        assert torch.equal(out.cell_data["pressure"], mesh.cell_data["pressure"])

    def test_custom_field_names(self):
        """Input and output field names are configurable."""
        mesh = Mesh(
            points=torch.zeros(3, 3),
            cells=torch.tensor([[0, 1, 2]]),
            global_data={"velocity": torch.tensor([0.0, -2.0, 0.0])},
        )
        out = ComputeFreestreamDirection(
            velocity_field="velocity", output_field="direction"
        )(mesh)

        torch.testing.assert_close(
            out.global_data["direction"], torch.tensor([0.0, -1.0, 0.0])
        )

    def test_domain_writes_domain_level_global_data(self):
        """On a DomainMesh the direction lands on the domain-level global_data."""
        boundary = _two_triangles()
        domain = DomainMesh(
            interior=Mesh(points=torch.zeros(2, 3)),
            boundaries={"vehicle": boundary},
            global_data=TensorDict({"U_inf": torch.tensor([0.0, 5.0, 0.0])}),
        )
        out = ComputeFreestreamDirection().apply_to_domain(domain)

        torch.testing.assert_close(
            out.global_data["U_inf_dir"], torch.tensor([0.0, 1.0, 0.0])
        )
        assert "U_inf_dir" not in out.boundaries["vehicle"].global_data.keys()

    def test_missing_velocity_raises(self):
        """A missing velocity field is a KeyError naming the field."""
        mesh = Mesh(points=torch.zeros(3, 3), cells=torch.tensor([[0, 1, 2]]))
        with pytest.raises(KeyError, match="U_inf"):
            ComputeFreestreamDirection()(mesh)

    def test_zero_velocity_raises(self):
        """A zero freestream vector has no direction and raises."""
        mesh = Mesh(
            points=torch.zeros(3, 3),
            cells=torch.tensor([[0, 1, 2]]),
            global_data={"U_inf": torch.zeros(3)},
        )
        with pytest.raises(ValueError, match="finite and positive"):
            ComputeFreestreamDirection()(mesh)


class TestDropDegenerateCells:
    """DropDegenerateCells on healthy, degenerate, and cell-free meshes."""

    def test_healthy_mesh_passes_through(self):
        """A mesh without degenerate cells is returned as-is."""
        mesh = _two_triangles()
        assert DropDegenerateCells()(mesh) is mesh

    def test_drops_zero_area_cell_and_its_data(self):
        """A collapsed triangle is dropped together with its cell_data row."""
        mesh = _two_triangles(second_degenerate=True)
        with pytest.warns(UserWarning, match="dropping 1 cell"):
            out = DropDegenerateCells()(mesh)

        assert out.n_cells == 1
        assert torch.equal(out.cells, torch.tensor([[0, 1, 2]]))
        assert torch.equal(out.cell_data["pressure"], torch.tensor([10.0]))
        assert torch.all(out.cell_areas > 0)

    def test_point_cloud_passes_through(self):
        """A mesh without cells has nothing to drop."""
        cloud = Mesh(points=torch.rand(4, 3))
        assert DropDegenerateCells()(cloud) is cloud
