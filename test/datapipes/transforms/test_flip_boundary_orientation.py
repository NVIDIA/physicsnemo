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

"""Tests for the FlipBoundaryOrientation transform."""

import pytest
import torch

from physicsnemo.datapipes.registry import COMPONENT_REGISTRY
from physicsnemo.datapipes.transforms.mesh import FlipBoundaryOrientation
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import single_triangle_3d


def _surface() -> Mesh:
    """Two triangles in 3D with stored normals and an effective measure."""
    mesh = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.5]]
        ),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        cell_data={"p": torch.tensor([1.0, 2.0]), "_effective_measure": torch.ones(2)},
    )
    return mesh.with_data(
        cell_data=mesh.cell_data.update({"normals": mesh.cell_normals})
    )


class TestFlipBoundaryOrientation:
    """FlipBoundaryOrientation reverses winding and stored normals together."""

    def test_mesh_reversed_with_its_stored_normals(self):
        """Each normal flips; points, measures and other fields stay as they are."""
        mesh = _surface()
        out = FlipBoundaryOrientation(normal_fields=["normals"])(mesh)
        assert torch.equal(out.cells, mesh.cells[:, [0, 2, 1]])
        torch.testing.assert_close(out.cell_normals, -mesh.cell_normals)
        torch.testing.assert_close(out.cell_data["normals"], out.cell_normals)
        torch.testing.assert_close(out.cell_areas, mesh.cell_areas)
        torch.testing.assert_close(out.cell_centroids, mesh.cell_centroids)
        for key in ("p", "_effective_measure"):
            assert torch.equal(out.cell_data[key], mesh.cell_data[key])
        assert torch.equal(out.points, mesh.points)
        ### The input keeps its own stored normals.
        torch.testing.assert_close(mesh.cell_data["normals"], mesh.cell_normals)

    def test_flip_twice_is_identity(self):
        """Reversing twice restores the cells and the stored normals exactly."""
        mesh = _surface()
        flip = FlipBoundaryOrientation(normal_fields=["normals"])
        out = flip(flip(mesh))
        assert torch.equal(out.cells, mesh.cells)
        assert torch.equal(out.cell_data["normals"], mesh.cell_data["normals"])

    def test_edges_in_2d(self):
        """A 2D boundary of edges reverses its normals too."""
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
            cells=torch.tensor([[0, 1], [1, 2]]),
        )
        out = FlipBoundaryOrientation()(mesh)
        torch.testing.assert_close(out.cell_normals, -mesh.cell_normals)

    def test_domain_flips_only_named_boundaries(self):
        """On a DomainMesh only the named boundaries change."""
        wall = _surface()
        inlet = single_triangle_3d.load()
        domain = DomainMesh(
            interior=Mesh(points=torch.randn(5, 3)),
            boundaries={"wall": wall, "inlet": inlet},
            global_data={"U_inf": torch.tensor([1.0, 0.0, 0.0])},
        )
        out = FlipBoundaryOrientation(
            boundaries=["wall"], normal_fields=["normals"]
        ).apply_to_domain(domain)
        torch.testing.assert_close(
            out.boundaries["wall"].cell_normals, -wall.cell_normals
        )
        assert torch.equal(
            out.boundaries["wall"].cell_data["normals"], -wall.cell_data["normals"]
        )
        assert torch.equal(out.boundaries["inlet"].cells, inlet.cells)
        assert torch.equal(out.interior.points, domain.interior.points)
        assert torch.equal(out.global_data["U_inf"], domain.global_data["U_inf"])

    def test_missing_boundary_raises(self):
        """A named boundary the domain lacks is an error."""
        domain = DomainMesh(
            interior=Mesh(points=torch.randn(5, 3)),
            boundaries={"inlet": single_triangle_3d.load()},
        )
        with pytest.raises(KeyError, match="wall"):
            FlipBoundaryOrientation(boundaries=["wall"]).apply_to_domain(domain)

    def test_point_cloud_raises(self):
        """A mesh without oriented cells cannot be reversed."""
        with pytest.raises(ValueError, match="at least 2 vertices"):
            FlipBoundaryOrientation()(Mesh(points=torch.randn(4, 3)))

    def test_registered(self):
        """The transform is in the datapipe registry, for ``${dp:...}`` configs."""
        assert "FlipBoundaryOrientation" in COMPONENT_REGISTRY
