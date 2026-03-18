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

"""Tests for DomainMesh transform passthrough methods."""

import math

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import (
    single_edge_2d,
    single_tetrahedron,
    single_triangle_2d,
    single_triangle_3d,
    two_tetrahedra,
)

### Fixtures


@pytest.fixture
def tet_domain():
    """DomainMesh: tet interior (m=3, s=3) with data, 2 tri boundaries, global_data."""
    interior = two_tetrahedra.load()
    interior.point_data["temperature"] = torch.randn(interior.n_points)
    interior.cell_data["pressure"] = torch.randn(interior.n_cells)
    wall = single_triangle_3d.load()
    wall.cell_data["wall_shear"] = torch.randn(wall.n_cells)
    inlet = single_triangle_3d.load()
    inlet.cell_data["mass_flux"] = torch.randn(inlet.n_cells)
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall, "inlet": inlet},
        global_data={"Re": torch.tensor(1e6), "AoA": torch.tensor(5.0)},
    )


@pytest.fixture
def no_boundary_domain():
    """DomainMesh: single tet interior with no boundaries or global data."""
    return DomainMesh(interior=single_tetrahedron.load())


### _map_meshes


class TestMapMeshes:
    """Tests for the _map_meshes private helper."""

    def test_applies_fn_to_interior(self, tet_domain):
        original_points = tet_domain.interior.points.clone()
        dm2 = tet_domain._map_meshes(lambda m: m.translate([1, 0, 0]))
        expected = original_points + torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(dm2.interior.points, expected)

    def test_applies_fn_to_all_boundaries(self, tet_domain):
        offset = torch.tensor([0.0, 0.0, 1.0])
        dm2 = tet_domain._map_meshes(lambda m: m.translate([0, 0, 1]))
        for name in tet_domain.boundary_names:
            original = tet_domain.boundaries[name].points
            assert torch.allclose(dm2.boundaries[name].points, original + offset)

    def test_preserves_global_data(self, tet_domain):
        dm2 = tet_domain._map_meshes(lambda m: m.translate([1, 1, 1]))
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])
        assert torch.equal(dm2.global_data["AoA"], tet_domain.global_data["AoA"])

    def test_works_with_no_boundaries(self, no_boundary_domain):
        dm2 = no_boundary_domain._map_meshes(lambda m: m.translate([1, 0, 0]))
        assert dm2.n_boundaries == 0
        assert dm2.interior.points[0, 0].item() == pytest.approx(1.0)

    def test_returns_domain_mesh(self, tet_domain):
        dm2 = tet_domain._map_meshes(lambda m: m)
        assert isinstance(dm2, DomainMesh)


### Geometric Transforms


class TestTranslate:
    """Tests for DomainMesh.translate passthrough."""

    def test_shifts_all_points(self, tet_domain):
        offset = [2.0, -1.0, 0.5]
        dm2 = tet_domain.translate(offset)
        offset_t = torch.tensor(offset)
        assert torch.allclose(
            dm2.interior.points, tet_domain.interior.points + offset_t
        )
        for name in tet_domain.boundary_names:
            assert torch.allclose(
                dm2.boundaries[name].points,
                tet_domain.boundaries[name].points + offset_t,
            )

    def test_preserves_cells(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0])
        assert torch.equal(dm2.interior.cells, tet_domain.interior.cells)

    def test_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0])
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])


class TestRotate:
    """Tests for DomainMesh.rotate passthrough."""

    def test_2d_rotation(self):
        """Rotate a 2D domain by 90 degrees; verify point (1,0) -> (0,1)."""
        dm = DomainMesh(
            interior=single_triangle_2d.load(),
            boundaries={"edge": single_edge_2d.load()},
        )
        dm2 = dm.rotate(angle=math.pi / 2)
        # Primitive point[1] = (1, 0) rotates to (0, 1)
        assert dm2.interior.points[1, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.interior.points[1, 1].item() == pytest.approx(1.0, abs=1e-6)

    def test_roundtrip(self, tet_domain):
        """Rotating and un-rotating recovers original points."""
        dm2 = tet_domain.rotate(angle=math.pi / 4, axis="z")
        dm3 = dm2.rotate(angle=-math.pi / 4, axis="z")
        assert torch.allclose(
            dm3.interior.points, tet_domain.interior.points, atol=1e-6
        )


class TestScale:
    """Tests for DomainMesh.scale passthrough."""

    def test_uniform_scale(self, tet_domain):
        dm2 = tet_domain.scale(factor=2.0)
        assert torch.allclose(dm2.interior.points, tet_domain.interior.points * 2.0)
        for name in tet_domain.boundary_names:
            assert torch.allclose(
                dm2.boundaries[name].points,
                tet_domain.boundaries[name].points * 2.0,
            )

    def test_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.scale(factor=3.0)
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])


class TestTransform:
    """Tests for DomainMesh.transform passthrough."""

    def test_identity(self, tet_domain):
        dm2 = tet_domain.transform(torch.eye(3))
        assert torch.allclose(dm2.interior.points, tet_domain.interior.points)

    def test_scale_via_matrix(self, tet_domain):
        dm2 = tet_domain.transform(2.0 * torch.eye(3))
        assert torch.allclose(dm2.interior.points, tet_domain.interior.points * 2.0)


### Cleanup / Refinement


class TestClean:
    """Tests for DomainMesh.clean passthrough."""

    def test_cleans_all_meshes(self):
        """clean() merges duplicate points in interior; boundary unchanged."""
        # Interior with intentional duplicate points (no primitive has this)
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
            cells=torch.tensor([[0, 1, 3], [2, 1, 3]]),
        )
        boundary = single_edge_2d.load()
        dm = DomainMesh(interior=interior, boundaries={"edge": boundary})
        dm2 = dm.clean()
        assert dm2.interior.n_points < dm.interior.n_points
        assert dm2.boundaries["edge"].n_points == boundary.n_points

    def test_no_boundaries(self, no_boundary_domain):
        dm2 = no_boundary_domain.clean()
        assert isinstance(dm2, DomainMesh)
        assert dm2.n_boundaries == 0


class TestStripCaches:
    """Tests for DomainMesh.strip_caches passthrough."""

    def test_clears_cached_geometry(self):
        """Accessing cell_normals populates cache; strip_caches clears it."""
        dm = DomainMesh(interior=single_triangle_3d.load())
        _ = dm.interior.cell_normals
        assert "normals" in dm.interior._cache["cell"].keys()
        dm2 = dm.strip_caches()
        assert "normals" not in dm2.interior._cache["cell"].keys()


class TestSubdivide:
    """Tests for DomainMesh.subdivide passthrough."""

    def test_increases_cell_count(self):
        """Linear subdivision: tet -> 8 child tets, tri -> 4 child tris."""
        dm = DomainMesh(
            interior=single_tetrahedron.load(),
            boundaries={"wall": single_triangle_3d.load()},
        )
        dm2 = dm.subdivide(levels=1, filter="linear")
        assert dm2.interior.n_cells == 8
        assert dm2.boundaries["wall"].n_cells == 4


### Data Operations


class TestCellDataToPointData:
    """Tests for DomainMesh.cell_data_to_point_data passthrough."""

    def test_converts_all_meshes(self, tet_domain):
        dm2 = tet_domain.cell_data_to_point_data()
        assert "pressure" in dm2.interior.point_data.keys()
        assert "wall_shear" in dm2.boundaries["wall"].point_data.keys()
        assert "mass_flux" in dm2.boundaries["inlet"].point_data.keys()

    def test_preserves_original_cell_data(self, tet_domain):
        dm2 = tet_domain.cell_data_to_point_data()
        assert "pressure" in dm2.interior.cell_data.keys()


class TestPointDataToCellData:
    """Tests for DomainMesh.point_data_to_cell_data passthrough."""

    def test_converts_all_meshes(self, tet_domain):
        dm2 = tet_domain.point_data_to_cell_data()
        assert "temperature" in dm2.interior.cell_data.keys()


### Validation


class TestValidate:
    """Tests for DomainMesh.validate passthrough."""

    def test_report_structure(self, tet_domain):
        report = tet_domain.validate()
        assert "interior" in report
        assert "boundaries" in report
        assert "valid" in report
        assert isinstance(report["valid"], bool)

    def test_report_contains_all_boundaries(self, tet_domain):
        report = tet_domain.validate()
        assert set(report["boundaries"].keys()) == {"wall", "inlet"}

    def test_no_boundaries(self, no_boundary_domain):
        report = no_boundary_domain.validate()
        assert report["boundaries"] == {}
        assert report["valid"] == report["interior"]["valid"]

    def test_invalid_mesh_propagates(self):
        """Out-of-bounds cell index causes valid=False."""
        # Intentionally invalid mesh (no primitive has out-of-bounds indices)
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            cells=torch.tensor([[0, 1, 99]]),
        )
        dm = DomainMesh(interior=interior)
        report = dm.validate()
        assert not report["valid"]


### Chaining


class TestChaining:
    """Tests for chaining multiple DomainMesh transforms."""

    def test_translate_scale_clean(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0]).scale(2.0).clean(merge_points=False)
        assert isinstance(dm2, DomainMesh)
        expected = (tet_domain.interior.points + torch.tensor([1.0, 0.0, 0.0])) * 2.0
        assert torch.allclose(dm2.interior.points, expected)

    def test_chain_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0]).rotate(0.1, axis="z").scale(0.5)
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])

    def test_chain_with_no_boundaries(self, no_boundary_domain):
        dm2 = no_boundary_domain.translate([1, 0, 0]).scale(3.0)
        assert dm2.n_boundaries == 0
        expected = (
            no_boundary_domain.interior.points + torch.tensor([1.0, 0.0, 0.0])
        ) * 3.0
        assert torch.allclose(dm2.interior.points, expected)
