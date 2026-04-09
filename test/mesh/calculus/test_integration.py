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

"""Tests for mesh integration (quadrature) operators.

Tests numerical integration of scalar, vector, and tensor fields over
simplicial meshes using cell-centered (P0) and vertex-centered (P1)
quadrature rules.
"""

import math

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.integration import (
    integrate,
    integrate_cell_data,
    integrate_flux,
    integrate_point_data,
)

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def unit_triangle() -> Mesh:
    """Right triangle with vertices (0,0), (1,0), (0,1).  Area = 0.5."""
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = torch.tensor([[0, 1, 2]])
    return Mesh(points=pts, cells=cells)


@pytest.fixture
def two_triangles() -> Mesh:
    """Two triangles forming a quadrilateral in 2D."""
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]])
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    return Mesh(points=pts, cells=cells)


@pytest.fixture
def unit_tet() -> Mesh:
    """Regular-ish tetrahedron.  Volume = 1/6."""
    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    cells = torch.tensor([[0, 1, 2, 3]])
    return Mesh(points=pts, cells=cells)


@pytest.fixture
def edge_mesh() -> Mesh:
    """Three edges in 2D (1-manifold)."""
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    cells = torch.tensor([[0, 1], [1, 2], [2, 3]])
    return Mesh(points=pts, cells=cells)


@pytest.fixture
def triangle_3d() -> Mesh:
    """Single triangle in 3D (codimension-1, has normals)."""
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells = torch.tensor([[0, 1, 2]])
    return Mesh(points=pts, cells=cells)


###############################################################################
# Cell data integration
###############################################################################


class TestIntegrateCellData:
    def test_constant_scalar(self, unit_triangle: Mesh):
        """Integral of constant c over domain = c * volume."""
        f = torch.tensor([7.0])
        result = integrate_cell_data(unit_triangle, f)
        assert torch.isclose(result, torch.tensor(7.0 * 0.5))

    def test_two_cells(self, two_triangles: Mesh):
        areas = two_triangles.cell_areas
        f = torch.tensor([2.0, 5.0])
        expected = (f * areas).sum()
        assert torch.isclose(integrate_cell_data(two_triangles, f), expected)

    def test_via_mesh_method(self, two_triangles: Mesh):
        two_triangles.cell_data["p"] = torch.tensor([2.0, 5.0])
        areas = two_triangles.cell_areas
        expected = (torch.tensor([2.0, 5.0]) * areas).sum()
        assert torch.isclose(two_triangles.integrate("p"), expected)

    def test_vector_field(self, unit_triangle: Mesh):
        """Trailing dimensions are preserved."""
        f = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        result = integrate_cell_data(unit_triangle, f)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.5, 1.0, 1.5]))

    def test_tensor_field(self, unit_triangle: Mesh):
        """2x2 tensor field on a single cell."""
        f = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        result = integrate_cell_data(unit_triangle, f)
        assert result.shape == (2, 2)
        expected = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
        assert torch.allclose(result, expected)


###############################################################################
# Point data integration (P1)
###############################################################################


class TestIntegratePointData:
    def test_constant_field_exact(self, unit_triangle: Mesh):
        """P1 integral of constant field = constant * volume."""
        f = torch.tensor([3.0, 3.0, 3.0])
        result = integrate_point_data(unit_triangle, f)
        assert torch.isclose(result, torch.tensor(3.0 * 0.5))

    def test_linear_field_exact(self, unit_triangle: Mesh):
        """P1 integral of linear field f(x,y)=x is exact.

        Vertex values at (0,0),(1,0),(0,1): x = [0, 1, 0].
        Analytic: integral of x over right triangle = area * x_centroid
                = 0.5 * (1/3) = 1/6.
        P1: 0.5 * mean(0, 1, 0) = 0.5 * 1/3 = 1/6.  Exact.
        """
        f = unit_triangle.points[:, 0]  # f = x coordinate
        result = integrate_point_data(unit_triangle, f)
        expected = torch.tensor(1.0 / 6.0)
        assert torch.isclose(result, expected)

    def test_linear_field_y(self, unit_triangle: Mesh):
        """P1 integral of f(x,y) = y.  Analytic = 1/6."""
        f = unit_triangle.points[:, 1]
        result = integrate_point_data(unit_triangle, f)
        assert torch.isclose(result, torch.tensor(1.0 / 6.0))

    def test_multiple_cells(self, two_triangles: Mesh):
        """Integration over mesh with two cells."""
        f = torch.ones(two_triangles.n_points)
        result = integrate_point_data(two_triangles, f)
        assert torch.isclose(result, two_triangles.cell_areas.sum())

    def test_vector_field(self, unit_triangle: Mesh):
        """Vector field preserves trailing dimension."""
        f = torch.stack(
            [unit_triangle.points[:, 0], unit_triangle.points[:, 1]], dim=-1
        )  # (3, 2)
        result = integrate_point_data(unit_triangle, f)
        assert result.shape == (2,)
        expected = torch.tensor([1.0 / 6.0, 1.0 / 6.0])
        assert torch.allclose(result, expected)

    def test_tet_constant(self, unit_tet: Mesh):
        """Constant field on tetrahedron: integral = c * V."""
        f = torch.full((4,), 5.0)
        result = integrate_point_data(unit_tet, f)
        expected = 5.0 / 6.0
        assert torch.isclose(result, torch.tensor(expected))

    def test_tet_linear(self, unit_tet: Mesh):
        """Linear field f(x,y,z) = x on tetrahedron.

        Vertex x-coords: [0, 1, 0, 0]. Mean = 0.25.
        Integral = (1/6) * 0.25 = 1/24.
        """
        f = unit_tet.points[:, 0]
        result = integrate_point_data(unit_tet, f)
        assert torch.isclose(result, torch.tensor(1.0 / 24.0))

    def test_edge_constant(self, edge_mesh: Mesh):
        """Constant field on edges: integral = c * total_length."""
        f = torch.full((4,), 2.0)
        result = integrate_point_data(edge_mesh, f)
        assert torch.isclose(result, torch.tensor(2.0 * 3.0))

    def test_edge_linear(self, edge_mesh: Mesh):
        """Linear field on edges: f(x) = x, x in [0,3].

        Each edge has length 1. For edge [i, i+1]: mean = i + 0.5.
        Integral = 1*(0.5) + 1*(1.5) + 1*(2.5) = 4.5.
        Analytic: integral of x from 0 to 3 = 9/2 = 4.5.
        """
        f = edge_mesh.points[:, 0]  # [0, 1, 2, 3]
        result = integrate_point_data(edge_mesh, f)
        assert torch.isclose(result, torch.tensor(4.5))

    def test_via_mesh_method(self, unit_triangle: Mesh):
        unit_triangle.point_data["T"] = unit_triangle.points[:, 0]
        result = unit_triangle.integrate("T", data_source="points")
        assert torch.isclose(result, torch.tensor(1.0 / 6.0))


###############################################################################
# NaN handling
###############################################################################


class TestNaNHandling:
    def test_cell_nan_excluded(self, two_triangles: Mesh):
        """Cells with NaN values are excluded from the integral."""
        f = torch.tensor([2.0, float("nan")])
        result = integrate_cell_data(two_triangles, f)
        expected = 2.0 * two_triangles.cell_areas[0]
        assert torch.isclose(result, expected)

    def test_point_nan_skips_affected_cells(self, two_triangles: Mesh):
        """P1: if any vertex of a cell is NaN, that cell contributes nothing.

        Vertex 1 is shared by both cells, so both cells are affected.
        """
        f = torch.tensor([1.0, float("nan"), 1.0, 1.0])
        result = integrate_point_data(two_triangles, f)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_cell_nan_vector_field(self, two_triangles: Mesh):
        """NaN in one component of a vector field propagates for that cell."""
        f = torch.tensor([[1.0, 2.0], [float("nan"), 3.0]])
        result = integrate_cell_data(two_triangles, f)
        areas = two_triangles.cell_areas
        # Component 0: only cell 0 contributes
        assert torch.isclose(result[0], 1.0 * areas[0])
        # Component 1: both cells contribute
        expected_1 = 2.0 * areas[0] + 3.0 * areas[1]
        assert torch.isclose(result[1], expected_1)


###############################################################################
# Flux integration
###############################################################################


class TestIntegrateFlux:
    def test_closed_surface_constant_field(self):
        """Divergence theorem: flux of constant field through closed surface = 0."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        sphere = sphere_icosahedral.load(subdivisions=2)
        v = torch.ones(sphere.n_cells, 3)
        flux = integrate_flux(sphere, v, data_source="cells")
        assert torch.abs(flux) < 1e-5

    def test_closed_surface_point_data(self):
        """Flux of constant point field through closed surface = 0."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        sphere = sphere_icosahedral.load(subdivisions=2)
        v = torch.ones(sphere.n_points, 3)
        flux = integrate_flux(sphere, v, data_source="points")
        assert torch.abs(flux) < 1e-5

    def test_single_triangle_3d(self, triangle_3d: Mesh):
        """Flux of a field normal to a single triangle in 3D."""
        # Triangle in xy-plane, normal is +z or -z
        normal = triangle_3d.cell_normals[0]
        area = triangle_3d.cell_areas[0]

        # Field aligned with normal -> flux = |field| * area
        field = normal.unsqueeze(0)  # (1, 3)
        flux = integrate_flux(triangle_3d, field, data_source="cells")
        assert torch.isclose(flux, area)

    def test_codimension_check(self, unit_triangle: Mesh):
        """integrate_flux rejects non-codimension-1 meshes."""
        f = torch.zeros(unit_triangle.n_cells, 2)
        with pytest.raises(ValueError, match="codimension-1"):
            integrate_flux(unit_triangle, f)

    def test_dimension_check(self, triangle_3d: Mesh):
        """Field dimension must match spatial dims."""
        f = torch.zeros(triangle_3d.n_cells, 2)  # should be 3
        with pytest.raises(ValueError, match="last dimension"):
            integrate_flux(triangle_3d, f)

    def test_via_mesh_method(self, triangle_3d: Mesh):
        normal = triangle_3d.cell_normals[0]
        triangle_3d.cell_data["v"] = normal.unsqueeze(0)
        flux = triangle_3d.integrate_flux("v")
        assert torch.isclose(flux, triangle_3d.cell_areas[0])


###############################################################################
# Top-level integrate() dispatch
###############################################################################


class TestIntegrateDispatch:
    def test_string_key_cell(self, two_triangles: Mesh):
        two_triangles.cell_data["p"] = torch.tensor([1.0, 2.0])
        result = integrate(two_triangles, "p", data_source="cells")
        areas = two_triangles.cell_areas
        assert torch.isclose(result, (torch.tensor([1.0, 2.0]) * areas).sum())

    def test_string_key_point(self, unit_triangle: Mesh):
        unit_triangle.point_data["T"] = torch.ones(3) * 4.0
        result = integrate(unit_triangle, "T", data_source="points")
        assert torch.isclose(result, torch.tensor(4.0 * 0.5))

    def test_tensor_direct(self, unit_triangle: Mesh):
        f = torch.tensor([6.0])
        result = integrate(unit_triangle, f, data_source="cells")
        assert torch.isclose(result, torch.tensor(3.0))

    def test_point_cloud_raises(self):
        """Integration over a point cloud (no cells) is undefined."""
        pc = Mesh(points=torch.randn(10, 3))
        with pytest.raises(ValueError, match="no cells"):
            integrate(pc, torch.ones(10))

    def test_invalid_data_source(self, unit_triangle: Mesh):
        with pytest.raises(ValueError, match="data_source"):
            integrate(unit_triangle, torch.ones(3), data_source="invalid")

    def test_missing_cell_key(self, unit_triangle: Mesh):
        """String key not in cell_data gives a helpful KeyError."""
        with pytest.raises(KeyError, match="cell.*_data"):
            integrate(unit_triangle, "nonexistent")

    def test_missing_point_key(self, unit_triangle: Mesh):
        """String key not in point_data gives a helpful KeyError."""
        with pytest.raises(KeyError, match="point.*_data"):
            integrate(unit_triangle, "nonexistent", data_source="points")

    def test_wrong_cell_tensor_shape(self, unit_triangle: Mesh):
        """Tensor with wrong leading dimension for cell data raises ValueError."""
        wrong = torch.ones(unit_triangle.n_cells + 5)
        with pytest.raises(ValueError, match="n_cells"):
            integrate(unit_triangle, wrong, data_source="cells")

    def test_wrong_point_tensor_shape(self, unit_triangle: Mesh):
        """Tensor with wrong leading dimension for point data raises ValueError."""
        wrong = torch.ones(unit_triangle.n_points + 5)
        with pytest.raises(ValueError, match="n_points"):
            integrate(unit_triangle, wrong, data_source="points")


###############################################################################
# Consistency checks
###############################################################################


class TestConsistency:
    def test_p1_equals_cell_data_to_point_data_pipeline(self, two_triangles: Mesh):
        """P1 integration should match: convert to cell data, then integrate.

        This validates the claim that P1 integration is equivalent to
        point_data_to_cell_data() followed by cell integration.
        """
        f = torch.randn(two_triangles.n_points)
        two_triangles.point_data["f"] = f
        p1_result = integrate(two_triangles, "f", data_source="points")

        converted = two_triangles.point_data_to_cell_data()
        cell_result = integrate(converted, "f", data_source="cells")

        assert torch.isclose(p1_result, cell_result)

    def test_sphere_area_convergence(self):
        """Icosahedral sphere area converges to 4*pi with subdivision."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        analytic = 4.0 * math.pi
        errors = []
        for subdiv in [1, 2, 3]:
            mesh = sphere_icosahedral.load(subdivisions=subdiv)
            area = mesh.cell_areas.sum().item()
            errors.append(abs(area - analytic) / analytic)

        # Error should decrease with refinement
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        # subdivision=3 should be within ~0.5% of analytic
        assert errors[2] < 0.005
