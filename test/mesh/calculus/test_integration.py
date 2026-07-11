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

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import integrate_cell_data, integrate_point_data
from physicsnemo.mesh.calculus.integration import (
    integrate,
    integrate_flux,
    integrate_moment,
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
# Cell/point field integration: shared behavior
###############################################################################


def _entity_count(mesh: Mesh, data_source: str) -> int:
    """Number of entities carrying the field for a given data source."""
    return mesh.n_cells if data_source == "cells" else mesh.n_points


@pytest.mark.parametrize("data_source", ["cells", "points"])
class TestIntegrateFields:
    """Behavior shared by cell-centered (P0) and point-centered (P1) quadrature."""

    @pytest.mark.parametrize(
        "mesh_fixture",
        ["unit_triangle", "two_triangles", "unit_tet", "edge_mesh"],
    )
    def test_constant_scalar(
        self, request: pytest.FixtureRequest, mesh_fixture: str, data_source: str
    ):
        """Integral of constant c over any manifold = c * total measure."""
        mesh = request.getfixturevalue(mesh_fixture)
        f = torch.full((_entity_count(mesh, data_source),), 7.0)
        result = integrate(mesh, f, data_source=data_source)
        assert torch.isclose(result, 7.0 * mesh.cell_areas.sum())

    def test_vector_field(self, unit_triangle: Mesh, data_source: str):
        """Trailing dimensions are preserved."""
        n = _entity_count(unit_triangle, data_source)
        f = torch.tensor([1.0, 2.0, 3.0]).expand(n, 3)
        result = integrate(unit_triangle, f, data_source=data_source)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([0.5, 1.0, 1.5]))

    def test_tensor_field(self, unit_triangle: Mesh, data_source: str):
        """Constant 2x2 tensor field: integral = area * value."""
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        n = _entity_count(unit_triangle, data_source)
        f = base.expand(n, 2, 2)
        result = integrate(unit_triangle, f, data_source=data_source)
        assert result.shape == (2, 2)
        assert torch.allclose(result, 0.5 * base)

    def test_via_mesh_method(self, two_triangles: Mesh, data_source: str):
        """Mesh.integrate() forwards to the functional entry point."""
        data = (
            two_triangles.cell_data
            if data_source == "cells"
            else two_triangles.point_data
        )
        data["f"] = torch.full((_entity_count(two_triangles, data_source),), 2.0)
        result = two_triangles.integrate("f", data_source=data_source)
        assert torch.isclose(result, 2.0 * two_triangles.cell_areas.sum())


###############################################################################
# Cell data integration (P0-specific)
###############################################################################


class TestIntegrateCellFields:
    def test_two_cells(self, two_triangles: Mesh):
        """P0 quadrature is exact for distinct piecewise-constant values."""
        areas = two_triangles.cell_areas
        f = torch.tensor([2.0, 5.0])
        expected = (f * areas).sum()
        assert torch.isclose(integrate(two_triangles, f, data_source="cells"), expected)


###############################################################################
# Point data integration (P1-specific)
###############################################################################


class TestIntegratePointFields:
    def test_linear_field_exact(self, unit_triangle: Mesh):
        """P1 integral of linear field f(x,y)=x is exact.

        Vertex values at (0,0),(1,0),(0,1): x = [0, 1, 0].
        Analytic: integral of x over right triangle = area * x_centroid
                = 0.5 * (1/3) = 1/6.
        P1: 0.5 * mean(0, 1, 0) = 0.5 * 1/3 = 1/6.  Exact.
        """
        f = unit_triangle.points[:, 0]  # f = x coordinate
        result = integrate(unit_triangle, f, data_source="points")
        expected = torch.tensor(1.0 / 6.0)
        assert torch.isclose(result, expected)

    def test_linear_field_y(self, unit_triangle: Mesh):
        """P1 integral of f(x,y) = y.  Analytic = 1/6."""
        f = unit_triangle.points[:, 1]
        result = integrate(unit_triangle, f, data_source="points")
        assert torch.isclose(result, torch.tensor(1.0 / 6.0))

    def test_linear_vector_field(self, unit_triangle: Mesh):
        """Linear vector field is integrated exactly per component."""
        f = torch.stack(
            [unit_triangle.points[:, 0], unit_triangle.points[:, 1]], dim=-1
        )  # (3, 2)
        result = integrate(unit_triangle, f, data_source="points")
        assert result.shape == (2,)
        expected = torch.tensor([1.0 / 6.0, 1.0 / 6.0])
        assert torch.allclose(result, expected)

    def test_tet_linear(self, unit_tet: Mesh):
        """Linear field f(x,y,z) = x on tetrahedron.

        Vertex x-coords: [0, 1, 0, 0]. Mean = 0.25.
        Integral = (1/6) * 0.25 = 1/24.
        """
        f = unit_tet.points[:, 0]
        result = integrate(unit_tet, f, data_source="points")
        assert torch.isclose(result, torch.tensor(1.0 / 24.0))

    def test_edge_linear(self, edge_mesh: Mesh):
        """Linear field on edges: f(x) = x, x in [0,3].

        Each edge has length 1. For edge [i, i+1]: mean = i + 0.5.
        Integral = 1*(0.5) + 1*(1.5) + 1*(2.5) = 4.5.
        Analytic: integral of x from 0 to 3 = 9/2 = 4.5.
        """
        f = edge_mesh.points[:, 0]  # [0, 1, 2, 3]
        result = integrate(edge_mesh, f, data_source="points")
        assert torch.isclose(result, torch.tensor(4.5))


###############################################################################
# Deprecated compatibility entry points
###############################################################################


class TestDeprecatedIntegrateAliases:
    @pytest.mark.parametrize(
        "alias, data_source",
        [(integrate_cell_data, "cells"), (integrate_point_data, "points")],
    )
    def test_alias_warns_and_matches_integrate(
        self, unit_triangle: Mesh, alias, data_source: str
    ):
        field = torch.arange(1.0, 1.0 + _entity_count(unit_triangle, data_source))
        with pytest.warns(LegacyFeatureWarning, match=alias.__name__):
            result = alias(unit_triangle, field)
        expected = integrate(unit_triangle, field, data_source=data_source)
        torch.testing.assert_close(result, expected)


###############################################################################
# NaN handling
###############################################################################


class TestNaNHandling:
    def test_cell_nan_excluded(self, two_triangles: Mesh):
        """Cells with NaN values are excluded from the integral."""
        f = torch.tensor([2.0, float("nan")])
        result = integrate(two_triangles, f, data_source="cells")
        expected = 2.0 * two_triangles.cell_areas[0]
        assert torch.isclose(result, expected)

    def test_point_nan_skips_affected_cells(self, two_triangles: Mesh):
        """P1: if any vertex of a cell is NaN, that cell contributes nothing.

        Vertex 1 is shared by both cells, so both cells are affected.
        """
        f = torch.tensor([1.0, float("nan"), 1.0, 1.0])
        result = integrate(two_triangles, f, data_source="points")
        assert torch.isclose(result, torch.tensor(0.0))

    def test_cell_nan_vector_field(self, two_triangles: Mesh):
        """NaN in one component of a vector field propagates for that cell."""
        f = torch.tensor([[1.0, 2.0], [float("nan"), 3.0]])
        result = integrate(two_triangles, f, data_source="cells")
        areas = two_triangles.cell_areas
        # Component 0: only cell 0 contributes
        assert torch.isclose(result[0], 1.0 * areas[0])
        # Component 1: both cells contribute
        expected_1 = 2.0 * areas[0] + 3.0 * areas[1]
        assert torch.isclose(result[1], expected_1)

    @pytest.mark.parametrize("data_source", ["cells", "points"])
    def test_nan_propagated(self, two_triangles: Mesh, data_source: str):
        f = torch.ones(_entity_count(two_triangles, data_source))
        f[1] = float("nan")
        result = integrate(
            two_triangles,
            f,
            data_source=data_source,
            nan_policy="propagate",
        )
        assert torch.isnan(result)

    def test_nan_policy_via_mesh_method(self, two_triangles: Mesh):
        two_triangles.cell_data["p"] = torch.tensor([2.0, float("nan")])
        assert torch.isnan(two_triangles.integrate("p", nan_policy="propagate"))

    def test_invalid_nan_policy(self, unit_triangle: Mesh):
        with pytest.raises(ValueError, match="nan_policy"):
            integrate(
                unit_triangle,
                torch.ones(1),
                data_source="cells",
                nan_policy="invalid",  # type: ignore[arg-type]
            )


###############################################################################
# Cell outer-product moments
###############################################################################


class TestIntegrateMoment:
    def test_matches_explicit_outer_product(self, two_triangles: Mesh):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        result = integrate_moment(two_triangles, left, right)
        explicit = torch.einsum(
            "n,ni,njk->ijk",
            two_triangles.cell_areas,
            left,
            right,
        )

        assert result.shape == (2, 2, 2)
        assert torch.allclose(result, explicit)

    def test_scalar_fields_return_scalar(self, two_triangles: Mesh):
        left = torch.tensor([2.0, 3.0])
        right = torch.tensor([5.0, 7.0])
        result = integrate_moment(two_triangles, left, right)
        expected = (two_triangles.cell_areas * left * right).sum()
        assert result.shape == torch.Size([])
        assert torch.allclose(result, expected)

    def test_named_fields(self, two_triangles: Mesh):
        two_triangles.cell_data["left"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        two_triangles.cell_data["right"] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = integrate_moment(two_triangles, "left", "right")
        expected = torch.einsum(
            "n,ni,nj->ij",
            two_triangles.cell_areas,
            two_triangles.cell_data["left"],
            two_triangles.cell_data["right"],
        )
        assert torch.allclose(result, expected)

    def test_via_mesh_method(self, two_triangles: Mesh):
        """Mesh.integrate_moment() forwards all arguments to the functional API."""
        two_triangles.cell_data["left"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        two_triangles.cell_data["right"] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = two_triangles.integrate_moment(
            "left",
            "right",
            aligned_dims=1,
            accumulation_dtype=torch.float64,
            nan_policy="propagate",
        )
        expected = integrate_moment(
            two_triangles,
            "left",
            "right",
            aligned_dims=1,
            accumulation_dtype=torch.float64,
            nan_policy="propagate",
        )
        assert result.dtype == torch.float64
        torch.testing.assert_close(result, expected)

    def test_aligned_group_dimensions(self, two_triangles: Mesh):
        left = torch.arange(24.0).reshape(2, 3, 4)
        right = torch.arange(30.0).reshape(2, 3, 5)

        result = integrate_moment(
            two_triangles,
            left,
            right,
            aligned_dims=1,
        )
        expected = torch.einsum(
            "n,nha,nhb->hab",
            two_triangles.cell_areas,
            left,
            right,
        )

        assert result.shape == (3, 4, 5)
        assert torch.allclose(result, expected)

    def test_aligned_dimensions_must_match(self, two_triangles: Mesh):
        with pytest.raises(ValueError, match="Aligned field dimensions"):
            integrate_moment(
                two_triangles,
                torch.ones(2, 3, 4),
                torch.ones(2, 2, 5),
                aligned_dims=1,
            )

    @pytest.mark.parametrize("aligned_dims", [-1, 2])
    def test_invalid_aligned_dimension_count(
        self, two_triangles: Mesh, aligned_dims: int
    ):
        with pytest.raises(ValueError, match="aligned_dims"):
            integrate_moment(
                two_triangles,
                torch.ones(2, 3),
                torch.ones(2, 3),
                aligned_dims=aligned_dims,
            )

    def test_default_accumulates_reduced_precision_in_fp32(self, two_triangles: Mesh):
        left = torch.ones((two_triangles.n_cells, 2), dtype=torch.float16)
        right = torch.ones((two_triangles.n_cells, 3), dtype=torch.float16)
        result = integrate_moment(two_triangles, left, right)
        assert result.dtype == torch.float32

    def test_configurable_accumulation_dtype(self, two_triangles: Mesh):
        left = torch.ones((two_triangles.n_cells, 2), dtype=torch.float32)
        right = torch.ones((two_triangles.n_cells, 3), dtype=torch.float32)
        result = integrate_moment(
            two_triangles,
            left,
            right,
            accumulation_dtype=torch.float64,
        )
        assert result.dtype == torch.float64

    def test_fp64_inputs_are_not_downcast(self, two_triangles: Mesh):
        left = torch.ones((two_triangles.n_cells, 2), dtype=torch.float64)
        right = torch.ones((two_triangles.n_cells, 3), dtype=torch.float64)
        result = integrate_moment(two_triangles, left, right)
        assert result.dtype == torch.float64

    def test_nan_policy(self, two_triangles: Mesh):
        left = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])
        right = torch.tensor([[4.0], [5.0]])

        omitted = integrate_moment(
            two_triangles,
            left,
            right,
            nan_policy="omit",
        )
        propagated = integrate_moment(
            two_triangles,
            left,
            right,
            nan_policy="propagate",
        )

        expected_omitted = torch.einsum(
            "n,ni,nj->ij",
            two_triangles.cell_areas,
            torch.nan_to_num(left, nan=0.0),
            right,
        )
        assert torch.allclose(omitted, expected_omitted)
        assert torch.isnan(propagated[1, 0])

    def test_invalid_nan_policy(self, two_triangles: Mesh):
        with pytest.raises(ValueError, match="nan_policy"):
            integrate_moment(
                two_triangles,
                torch.ones(two_triangles.n_cells, 2),
                torch.ones(two_triangles.n_cells, 3),
                nan_policy="invalid",  # type: ignore[arg-type]
            )

    def test_gradients(self, two_triangles: Mesh):
        left = torch.randn(two_triangles.n_cells, 2, requires_grad=True)
        right = torch.randn(two_triangles.n_cells, 3, requires_grad=True)
        result = integrate_moment(two_triangles, left, right)
        result.square().sum().backward()
        assert left.grad is not None and torch.isfinite(left.grad).all()
        assert right.grad is not None and torch.isfinite(right.grad).all()

    def test_wrong_leading_dimension(self, unit_triangle: Mesh):
        with pytest.raises(ValueError, match="left.*n_cells"):
            integrate_moment(
                unit_triangle,
                torch.ones(2, 3),
                torch.ones(1, 4),
            )

    def test_empty_mesh_raises(self):
        point_cloud = Mesh(points=torch.randn(3, 2))
        with pytest.raises(ValueError, match="no cells"):
            integrate_moment(
                point_cloud,
                torch.empty(0, 2),
                torch.empty(0, 3),
            )


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

    def test_nan_policy_and_mesh_forwarding(self, triangle_3d: Mesh):
        field = torch.full((triangle_3d.n_cells, 3), float("nan"))
        triangle_3d.cell_data["v"] = field

        omitted = integrate_flux(
            triangle_3d,
            field,
            nan_policy="omit",
        )
        propagated = triangle_3d.integrate_flux(
            "v",
            nan_policy="propagate",
        )

        assert torch.equal(omitted, torch.zeros_like(omitted))
        assert torch.isnan(propagated)

    def test_invalid_nan_policy(self, triangle_3d: Mesh):
        with pytest.raises(ValueError, match="nan_policy"):
            integrate_flux(
                triangle_3d,
                torch.ones(triangle_3d.n_cells, 3),
                nan_policy="invalid",  # type: ignore[arg-type]
            )


###############################################################################
# Top-level integrate() dispatch
###############################################################################


class TestIntegrateDispatch:
    @pytest.mark.parametrize("data_source", ["cells", "points"])
    def test_string_key(self, two_triangles: Mesh, data_source: str):
        """String keys are resolved from the data dictionary for the source."""
        data = (
            two_triangles.cell_data
            if data_source == "cells"
            else two_triangles.point_data
        )
        data["f"] = torch.full((_entity_count(two_triangles, data_source),), 4.0)
        result = integrate(two_triangles, "f", data_source=data_source)
        assert torch.isclose(result, 4.0 * two_triangles.cell_areas.sum())

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

    @pytest.mark.parametrize(
        "data_source, match",
        [("cells", "cell_data"), ("points", "point_data")],
    )
    def test_missing_key(self, unit_triangle: Mesh, data_source: str, match: str):
        """String key not in the data dictionary gives a helpful KeyError."""
        with pytest.raises(KeyError, match=match):
            integrate(unit_triangle, "nonexistent", data_source=data_source)

    @pytest.mark.parametrize(
        "data_source, match",
        [("cells", "n_cells"), ("points", "n_points")],
    )
    def test_wrong_tensor_shape(
        self, unit_triangle: Mesh, data_source: str, match: str
    ):
        """Tensor with wrong leading dimension raises ValueError."""
        wrong = torch.ones(_entity_count(unit_triangle, data_source) + 5)
        with pytest.raises(ValueError, match=match):
            integrate(unit_triangle, wrong, data_source=data_source)


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
