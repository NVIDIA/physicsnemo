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

"""Tests for boundary detection functions."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.boundaries import (
    get_boundary_cells,
    get_boundary_edges,
    get_boundary_vertices,
)


class TestBoundaryVertices:
    """Tests for get_boundary_vertices."""

    def test_closed_surface_no_boundaries(self, device):
        """Closed surfaces (watertight) should have no boundary vertices."""
        # Tetrahedron (closed 2D surface)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_vertices(mesh)

        assert is_boundary.shape == (4,)
        assert not is_boundary.any(), "Closed surface should have no boundary vertices"

    def test_single_triangle_all_boundaries(self, device):
        """Single triangle should have all 3 vertices as boundary."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_vertices(mesh)

        assert is_boundary.shape == (3,)
        assert is_boundary.all(), "All vertices of single triangle are on boundary"

    def test_two_triangles_shared_edge(self, device):
        """Two triangles sharing an edge."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_vertices(mesh)

        # All 4 vertices are on boundary (edge [1,2] is interior, others are boundary)
        assert is_boundary.all(), "All vertices touch boundary edges"

    def test_cylinder_boundaries(self, device):
        """Cylinder should have boundary vertices on top and bottom circles."""
        # Simple cylinder: 2 circles (top and bottom) with 8 vertices each
        n_circ = 8
        n_height = 4

        theta = torch.linspace(0, 2 * torch.pi, n_circ + 1, device=device)[:-1]
        z_vals = torch.linspace(-1.0, 1.0, n_height, device=device)

        # Vectorized cylinder point generation: (n_height, n_circ) grid
        z_grid, theta_grid = torch.meshgrid(z_vals, theta, indexing="ij")
        points = torch.stack(
            [theta_grid.cos(), theta_grid.sin(), z_grid], dim=-1
        ).reshape(-1, 3)

        # Vectorized cell generation for cylinder (wrapping around circumference)
        i_idx, j_idx = torch.meshgrid(
            torch.arange(n_height - 1, device=device),
            torch.arange(n_circ, device=device),
            indexing="ij",
        )
        i_idx, j_idx = i_idx.reshape(-1), j_idx.reshape(-1)
        j_next = (j_idx + 1) % n_circ  # Wrap around for cylinder
        # Vertex indices for quad corners
        v0 = i_idx * n_circ + j_idx
        v1 = i_idx * n_circ + j_next
        v2 = (i_idx + 1) * n_circ + j_idx
        v3 = (i_idx + 1) * n_circ + j_next
        # Two triangles per quad
        tri1 = torch.stack([v0, v1, v2], dim=-1)
        tri2 = torch.stack([v1, v3, v2], dim=-1)
        cells = torch.cat([tri1, tri2], dim=0).to(torch.int64)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_vertices(mesh)

        # Top and bottom circles (z=±1) are boundaries
        expected_n_boundary = 2 * n_circ  # 16 vertices
        assert is_boundary.sum() == expected_n_boundary

        # Verify boundary vertices are at z=±1
        boundary_points = mesh.points[is_boundary]
        z_coords = boundary_points[:, 2]
        assert torch.allclose(z_coords.abs(), torch.ones_like(z_coords), atol=1e-5), (
            "Boundary vertices should be at z=±1"
        )

    def test_empty_mesh(self, device):
        """Empty mesh should have no boundary vertices."""
        points = torch.zeros((0, 3), device=device)
        cells = torch.zeros((0, 3), dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_vertices(mesh)

        assert is_boundary.shape == (0,)


class TestBoundaryEdges:
    """Tests for get_boundary_edges."""

    def test_single_triangle(self, device):
        """Single triangle has 3 boundary edges."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        boundary_edges = get_boundary_edges(mesh)

        assert len(boundary_edges) == 3, "Single triangle has 3 boundary edges"

    def test_closed_surface_no_boundary_edges(self, device):
        """Closed surface has no boundary edges."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        boundary_edges = get_boundary_edges(mesh)

        assert len(boundary_edges) == 0, "Closed surface has no boundary edges"

    def test_boundary_edges_connectivity(self, device):
        """Boundary edges should form proper connectivity."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        boundary_edges = get_boundary_edges(mesh)

        # Should have 4 boundary edges forming a square
        assert len(boundary_edges) == 4


class TestBoundaryCells:
    """Tests for get_boundary_cells."""

    def test_single_triangle_is_boundary(self, device):
        """Single triangle is a boundary cell."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_cells(mesh, boundary_codimension=1)

        assert is_boundary.shape == (1,)
        assert is_boundary.all(), "Single triangle is on boundary"

    def test_closed_surface_no_boundary_cells(self, device):
        """Closed surface has no boundary cells."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_cells(mesh, boundary_codimension=1)

        assert is_boundary.shape == (4,)
        assert not is_boundary.any(), "Closed surface has no boundary cells"

    def test_two_triangles_both_boundaries(self, device):
        """Two triangles sharing edge - both are boundary cells."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_cells(mesh, boundary_codimension=1)

        assert is_boundary.all(), "Both triangles have boundary edges"

    def test_boundary_codimension_validation(self, device):
        """Invalid boundary_codimension should raise error."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=device)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # manifold_dims=2, so valid codimensions are 1, 2
        with pytest.raises(ValueError, match="Invalid boundary_codimension"):
            get_boundary_cells(mesh, boundary_codimension=0)

        with pytest.raises(ValueError, match="Invalid boundary_codimension"):
            get_boundary_cells(mesh, boundary_codimension=3)

    def test_tetrahedra_boundary_cells(self, device):
        """Test boundary detection for 3D tetrahedra."""
        # Two tets sharing a triangular face
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 4]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        # With boundary_codimension=1: cells with boundary triangular faces
        is_boundary_1 = get_boundary_cells(mesh, boundary_codimension=1)
        assert is_boundary_1.all(), "Both tets have boundary faces"

        # With boundary_codimension=2: cells with boundary edges
        is_boundary_2 = get_boundary_cells(mesh, boundary_codimension=2)
        assert is_boundary_2.all(), "Both tets have boundary edges"

    def test_empty_mesh(self, device):
        """Empty mesh should have no boundary cells."""
        points = torch.zeros((0, 3), device=device)
        cells = torch.zeros((0, 3), dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary = get_boundary_cells(mesh)

        assert is_boundary.shape == (0,)


class TestBoundaryConsistency:
    """Tests for consistency between boundary detection functions."""

    def test_boundary_vertices_match_boundary_edges(self, device):
        """Vertices marked as boundary should be incident to boundary edges."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary_vertex = get_boundary_vertices(mesh)
        boundary_edges = get_boundary_edges(mesh)

        # All boundary edge vertices should be marked as boundary
        boundary_verts_from_edges = torch.unique(boundary_edges.flatten())
        is_boundary_from_edges = torch.zeros(
            mesh.n_points, dtype=torch.bool, device=device
        )
        is_boundary_from_edges[boundary_verts_from_edges] = True

        assert torch.equal(is_boundary_vertex, is_boundary_from_edges), (
            "Boundary vertices should match boundary edge endpoints"
        )

    def test_boundary_cells_contain_boundary_vertices(self, device):
        """Boundary cells should contain at least one boundary vertex."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        is_boundary_vertex = get_boundary_vertices(mesh)
        is_boundary_cell = get_boundary_cells(mesh, boundary_codimension=1)

        # All boundary cells should contain at least one boundary vertex
        for cell_idx in torch.where(is_boundary_cell)[0]:
            cell_vertices = mesh.cells[cell_idx]
            assert is_boundary_vertex[cell_vertices].any(), (
                f"Boundary cell {cell_idx} should contain at least one boundary vertex"
            )
