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

"""Tests for boundary mesh extraction.

Tests validate that boundary mesh extraction correctly identifies and extracts
only the facets that lie on the boundary of a mesh (appearing in exactly one cell).
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh


class TestBoundaryExtraction2D:
    """Test boundary extraction for 2D triangular meshes."""

    def test_single_triangle_boundary(self, device):
        """Single triangle has 3 boundary edges."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Single triangle has 3 boundary edges
        assert boundary.n_cells == 3
        assert boundary.n_manifold_dims == 1

        ### Boundary should contain all edges
        expected_edges = torch.tensor([[0, 1], [0, 2], [1, 2]], device=device)
        assert torch.all(
            torch.sort(boundary.cells, dim=-1)[0]
            == torch.sort(expected_edges, dim=-1)[0]
        )

    def test_two_triangles_shared_edge(self, device):
        """Two triangles sharing an edge have 4 boundary edges."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Should have 4 boundary edges (perimeter of quad)
        assert boundary.n_cells == 4

        ### Check that the shared edge [1, 2] is not in boundary
        boundary_sorted = torch.sort(boundary.cells, dim=-1)[0]
        shared_edge = torch.tensor([[1, 2]], device=device)
        matches = torch.all(boundary_sorted == shared_edge, dim=1)
        assert not torch.any(matches), "Shared edge should not be in boundary"

    def test_closed_2d_mesh_no_boundary(self, device):
        """Closed 2D mesh (all edges shared) has empty boundary."""
        ### Create a simple quad (4 triangles)
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Should have 4 boundary edges around the perimeter
        assert boundary.n_cells == 4


class TestBoundaryExtraction3D:
    """Test boundary extraction for 3D tetrahedral meshes."""

    def test_single_tetrahedron_boundary(self, device):
        """Single tetrahedron has 4 boundary triangular faces."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Single tet has 4 boundary triangular faces
        assert boundary.n_cells == 4
        assert boundary.n_manifold_dims == 2

        ### Check that all 4 faces are present
        expected_faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            device=device,
        )
        boundary_sorted = torch.sort(boundary.cells, dim=-1)[0]
        expected_sorted = torch.sort(expected_faces, dim=-1)[0]

        ### Check that boundary contains all expected faces
        for expected_face in expected_sorted:
            matches = torch.all(boundary_sorted == expected_face.unsqueeze(0), dim=1)
            assert torch.any(matches), f"Face {expected_face} should be in boundary"

    def test_two_tets_shared_face(self, device):
        """Two tets sharing a face have 6 boundary faces."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],  # Point on opposite side
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 4]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Two tets sharing face [0,1,2] have 6 boundary faces
        ### (4 faces from first tet + 4 from second - 2 shared = 6)
        assert boundary.n_cells == 6

        ### Check that the shared face [0, 1, 2] is not in boundary
        boundary_sorted = torch.sort(boundary.cells, dim=-1)[0]
        shared_face = torch.tensor([[0, 1, 2]], device=device)
        matches = torch.all(boundary_sorted == shared_face, dim=1)
        assert not torch.any(matches), "Shared face should not be in boundary"


class TestBoundaryExtraction1D:
    """Test boundary extraction for 1D edge meshes."""

    def test_single_edge_boundary(self, device):
        """Single edge has 2 boundary vertices."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        cells = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Single edge has 2 boundary vertices
        assert boundary.n_cells == 2
        assert boundary.n_manifold_dims == 0

    def test_chain_of_edges(self, device):
        """Chain of edges has 2 boundary vertices at ends."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1], [1, 2], [2, 3]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Chain has 2 boundary vertices (at ends: 0 and 3)
        assert boundary.n_cells == 2

        ### Check that boundary vertices are 0 and 3
        boundary_vertices = boundary.cells.flatten()
        assert torch.all(
            torch.sort(boundary_vertices)[0] == torch.tensor([0, 3], device=device)
        )

    def test_closed_loop_no_boundary(self, device):
        """Closed loop of edges has no boundary."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        ### Closed loop has no boundary
        assert boundary.n_cells == 0


class TestBoundaryDataInheritance:
    """Test that boundary mesh correctly inherits data from parent."""

    def test_boundary_inherits_cell_data(self, device):
        """Boundary mesh inherits data from parent cells."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)

        ### Add cell data
        cell_data = {"pressure": torch.tensor([1.0, 2.0], device=device)}
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        boundary = mesh.get_boundary_mesh(data_source="cells")

        ### Boundary should have cell_data
        assert "pressure" in boundary.cell_data.keys()
        assert len(boundary.cell_data["pressure"]) == boundary.n_cells

    def test_boundary_inherits_point_data(self, device):
        """Boundary mesh can inherit data from points."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

        ### Add point data
        point_data = {"temperature": torch.tensor([10.0, 20.0, 15.0], device=device)}
        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        boundary = mesh.get_boundary_mesh(data_source="points")

        ### Boundary should have cell_data averaged from points
        assert "temperature" in boundary.cell_data.keys()
        assert len(boundary.cell_data["temperature"]) == boundary.n_cells


class TestBoundaryEmptyMesh:
    """Test boundary extraction on edge cases."""

    def test_empty_mesh(self, device):
        """Empty mesh has empty boundary."""
        points = torch.empty((0, 2), device=device)
        cells = torch.empty((0, 3), device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        boundary = mesh.get_boundary_mesh()

        assert boundary.n_cells == 0
        assert boundary.n_points == 0


class TestLumpyBallBoundary:
    """Test boundary extraction from lumpy_ball volumetric meshes."""

    @pytest.mark.parametrize("subdivisions", [0, 1, 2, 3])
    def test_boundary_cell_count(self, device, subdivisions):
        """Boundary of lumpy_ball has exactly n_faces = 20 * 4^subdivisions cells."""
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        mesh = lumpy_ball.load(subdivisions=subdivisions, device=device)
        boundary = mesh.get_boundary_mesh()

        expected_faces = 20 * (4**subdivisions)
        assert boundary.n_cells == expected_faces, (
            f"Expected {expected_faces} boundary faces for subdivisions={subdivisions}, "
            f"got {boundary.n_cells}"
        )

    @pytest.mark.parametrize("n_shells", [1, 2, 3])
    def test_boundary_independent_of_shells(self, device, n_shells):
        """Boundary cell count is independent of n_shells (only outer shell matters)."""
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        subdivisions = 1
        mesh = lumpy_ball.load(
            n_shells=n_shells, subdivisions=subdivisions, device=device
        )
        boundary = mesh.get_boundary_mesh()

        expected_faces = 20 * (4**subdivisions)
        assert boundary.n_cells == expected_faces

    def test_boundary_is_watertight(self, device):
        """Boundary surface of lumpy_ball is watertight (closed, no holes)."""
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        mesh = lumpy_ball.load(n_shells=2, subdivisions=2, device=device)
        boundary = mesh.get_boundary_mesh()

        assert boundary.is_watertight(), (
            "Boundary surface should be watertight (every edge shared by exactly 2 faces)"
        )

    def test_boundary_is_manifold(self, device):
        """Boundary surface of lumpy_ball is a valid 2D manifold."""
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        mesh = lumpy_ball.load(n_shells=2, subdivisions=2, device=device)
        boundary = mesh.get_boundary_mesh()

        assert boundary.is_manifold(), (
            "Boundary surface should be manifold (no T-junctions or non-manifold edges)"
        )

    def test_boundary_manifold_dims(self, device):
        """Boundary of 3D tetrahedral mesh is 2D triangular mesh."""
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        mesh = lumpy_ball.load(device=device)
        boundary = mesh.get_boundary_mesh()

        assert mesh.n_manifold_dims == 3, "lumpy_ball should be 3D (tetrahedra)"
        assert boundary.n_manifold_dims == 2, "Boundary should be 2D (triangles)"
        assert boundary.cells.shape[1] == 3, (
            "Boundary cells should have 3 vertices each"
        )

    @pytest.mark.parametrize("noise_amplitude", [0.0, 0.3, 0.5])
    def test_boundary_valid_with_noise(self, device, noise_amplitude):
        """Boundary remains well-formed regardless of noise amplitude."""
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        mesh = lumpy_ball.load(
            n_shells=2,
            noise_amplitude=noise_amplitude,
            seed=42,
            subdivisions=2,
            device=device,
        )
        boundary = mesh.get_boundary_mesh()

        # Cell count unaffected by noise (topology preserved)
        expected_faces = 20 * (4**2)
        assert boundary.n_cells == expected_faces

        # Topology preserved
        assert boundary.is_watertight()
        assert boundary.is_manifold()
