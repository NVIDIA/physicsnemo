# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for topology validation (watertight and manifold checking).

Tests validate that topology checking functions correctly identify watertight
meshes and topological manifolds.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh


class TestWatertight2D:
    """Test watertight checking for 2D meshes."""

    def test_single_triangle_not_watertight(self, device):
        """Single triangle is not watertight (has boundary edges)."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert not mesh.is_watertight()

    def test_two_triangles_not_watertight(self, device):
        """Two triangles with shared edge are not watertight (have boundary edges)."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert not mesh.is_watertight()

    def test_closed_quad_watertight(self, device):
        """Closed quad (4 triangles meeting at center) is watertight in 2D sense."""
        ### In 2D, "watertight" means all edges are shared by exactly 2 triangles
        ### This creates a closed shape with no boundary
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

        ### This should NOT be watertight because outer edges are only shared by 1 triangle
        assert not mesh.is_watertight()

    def test_empty_mesh_watertight(self, device):
        """Empty mesh is considered watertight."""
        points = torch.empty((0, 2), device=device)
        cells = torch.empty((0, 3), device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_watertight()


class TestWatertight3D:
    """Test watertight checking for 3D meshes."""

    def test_single_tet_not_watertight(self, device):
        """Single tetrahedron is not watertight (has boundary faces)."""
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

        assert not mesh.is_watertight()

    def test_two_tets_not_watertight(self, device):
        """Two tets sharing a face are not watertight (have boundary faces)."""
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
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        assert not mesh.is_watertight()

    def test_filled_volume_not_watertight(self, device):
        """A filled volume mesh is not watertight (has exterior boundary).

        Note: For codimension-0 meshes (3D in 3D), being watertight means every
        triangular face is shared by exactly 2 tets. This is topologically impossible
        for finite meshes in Euclidean 3D space - any solid volume must have an
        exterior boundary. A truly watertight 3D mesh would require periodic boundaries
        or non-Euclidean topology (like a 3-torus embedded in 4D).
        """
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        ### Create a filled volume (tetrahedral mesh)
        mesh = lumpy_ball.load(device=device)

        ### Even though this is a filled volume, it's NOT watertight
        # The exterior faces are boundary faces (appear only once)
        # Only the interior faces are shared by 2 tets
        assert not mesh.is_watertight()

        ### Verify it has boundary faces
        from physicsnemo.mesh.boundaries import extract_candidate_facets

        candidate_facets, _ = extract_candidate_facets(
            mesh.cells, manifold_codimension=1
        )
        _, counts = torch.unique(candidate_facets, dim=0, return_counts=True)

        # Should have some boundary faces (appearing once)
        n_boundary_faces = (counts == 1).sum().item()
        assert n_boundary_faces > 0, "Expected some boundary faces on volume exterior"


class TestWatertight1D:
    """Test watertight checking for 1D meshes."""

    def test_single_edge_not_watertight(self, device):
        """Single edge is not watertight."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        cells = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert not mesh.is_watertight()

    def test_closed_loop_watertight(self, device):
        """Closed loop of edges is watertight."""
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

        assert mesh.is_watertight()


class TestManifold2D:
    """Test manifold checking for 2D meshes."""

    def test_single_triangle_manifold(self, device):
        """Single triangle is a valid manifold with boundary."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_manifold()

    def test_two_triangles_manifold(self, device):
        """Two triangles sharing an edge form a valid manifold."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_manifold()

    def test_non_manifold_edge(self, device):
        """Three triangles sharing an edge create non-manifold configuration."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0]],
            device=device,
        )
        ### All three triangles share edge [0, 1]
        cells = torch.tensor(
            [[0, 1, 2], [1, 0, 3], [0, 1, 3]],  # Three different triangles on same edge
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        assert not mesh.is_manifold()

    def test_manifold_check_levels(self, device):
        """Test different manifold check levels."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### All check levels should pass for simple triangle
        assert mesh.is_manifold(check_level="facets")
        assert mesh.is_manifold(check_level="edges")
        assert mesh.is_manifold(check_level="full")


class TestManifold3D:
    """Test manifold checking for 3D meshes."""

    def test_single_tet_manifold(self, device):
        """Single tetrahedron is a valid manifold with boundary."""
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

        assert mesh.is_manifold()

    def test_two_tets_manifold(self, device):
        """Two tets sharing a face form a valid manifold."""
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
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_manifold()

    def test_non_manifold_face(self, device):
        """Three tets sharing a face create non-manifold configuration."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.5, 0.5, 0.5],  # Extra point
            ],
            device=device,
        )
        ### Three tets share face [0, 1, 2]
        cells = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 4],
                [0, 1, 2, 5],  # Third tet sharing same face
            ],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        assert not mesh.is_manifold()


class TestManifold1D:
    """Test manifold checking for 1D meshes."""

    def test_single_edge_manifold(self, device):
        """Single edge is a valid manifold."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        cells = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_manifold()

    def test_chain_of_edges_manifold(self, device):
        """Chain of edges is a valid manifold."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1], [1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_manifold()

    def test_non_manifold_vertex(self, device):
        """Three edges meeting at a vertex create non-manifold configuration."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            device=device,
        )
        ### Three edges share vertex 0
        cells = torch.tensor(
            [[0, 1], [0, 2], [0, 3]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        ### For 1D meshes, a vertex with 3 incident edges is non-manifold
        ### (locally doesn't look like R^1)
        ### Each vertex should have at most 2 incident edges
        assert not mesh.is_manifold()


class TestEmptyMesh:
    """Test topology checks on empty mesh."""

    def test_empty_mesh_watertight_and_manifold(self, device):
        """Empty mesh is considered both watertight and manifold."""
        points = torch.empty((0, 3), device=device)
        cells = torch.empty((0, 4), device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_watertight()
        assert mesh.is_manifold()


class TestWatertightFaceDeletion:
    """Test that deleting faces from a watertight mesh makes it non-watertight."""

    def test_lumpy_sphere_is_watertight(self, device):
        """Verify that lumpy_sphere is watertight before any modifications."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        assert mesh.is_watertight(), (
            "lumpy_sphere should be watertight (closed surface with no boundary)"
        )

    @pytest.mark.parametrize(
        "n_faces_to_delete,description",
        [
            (1, "single face deleted"),
            (3, "three faces deleted"),
            ("half", "half of all faces deleted"),
        ],
    )
    def test_deleted_faces_not_watertight(self, device, n_faces_to_delete, description):
        """Deleting faces from lumpy_sphere should make it non-watertight.

        Args:
            device: Test device (CPU or CUDA)
            n_faces_to_delete: Number of faces to delete, or "half" for half of all faces
            description: Human-readable description for test output
        """
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        mesh = lumpy_sphere.load(subdivisions=2, device=device)
        n_cells = mesh.n_cells

        ### Determine how many faces to delete
        if n_faces_to_delete == "half":
            num_to_delete = n_cells // 2
        else:
            num_to_delete = n_faces_to_delete

        ### Verify we have enough faces to delete
        assert num_to_delete <= n_cells, (
            f"Cannot delete {num_to_delete} faces from mesh with {n_cells} cells"
        )

        ### Create broken mesh by keeping only cells after the deleted ones
        # Construct directly to avoid TensorDict indexing issues
        broken_mesh = Mesh(
            points=mesh.points,
            cells=mesh.cells[num_to_delete:],
        )

        ### Verify the mesh now has fewer cells
        assert broken_mesh.n_cells == n_cells - num_to_delete

        ### The mesh should no longer be watertight (has boundary edges)
        assert not broken_mesh.is_watertight(), (
            f"Mesh with {description} should NOT be watertight "
            f"(deleted {num_to_delete} of {n_cells} faces)"
        )
