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

    def test_pinch_point_non_manifold_vertex(self, device):
        """Two tets sharing only a single vertex create a non-manifold pinch point.

        Vertex 0 is shared by both tets, but they share no face containing vertex 0.
        The link of vertex 0 consists of two disconnected triangles: {1,2,3} and
        {4,5,6}. This passes facet and edge checks but fails the vertex check.
        """
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # vertex 0: pinch point
                [1.0, 0.0, 0.0],  # tet 1 only
                [0.0, 1.0, 0.0],  # tet 1 only
                [0.0, 0.0, 1.0],  # tet 1 only
                [-1.0, 0.0, 0.0],  # tet 2 only
                [0.0, -1.0, 0.0],  # tet 2 only
                [0.0, 0.0, -1.0],  # tet 2 only
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 4, 5, 6]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Facet and edge checks pass (no shared faces or edges between the two tets)
        assert mesh.is_manifold(check_level="facets") is True
        assert mesh.is_manifold(check_level="edges") is True

        ### Full check catches the pinch point
        assert mesh.is_manifold(check_level="full") is False

    def test_shared_edge_but_no_shared_face_non_manifold(self, device):
        """Two tets sharing a vertex and an edge but no face containing that vertex.

        Tets share edge {0,1}, but at vertex 0 the link faces {1,2,3} and {1,4,5}
        share only vertex 1 (not an edge), so vertex 0's link is disconnected.
        """
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 4, 5]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Vertex 0's link has two triangles sharing only a vertex, not an edge
        assert mesh.is_manifold(check_level="full") is False

    def test_three_tets_ring_manifold(self, device):
        """Three tets forming a ring around a shared edge are manifold.

        Tets share edge {0,1}. At vertex 0, the link faces all share edges
        through vertex 1, forming a connected fan.
        """
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            device=device,
        )
        ### Three tets: each pair shares a face containing both v0 and v1
        cells = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 3, 4],
                [0, 1, 4, 2],
            ],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Link of vertex 0: {1,2,3}, {1,3,4}, {1,4,2}
        ### These share edges {1,3}, {1,4}, {1,2} respectively → connected
        assert mesh.is_manifold(check_level="full") is True


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


###############################################################################
# Regression: _check_edges_manifold for 3D tetrahedral meshes
###############################################################################


class TestEdgeManifoldCheck3DRegression:
    """Regression tests for 3D edge-link connectivity checking.

    The original ``_check_edges_manifold`` for 3D meshes only checked that
    each edge appeared in at least one cell (trivially true), so it always
    returned True.  The fix implements proper face-link connectivity around
    each edge via union-find.
    """

    def test_two_tets_sharing_only_edge_is_non_manifold(self, device):
        """Two tets sharing only an edge (no shared face) are non-manifold.

        T1 = (0,1,2,3), T2 = (0,1,4,5).  All 8 triangular faces appear
        exactly once (passes facet check).  But the face-link around edge
        (0,1) is disconnected: {(0,1,2),(0,1,3)} and {(0,1,4),(0,1,5)} form
        two components.  The old implementation returned True; the fix
        correctly returns False.
        """
        from physicsnemo.mesh.boundaries._topology import (
            _check_edges_manifold,
            _check_facets_manifold,
        )

        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, -1.0, 0.0],
                [0.5, -0.5, -1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 4, 5]], device=device, dtype=torch.int64
        )
        mesh = Mesh(points=points, cells=cells)

        assert _check_facets_manifold(mesh), "Facets should pass"
        assert not _check_edges_manifold(mesh), (
            "Edges check should fail for two tets sharing only an edge"
        )

    def test_two_tets_sharing_face_is_manifold(self, device):
        """Two tets sharing a complete face are manifold."""
        from physicsnemo.mesh.boundaries._topology import _check_edges_manifold

        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.5, -1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 4]], device=device, dtype=torch.int64
        )
        mesh = Mesh(points=points, cells=cells)

        assert _check_edges_manifold(mesh), (
            "Two tets sharing face (0,1,2) should pass edge check"
        )

    def test_tet_cycle_around_edge_is_manifold(self, device):
        """Four tets forming a closed cycle around edge (0,1) are manifold."""
        from physicsnemo.mesh.boundaries._topology import _check_edges_manifold

        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.0, 1.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, -1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        assert _check_edges_manifold(mesh), (
            "4-tet cycle around an edge should pass edge check"
        )

    def test_is_manifold_edges_level_catches_nonmanifold(self, device):
        """is_manifold(check_level='edges') returns False for the non-manifold case."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, -1.0, 0.0],
                [0.5, -0.5, -1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 4, 5]], device=device, dtype=torch.int64
        )
        mesh = Mesh(points=points, cells=cells)

        assert mesh.is_manifold(check_level="facets") is True
        assert mesh.is_manifold(check_level="edges") is False
        assert mesh.is_manifold(check_level="full") is False
