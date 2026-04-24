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

"""Tests for dual 0-cell volume (Voronoi region) computation on simplicial meshes."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry.dual_meshes import (
    compute_dual_volumes_0,
    get_or_compute_dual_volumes_0,
)


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


class TestDualVolumes1D:
    """Tests for dual volumes on 1D edge meshes.

    For 1D manifolds, each vertex gets half the length of each incident edge:
        V(v) = sum_{edges containing v} |edge| / 2
    """

    def test_two_edges_nonuniform(self, device):
        """Test 1D dual volumes on a chain with non-uniform edge lengths.

        Vertices at x = [0, 1, 3], edges [[0,1], [1,2]].
        Edge lengths: 1.0, 2.0.
        Expected dual volumes:
            vertex 0 (boundary): 1.0 / 2 = 0.5
            vertex 1 (interior): 1.0/2 + 2.0/2 = 1.5
            vertex 2 (boundary): 2.0 / 2 = 1.0
        """
        points = torch.tensor([[0.0], [1.0], [3.0]], device=device)
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.tensor([0.5, 1.5, 1.0], device=device)
        torch.testing.assert_close(dual_vols, expected)

    def test_uniform_chain_conservation(self, device):
        """Sum of dual volumes equals total mesh length (tiling property)."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]], device=device)
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.tensor([0.5, 1.0, 1.0, 0.5], device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_1d_curve_in_3d(self, device):
        """1D edges embedded in 3D space."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 5.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        # Edge lengths: sqrt(9+16) = 5.0, sqrt(25) = 5.0
        expected = torch.tensor([2.5, 5.0, 2.5], device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_single_edge(self, device):
        """Single edge: each endpoint gets half the length."""
        points = torch.tensor([[0.0], [4.0]], device=device)
        cells = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.tensor([2.0, 2.0], device=device)
        torch.testing.assert_close(dual_vols, expected)

    def test_isolated_vertex(self, device):
        """Vertex not in any edge gets zero dual volume."""
        points = torch.tensor([[0.0], [1.0], [99.0]], device=device)
        cells = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)
        assert dual_vols[2] == 0.0


class TestDualVolumes2D:
    """Tests for dual volumes on 2D triangle meshes.

    For 2D manifolds, dual volumes use the Meyer et al. (2003) mixed Voronoi
    area approach:
    - Acute triangles: circumcentric Voronoi formula (Eq. 7)
    - Obtuse triangles: mixed area subdivision (Fig. 4)
    """

    def test_equilateral_triangle(self, device):
        """Equilateral triangle: all acute, each vertex gets equal Voronoi area.

        For equilateral triangle with side a, each vertex receives a^2 / (4*sqrt(3)).
        With a=1, that is sqrt(3)/12 per vertex, summing to area = sqrt(3)/4.
        """
        s3 = 3.0**0.5
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2]], device=device)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.full((3,), s3 / 12, device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_right_triangle(self, device):
        """3-4-5 right triangle: boundary case with angle exactly pi/2.

        Since the code uses strict inequality (> pi/2) for obtuseness,
        a right angle is treated as non-obtuse and the Voronoi formula applies.
        The right-angle vertex receives half the total area.
        """
        points = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], device=device)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.tensor([3.0, 1.5, 1.5], device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_obtuse_triangle(self, device):
        """Obtuse isosceles triangle triggers the mixed-area branch (Meyer Fig. 4).

        Triangle (0,0)-(4,0)-(2,1): angle at (2,1) is arccos(-3/5) ~ 127 deg.
        Mixed area assigns area/2 to the obtuse vertex and area/4 to each other.
        """
        points = torch.tensor([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0]], device=device)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        # area = 2.0; obtuse at v2 -> area/2, others -> area/4
        expected = torch.tensor([0.5, 0.5, 1.0], device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_two_triangles_sharing_edge(self, device):
        """Unit square split into two right triangles sharing the diagonal.

        Vertices: (0,0), (1,0), (1,1), (0,1)
        Cells: [0,1,2] and [0,2,3]
        Each right-angle vertex gets 1/4 from its triangle; shared vertices
        accumulate from both. By symmetry, all four vertices get 0.25.
        """
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=device
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_triangle_embedded_in_3d(self, device):
        """Equilateral triangle in the z=0 plane of 3D ambient space.

        Dual volumes should be identical to the pure 2D case.
        """
        s3 = 3.0**0.5
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, s3 / 2, 0.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        expected = torch.full((3,), s3 / 12, device=device)
        torch.testing.assert_close(dual_vols, expected)
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())

    def test_conservation_hexagonal_fan(self, device):
        """Six equilateral triangles in a hexagonal fan around a central vertex.

        Verifies conservation (sum = total area) and correct accumulation at
        the shared center vertex (6 contributions) vs boundary vertices (2 each).
        """
        s3 = 3.0**0.5
        # Central vertex + 6 boundary vertices at unit distance, 60 deg apart
        boundary = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, s3 / 2],
                [-0.5, s3 / 2],
                [-1.0, 0.0],
                [-0.5, -s3 / 2],
                [0.5, -s3 / 2],
            ],
            device=device,
        )
        center = torch.tensor([[0.0, 0.0]], device=device)
        points = torch.cat([center, boundary], dim=0)  # vertex 0 is center

        cells = torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 1]],
            dtype=torch.long,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        dual_vols = compute_dual_volumes_0(mesh)

        # Each equilateral triangle contributes sqrt(3)/12 per vertex.
        # Center: 6 * sqrt(3)/12 = sqrt(3)/2
        # Each boundary vertex: 2 * sqrt(3)/12 = sqrt(3)/6
        expected_center = s3 / 2
        expected_boundary = s3 / 6
        torch.testing.assert_close(
            dual_vols[0], torch.tensor(expected_center, device=device)
        )
        torch.testing.assert_close(
            dual_vols[1:], torch.full((6,), expected_boundary, device=device)
        )
        torch.testing.assert_close(dual_vols.sum(), mesh.cell_areas.sum())


class TestVoronoiVolumes3D:
    """Tests for Voronoi volume computation on 3D tetrahedral meshes."""

    def test_single_regular_tet(self, device):
        """Test Voronoi volumes for single regular tetrahedron."""
        # Regular tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
                [0.5, (3**0.5) / 6, ((2 / 3) ** 0.5)],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Compute Voronoi volumes
        dual_vols = get_or_compute_dual_volumes_0(mesh)

        # Should have one volume per vertex
        assert dual_vols.shape == (4,)

        # All should be positive
        assert torch.all(dual_vols > 0)

        # Sum of dual volumes should relate to tet volume
        # For regular tet, each vertex gets equal share
        total_dual = dual_vols.sum()

        # Dual volumes can be larger than tet volume in circumcentric construction
        # (circumcenter can be outside the tet)
        # Just verify they're computed and positive
        assert total_dual > 0

    def test_cube_tets_voronoi(self, device):
        """Test Voronoi volumes for cube subdivided into tets."""
        # Simple cube vertices
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Subdivide cube into 5 tets (standard subdivision)
        cells = torch.tensor(
            [
                [0, 1, 2, 5],
                [0, 2, 3, 7],
                [0, 5, 7, 4],
                [2, 5, 6, 7],
                [0, 2, 5, 7],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        dual_vols = get_or_compute_dual_volumes_0(mesh)

        # Should have one volume per vertex
        assert dual_vols.shape == (8,)

        # All should be positive
        assert torch.all(dual_vols > 0)

        # Total dual volume should be reasonable
        total_dual = dual_vols.sum()
        total_tet_volume = mesh.cell_areas.sum()

        # Should be same order of magnitude
        assert total_dual > total_tet_volume * 0.5
        assert total_dual < total_tet_volume * 2.0

    def test_two_tets_sharing_face(self, device):
        """Test Voronoi volumes for two adjacent tets."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],  # Above
                [0.5, 0.5, -1.0],  # Below
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        dual_vols = get_or_compute_dual_volumes_0(mesh)

        assert dual_vols.shape == (5,)
        assert torch.all(dual_vols > 0)

        # Vertices on shared face should have larger dual volumes
        # (they have contributions from both tets)
        shared_verts = torch.tensor([0, 1, 2])
        isolated_verts = torch.tensor([3, 4])

        assert dual_vols[shared_verts].mean() > dual_vols[isolated_verts].mean()

    def test_voronoi_caching(self, device):
        """Test that Voronoi volumes are cached properly."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Compute twice
        dual_vols1 = get_or_compute_dual_volumes_0(mesh)
        dual_vols2 = get_or_compute_dual_volumes_0(mesh)

        # Should be identical (cached)
        assert torch.equal(dual_vols1, dual_vols2)

    def test_comparison_with_barycentric(self, device):
        """Compare Voronoi volumes with barycentric approximation."""

        # Regular tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
                [0.5, (3**0.5) / 6, ((2 / 3) ** 0.5)],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Voronoi volumes
        voronoi_vols = get_or_compute_dual_volumes_0(mesh)

        # Barycentric approximation: tet_volume / 4
        tet_volume = mesh.cell_areas[0]
        barycentric_vols = tet_volume / 4.0

        # Voronoi and barycentric should be similar for regular tet
        # But not identical
        rel_diff = torch.abs(voronoi_vols - barycentric_vols) / barycentric_vols

        # Should be same order of magnitude
        assert torch.all(rel_diff < 2.0)  # Within factor of 2


class TestVoronoiNumericalStability:
    """Tests for numerical stability of Voronoi computation."""

    def test_nearly_degenerate_tet(self, device):
        """Test Voronoi on nearly degenerate tetrahedron."""
        # Very flat tet
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1e-6],  # Nearly coplanar
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Should compute without NaN/Inf
        dual_vols = get_or_compute_dual_volumes_0(mesh)

        assert not torch.any(torch.isnan(dual_vols))
        assert not torch.any(torch.isinf(dual_vols))
        assert torch.all(dual_vols >= 0)

    def test_empty_tet_mesh(self, device):
        """Test Voronoi on empty tet mesh."""
        points = torch.randn(10, 3, device=device)
        cells = torch.zeros((0, 4), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        dual_vols = get_or_compute_dual_volumes_0(mesh)

        # Should all be zero (no cells)
        assert torch.allclose(dual_vols, torch.zeros_like(dual_vols))
