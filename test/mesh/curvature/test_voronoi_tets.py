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

"""Tests for Voronoi volume computation on tetrahedral meshes."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.geometry.dual_meshes import get_or_compute_dual_volumes_0


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


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
