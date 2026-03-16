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

"""Comprehensive tests for mesh repair operations."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.repair import (
    fill_holes,
    remove_degenerate_cells,
    remove_isolated_points,
    repair_mesh,
)


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


class TestDuplicateRemoval:
    """Tests for duplicate vertex removal."""

    def test_remove_exact_duplicates(self, device):
        """Test removing exact duplicate vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Exact duplicate of 0
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean(tolerance=1e-10)

        assert mesh_clean.n_points == 3
        assert mesh_clean.n_cells == 1

    def test_remove_near_duplicates(self, device):
        """Test removing near-duplicate vertices within tolerance."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 1e-7],  # Near duplicate of 0
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean(tolerance=1e-6)

        assert mesh_clean.n_points == 3

    def test_no_duplicates(self, device):
        """Test mesh with no duplicates."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean()

        assert mesh_clean.n_points == 3
        assert torch.equal(mesh_clean.points, mesh.points)

    def test_multiple_duplicates(self, device):
        """Test removing multiple duplicate vertex groups."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Dup of 0
                [1.0, 0.0],  # Dup of 1
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean()

        assert mesh_clean.n_points == 3

    def test_preserves_cell_connectivity(self, device):
        """Test that cell connectivity is correctly remapped."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Dup of 0
            ],
            dtype=torch.float32,
            device=device,
        )

        # Cell references duplicate
        cells = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean()

        # Verify cell still forms valid triangle
        assert mesh_clean.n_cells == 1

        # Should form a triangle
        area = mesh_clean.cell_areas[0]
        assert area > 0


class TestDegenerateRemoval:
    """Tests for degenerate cell removal."""

    def test_remove_zero_area_cells(self, device):
        """Test removing cells with zero area."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.0],  # Collinear with 1, makes degenerate triangle
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],  # Good triangle
                [1, 3, 1],  # Degenerate (duplicate vertex)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_degenerate_cells(mesh)

        assert stats["n_duplicate_vertex_cells"] == 1
        assert mesh_clean.n_cells == 1

    def test_no_degenerates(self, device):
        """Test mesh with no degenerate cells."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_degenerate_cells(mesh)

        assert stats["n_zero_area_cells"] == 0
        assert stats["n_duplicate_vertex_cells"] == 0
        assert mesh_clean.n_cells == 1


class TestIsolatedRemoval:
    """Tests for isolated vertex removal."""

    def test_remove_single_isolated(self, device):
        """Test removing single isolated vertex."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_isolated_points(mesh)

        assert stats["n_isolated_removed"] == 1
        assert mesh_clean.n_points == 3
        assert mesh_clean.n_cells == 1

    def test_remove_multiple_isolated(self, device):
        """Test removing multiple isolated vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
                [6.0, 6.0],  # Isolated
                [7.0, 7.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_isolated_points(mesh)

        assert stats["n_isolated_removed"] == 3
        assert mesh_clean.n_points == 3

    def test_no_isolated(self, device):
        """Test mesh with no isolated vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_isolated_points(mesh)

        assert stats["n_isolated_removed"] == 0
        assert mesh_clean.n_points == 3


class TestRepairPipeline:
    """Tests for comprehensive repair pipeline."""

    def test_pipeline_all_operations(self, device):
        """Test full pipeline with all problems."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Duplicate
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],  # Good
                [1, 1, 2],  # Degenerate
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, all_stats = repair_mesh(
            mesh,
        )

        # Should have fixed all problems
        assert mesh_clean.n_points == 3
        assert mesh_clean.n_cells == 1

        # Verify individual stats
        assert "degenerates" in all_stats
        assert "merge_points" in all_stats
        assert "isolated" in all_stats

        assert all_stats["degenerates"]["n_cells_original"] == 2
        assert all_stats["degenerates"]["n_cells_final"] == 1

    def test_pipeline_clean_mesh_unchanged(self, device):
        """Test that clean mesh is unchanged by pipeline."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Use lumpy_sphere - a complex, watertight mesh that should be clean
        mesh = lumpy_sphere.load(subdivisions=2, device=device)
        original_n_points = mesh.n_points
        original_n_cells = mesh.n_cells

        mesh_clean, stats = repair_mesh(mesh)

        # Should be unchanged
        assert mesh_clean.n_points == original_n_points
        assert mesh_clean.n_cells == original_n_cells
        assert stats["degenerates"]["n_zero_area_cells"] == 0
        assert stats["merge_points"]["n_duplicates_merged"] == 0
        assert stats["isolated"]["n_isolated_removed"] == 0

    def test_pipeline_preserves_data(self, device):
        """Test that repair preserves point and cell data."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)
        mesh.point_data["temperature"] = torch.tensor(
            [1.0, 2.0, 3.0, 999.0], device=device
        )
        mesh.cell_data["pressure"] = torch.tensor([100.0], device=device)

        mesh_clean, stats = repair_mesh(mesh, remove_isolated=True)

        # Data should be preserved for remaining points/cells
        assert "temperature" in mesh_clean.point_data
        assert "pressure" in mesh_clean.cell_data
        assert mesh_clean.point_data["temperature"].shape == (3,)
        assert mesh_clean.cell_data["pressure"].shape == (1,)


class TestHoleFilling:
    """Tests for hole filling."""

    def test_fill_simple_hole(self, device):
        """Test filling a simple boundary loop."""
        # Create mesh with hole (triangle with one missing face)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 0.5, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Only one triangle - leaves edges as boundaries
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_filled, stats = fill_holes(mesh)

        # Should add faces
        assert stats["n_holes_detected"] >= 1
        assert (
            mesh_filled.n_cells > mesh.n_cells or mesh_filled.n_points > mesh.n_points
        )

    def test_closed_mesh_no_holes(self, device):
        """Test that closed mesh is unchanged."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        # Use sphere_icosahedral - a complex watertight closed surface
        mesh = sphere_icosahedral.load(subdivisions=1, device=device)

        mesh_filled, stats = fill_holes(mesh)

        # Should find no holes
        assert stats["n_holes_filled"] == 0


class TestRepairIntegration:
    """Integration tests for repair operations."""

    def test_repair_sequence_order_matters(self, device):
        """Test that repair operations work correctly in sequence."""
        # Create mesh with multiple problems
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Duplicate
                [5.0, 5.0],  # Will become isolated after degenerate removal
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],  # Good triangle
                [3, 4, 4],  # Degenerate (duplicate vertex)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        # Apply repairs in correct order
        mesh1, _ = remove_degenerate_cells(mesh)
        assert mesh1.n_cells == 1  # Removed degenerate

        mesh2 = mesh1.clean(remove_unused_points=False)
        assert mesh2.n_points == 4  # Merged duplicates

        mesh3, _ = remove_isolated_points(mesh2)
        assert mesh3.n_points == 3  # Removed isolated

        # Final mesh should be clean
        validation = mesh3.validate()
        assert validation["valid"]

    def test_idempotence(self, device):
        """Test that applying repair twice doesn't change result."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Duplicate
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Apply twice
        mesh1, stats1 = repair_mesh(mesh)
        mesh2, stats2 = repair_mesh(mesh1)

        # Second application should find no problems
        assert stats2["merge_points"]["n_duplicates_merged"] == 0
        assert stats2["degenerates"]["n_zero_area_cells"] == 0
        assert stats2["isolated"]["n_isolated_removed"] == 0

        # Meshes should be identical
        assert mesh1.n_points == mesh2.n_points
        assert mesh1.n_cells == mesh2.n_cells
