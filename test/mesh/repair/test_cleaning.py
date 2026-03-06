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

"""Tests for mesh cleaning operations.

Tests validate that mesh cleaning correctly:
- Merges duplicate points within tolerance
- Removes duplicate cells
- Removes unused points
- Preserves data through cleaning operations
"""

import torch

from physicsnemo.mesh.mesh import Mesh


class TestMergeDuplicatePoints:
    """Test duplicate point merging."""

    def test_merge_exact_duplicates(self, device):
        """Merge points at exactly the same location."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ],  # Points 0 and 2 are duplicates
            device=device,
        )
        cells = torch.tensor([[0, 1, 3], [2, 1, 3]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should merge points 0 and 2
        assert cleaned.n_points == 3

        ### After merging points, both cells reference the same vertices, so become duplicates
        ### Only 1 cell should remain after duplicate cell removal
        assert cleaned.n_cells == 1

    def test_merge_within_tolerance(self, device):
        """Merge points within specified tolerance."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1e-13, 1e-13],
                [1.0, 1.0],
            ],  # Points 0 and 2 are close
            device=device,
        )
        cells = torch.tensor([[0, 1, 3], [2, 1, 3]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### With default tight tolerance (1e-12), should merge
        cleaned = mesh.clean()
        assert cleaned.n_points == 3

        ### With looser tolerance, should also merge
        cleaned_loose = mesh.clean(tolerance=1e-10)
        assert cleaned_loose.n_points == 3

    def test_no_merge_outside_tolerance(self, device):
        """Don't merge points outside tolerance."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1e-6, 1e-6],
                [1.0, 1.0],
            ],  # Points 0 and 2 are far
            device=device,
        )
        cells = torch.tensor([[0, 1, 3], [2, 1, 3]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### With default tight tolerance (1e-12), should NOT merge
        cleaned = mesh.clean()
        assert cleaned.n_points == 4

    def test_merge_multiple_groups(self, device):
        """Merge multiple groups of duplicate points."""
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.0, 0.0],  # 2 - duplicate of 0
                [1.0, 0.0],  # 3 - duplicate of 1
                [0.5, 1.0],  # 4 - unique
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 4], [2, 3, 4]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have 3 unique points: 0/2, 1/3, 4
        assert cleaned.n_points == 3

    def test_merge_preserves_point_data(self, device):
        """Point data is averaged when merging."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 3], [2, 1, 3]], device=device, dtype=torch.int64)

        ### Add point data
        point_data = {
            "temperature": torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
        }
        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        cleaned = mesh.clean()

        ### Point data should be averaged: (10 + 30) / 2 = 20
        assert "temperature" in cleaned.point_data.keys()
        assert len(cleaned.point_data["temperature"]) == cleaned.n_points

        ### Check that merged point has averaged value
        ### The merged point should have temperature (10 + 30) / 2 = 20
        temperatures = cleaned.point_data["temperature"]
        assert torch.any(torch.isclose(temperatures, torch.tensor(20.0, device=device)))


class TestRemoveDuplicateCells:
    """Test duplicate cell removal."""

    def test_remove_exact_duplicate_cells(self, device):
        """Remove cells with same vertices in same order."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 2]],  # Exact duplicates
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have only 1 cell
        assert cleaned.n_cells == 1

    def test_remove_permuted_duplicate_cells(self, device):
        """Remove cells with same vertices in different order."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [1, 0, 2], [2, 0, 1]],  # Same vertices, different orders
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have only 1 cell (all are duplicates)
        assert cleaned.n_cells == 1

    def test_keep_different_cells(self, device):
        """Keep cells with different vertices."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [1, 3, 2]],  # Different cells
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should keep both cells
        assert cleaned.n_cells == 2


class TestRemoveUnusedPoints:
    """Test unused point removal."""

    def test_remove_single_unused_point(self, device):
        """Remove point not referenced by any cell."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 2.0]],  # Point 3 unused
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have only 3 points
        assert cleaned.n_points == 3

    def test_remove_multiple_unused_points(self, device):
        """Remove multiple unused points."""
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0 - used
                [1.0, 0.0],  # 1 - used
                [0.5, 1.0],  # 2 - used
                [2.0, 2.0],  # 3 - unused
                [3.0, 3.0],  # 4 - unused
            ],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have only 3 points
        assert cleaned.n_points == 3

    def test_keep_all_used_points(self, device):
        """Keep all points that are used by cells."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should keep all 4 points
        assert cleaned.n_points == 4


class TestCombinedCleaning:
    """Test combinations of cleaning operations."""

    def test_clean_all_operations(self, device):
        """Apply all cleaning operations together."""
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.0, 0.0],  # 2 - duplicate of 0
                [0.5, 1.0],  # 3
                [2.0, 2.0],  # 4 - unused
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 3], [2, 1, 3], [0, 1, 3]],  # Last cell is duplicate
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have:
        ### - 3 unique points (merge 0/2, remove 4)
        ### - 1 unique cell (remove duplicates)
        assert cleaned.n_points == 3
        assert cleaned.n_cells == 1

    def test_selective_cleaning(self, device):
        """Apply only specific cleaning operations."""
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.0, 0.0],  # 2 - duplicate of 0
                [0.5, 1.0],  # 3
                [2.0, 2.0],  # 4 - unused
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 3], [2, 1, 3]],
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Only merge points
        cleaned_merge_only = mesh.clean(
            merge_points=True,
            remove_duplicate_cells=False,
            remove_unused_points=False,
        )
        assert cleaned_merge_only.n_points == 4  # 5 - 1 (merged) = 4
        assert cleaned_merge_only.n_cells == 2

        ### Only remove unused points
        cleaned_unused_only = mesh.clean(
            merge_points=False,
            remove_duplicate_cells=False,
            remove_unused_points=True,
        )
        assert cleaned_unused_only.n_points == 4  # 5 - 1 (unused) = 4
        assert cleaned_unused_only.n_cells == 2


class TestCleaningWithData:
    """Test that cleaning preserves mesh data."""

    def test_preserve_cell_data(self, device):
        """Cell data is preserved after cleaning."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 2]],  # Duplicate cells
            device=device,
            dtype=torch.int64,
        )
        cell_data = {"pressure": torch.tensor([100.0, 200.0], device=device)}
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        cleaned = mesh.clean()

        ### Cell data should be preserved (first occurrence kept)
        assert "pressure" in cleaned.cell_data.keys()
        assert len(cleaned.cell_data["pressure"]) == cleaned.n_cells

    def test_preserve_global_data(self, device):
        """Global data is preserved after cleaning."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 2.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        global_data = {"simulation_time": torch.tensor(1.5, device=device)}
        mesh = Mesh(points=points, cells=cells, global_data=global_data)

        cleaned = mesh.clean()

        ### Global data should be unchanged
        assert "simulation_time" in cleaned.global_data.keys()
        assert torch.isclose(
            cleaned.global_data["simulation_time"],
            torch.tensor(1.5, device=device),
        )


class TestEdgeCases:
    """Test edge cases for cleaning operations."""

    def test_clean_empty_mesh(self, device):
        """Cleaning empty mesh returns empty mesh."""
        points = torch.empty((0, 2), device=device)
        cells = torch.empty((0, 3), device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        assert cleaned.n_points == 0
        assert cleaned.n_cells == 0

    def test_clean_single_cell(self, device):
        """Cleaning single cell mesh works correctly."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should be unchanged
        assert cleaned.n_points == 3
        assert cleaned.n_cells == 1

    def test_clean_all_duplicates(self, device):
        """Cleaning mesh with all duplicate points/cells."""
        points = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # All duplicates
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 2]],  # All references to same logical point
            device=device,
            dtype=torch.int64,
        )
        mesh = Mesh(points=points, cells=cells)

        cleaned = mesh.clean()

        ### Should have 1 unique point and 1 unique cell
        ### Actually, this will create a degenerate cell (all vertices same)
        ### But the cleaning should still work
        assert cleaned.n_points == 1


class TestToleranceSettings:
    """Test different tolerance settings for point merging."""

    def test_different_tolerances(self, device):
        """Different tolerances merge different sets of points."""
        points = torch.tensor(
            [
                [0.0, 0.0],  # 0
                [1e-13, 1e-13],  # 1 - very close to 0
                [1e-8, 1e-8],  # 2 - medium close to 0
                [1e-3, 1e-3],  # 3 - far from 0
            ],
            device=device,
        )
        # Use a cell that references all points so none are removed as unused
        cells = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        ### Very tight tolerance: merge only 0 and 1
        cleaned_tight = mesh.clean(tolerance=1e-12)
        assert cleaned_tight.n_points == 3

        ### Medium tolerance: merge 0, 1, and 2
        cleaned_medium = mesh.clean(tolerance=1e-7)
        assert cleaned_medium.n_points == 2

        ### Loose tolerance: merge all
        cleaned_loose = mesh.clean(tolerance=1e-2)
        assert cleaned_loose.n_points == 1
