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

"""Tests for Mesh.slice_points and Mesh.slice_cells methods.

This module tests the slicing operations that create submeshes by selecting
subsets of points or cells. Special attention is paid to:
- Cell index remapping when slicing points
- Proper filtering of cells that reference removed points
- Preservation of point_data and cell_data through slicing
- Various index types (int, slice, tensor, boolean mask, list)
"""

import pytest
import torch

from physicsnemo.mesh import Mesh


class TestSlicePoints:
    """Tests for Mesh.slice_points method."""

    def test_slice_points_basic_integer_indices(self):
        """Slicing with integer indices should remap cells correctly."""
        # Create a mesh with 4 points and 2 triangular cells
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep points 0, 1, 2 - only first cell should survive
        sliced = mesh.slice_points([0, 1, 2])

        assert sliced.n_points == 3
        assert sliced.n_cells == 1
        # Indices should be remapped to 0, 1, 2 (same as original since we kept 0,1,2)
        assert sliced.cells.tolist() == [[0, 1, 2]]

    def test_slice_points_removes_all_cells(self):
        """Slicing that removes vertices used by all cells should result in empty cells."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep only points 0 and 2 - no cell can survive (each needs at least 3 verts)
        sliced = mesh.slice_points([0, 2])

        assert sliced.n_points == 2
        assert sliced.n_cells == 0
        assert sliced.cells.shape == (0, 3)
        # Verify points are correctly sliced
        expected_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        assert torch.allclose(sliced.points, expected_points)

    def test_slice_points_index_remapping(self):
        """Verify that cell indices are correctly remapped to new point numbering."""
        # Create a mesh where we'll remove a point in the middle
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=torch.float32
        )
        # One triangle using points 0, 1, 3 (skipping point 2)
        cells = torch.tensor([[0, 1, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep only points 0, 1, 3 (removing point 2)
        sliced = mesh.slice_points([0, 1, 3])

        assert sliced.n_points == 3
        assert sliced.n_cells == 1
        # Old point 3 should now be point 2 (since we removed point 2)
        assert sliced.cells.tolist() == [[0, 1, 2]]

    def test_slice_points_single_int(self):
        """Slicing with a single integer should work."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep only point 1
        sliced = mesh.slice_points(1)

        assert sliced.n_points == 1
        assert sliced.n_cells == 0  # No cells can survive with only 1 point
        assert torch.allclose(sliced.points, torch.tensor([[1.0, 0.0]]))

    def test_slice_points_with_slice_object(self):
        """Slicing with Python slice should work."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep first 3 points - only first cell should survive
        sliced = mesh.slice_points(slice(0, 3))

        assert sliced.n_points == 3
        assert sliced.n_cells == 1
        assert sliced.cells.tolist() == [[0, 1, 2]]

    def test_slice_points_with_boolean_mask(self):
        """Slicing with boolean mask should work."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep points 0, 2, 3 via boolean mask - second cell should survive
        mask = torch.tensor([True, False, True, True])
        sliced = mesh.slice_points(mask)

        assert sliced.n_points == 3
        assert sliced.n_cells == 1
        # Old indices [0, 2, 3] -> new indices [0, 1, 2]
        assert sliced.cells.tolist() == [[0, 1, 2]]

    def test_slice_points_with_tensor_indices(self):
        """Slicing with integer tensor should work."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep points 0, 2, 3 via tensor
        indices = torch.tensor([0, 2, 3])
        sliced = mesh.slice_points(indices)

        assert sliced.n_points == 3
        assert sliced.n_cells == 1
        # Old indices [0, 2, 3] -> new indices [0, 1, 2]
        assert sliced.cells.tolist() == [[0, 1, 2]]

    def test_slice_points_none_returns_self(self):
        """Slicing with None should return the same mesh."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        sliced = mesh.slice_points(None)

        assert sliced is mesh

    def test_slice_points_ellipsis_returns_self(self):
        """Slicing with Ellipsis should return the same mesh."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        sliced = mesh.slice_points(...)

        assert sliced is mesh

    def test_slice_points_preserves_point_data(self):
        """Point data should be correctly sliced along with points."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        point_data = {"temperature": torch.tensor([100.0, 200.0, 300.0, 400.0])}
        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        # Keep points 0, 2, 3
        sliced = mesh.slice_points([0, 2, 3])

        expected_temps = torch.tensor([100.0, 300.0, 400.0])
        assert torch.allclose(sliced.point_data["temperature"], expected_temps)

    def test_slice_points_filters_cell_data(self):
        """Cell data should be correctly filtered when cells are removed."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        cell_data = {"pressure": torch.tensor([10.0, 20.0])}
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        # Keep points 0, 2, 3 - only second cell survives
        sliced = mesh.slice_points([0, 2, 3])

        assert sliced.n_cells == 1
        expected_pressure = torch.tensor([20.0])
        assert torch.allclose(sliced.cell_data["pressure"], expected_pressure)

    def test_slice_points_preserves_global_data(self):
        """Global data should be preserved through slicing."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        global_data = {"time": torch.tensor(1.5)}
        mesh = Mesh(points=points, cells=cells, global_data=global_data)

        sliced = mesh.slice_points([0, 1])

        assert torch.allclose(sliced.global_data["time"], torch.tensor(1.5))

    def test_slice_points_1d_mesh(self):
        """Slicing should work correctly for edge (1-simplex) meshes."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32
        )
        # Edges: 0-1, 1-2, 2-3
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep points 1, 2, 3 - edges 1-2 and 2-3 survive, edge 0-1 is removed
        sliced = mesh.slice_points([1, 2, 3])

        assert sliced.n_points == 3
        assert sliced.n_cells == 2
        # Old indices [1,2], [2,3] -> new indices [0,1], [1,2]
        assert sliced.cells.tolist() == [[0, 1], [1, 2]]

    def test_slice_points_3d_tetrahedra(self):
        """Slicing should work correctly for tetrahedral (3-simplex) meshes."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        # Two tetrahedra: 0-1-2-3 and 1-2-3-4
        cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep points 1, 2, 3, 4 - only second tet survives
        sliced = mesh.slice_points([1, 2, 3, 4])

        assert sliced.n_points == 4
        assert sliced.n_cells == 1
        # Old indices [1,2,3,4] -> new indices [0,1,2,3]
        assert sliced.cells.tolist() == [[0, 1, 2, 3]]


class TestSliceCells:
    """Tests for Mesh.slice_cells method."""

    def test_slice_cells_basic(self):
        """Basic cell slicing should work and keep all points."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep only first cell
        sliced = mesh.slice_cells([0])

        assert sliced.n_points == 4  # All points kept
        assert sliced.n_cells == 1
        assert sliced.cells.tolist() == [[0, 1, 2]]

    def test_slice_cells_preserves_cell_data(self):
        """Cell data should be correctly sliced."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        cell_data = {"pressure": torch.tensor([10.0, 20.0])}
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        # Keep only second cell
        sliced = mesh.slice_cells([1])

        assert sliced.n_cells == 1
        assert torch.allclose(sliced.cell_data["pressure"], torch.tensor([20.0]))

    def test_slice_cells_with_boolean_mask(self):
        """Slicing cells with boolean mask should work."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        mask = torch.tensor([False, True])
        sliced = mesh.slice_cells(mask)

        assert sliced.n_cells == 1
        assert sliced.cells.tolist() == [[0, 2, 3]]

    def test_slice_cells_single_int(self):
        """Slicing cells with single integer should work."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        sliced = mesh.slice_cells(1)

        assert sliced.n_cells == 1
        assert sliced.cells.tolist() == [[0, 2, 3]]


class TestSlicingEdgeCases:
    """Edge cases and special scenarios for slicing operations."""

    def test_empty_mesh_after_slicing(self):
        """Slicing that removes all cells should produce valid empty-cell mesh."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep only 2 points - not enough for a triangle
        sliced = mesh.slice_points([0, 1])

        assert sliced.n_points == 2
        assert sliced.n_cells == 0
        assert sliced.cells.shape == (0, 3)

    def test_orphan_points_from_slice_cells(self):
        """slice_cells can leave orphan points (points not in any cell)."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep only first cell - point 3 becomes orphan
        sliced = mesh.slice_cells([0])

        assert sliced.n_points == 4  # All points kept (including orphan)
        assert sliced.n_cells == 1
        # Point 3 is not referenced by any cell but still exists

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_slicing_preserves_device(self, device):
        """Sliced mesh should remain on the same device."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        sliced = mesh.slice_points([0, 1, 2])

        assert sliced.points.device.type == device
        assert sliced.cells.device.type == device

    def test_slice_keeps_all_points_preserves_all_cells(self):
        """Slicing that keeps all points should preserve all cells."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Keep all 4 points
        sliced = mesh.slice_points([0, 1, 2, 3])

        assert sliced.n_points == 4
        assert sliced.n_cells == 2
        assert sliced.cells.tolist() == [[0, 1, 2], [0, 2, 3]]
