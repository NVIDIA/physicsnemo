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

"""Tests for physicsnemo.mesh.io module - 1D mesh conversion."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestFromPyvista1D:
    """Tests for converting 1D (line) meshes."""

    def test_line_mesh_auto_detection(self):
        """Test automatic detection of 1D manifold from line mesh."""
        # Create a simple line mesh with 3 separate line segments
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
            ],
            dtype=np.float32,
        )
        # Lines array format: [n_points, point_id_0, point_id_1, ..., n_points, ...]
        # Creating 3 line segments: (0,1), (1,2), (2,3)
        lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3])

        pv_mesh = pv.PolyData(points, lines=lines)

        # Verify it's detected as lines
        assert pv_mesh.n_lines == 3
        assert pv_mesh.n_cells == 3

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 1
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 2  # Line segments
        assert mesh.n_cells == 3  # Three line segments
        assert mesh.n_points == 4

        # Verify connectivity is correct
        expected_cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        assert torch.equal(mesh.cells, expected_cells)

    def test_line_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification for 1D mesh."""
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        lines = np.array([2, 0, 1])  # One line segment with 2 points

        pv_mesh = pv.PolyData(points, lines=lines)
        mesh = from_pyvista(pv_mesh, manifold_dim=1)

        assert mesh.n_manifold_dims == 1
        assert mesh.n_cells == 1
        assert torch.equal(mesh.cells, torch.tensor([[0, 1]], dtype=torch.long))

    def test_spline_from_examples(self):
        """Test conversion of the example spline (polyline curve).

        The example spline is a single continuous polyline with many points,
        which should be converted to line segments between consecutive points.
        """
        pv_mesh = pv.examples.load_spline()

        # Verify it's a polyline (one continuous curve)
        assert pv_mesh.n_lines == 1  # One polyline
        n_points_in_spline = pv_mesh.n_points

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 1
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 2  # Line segments
        assert mesh.n_points == n_points_in_spline
        # A polyline with N points becomes N-1 line segments
        assert mesh.n_cells == n_points_in_spline - 1

        # Verify segments are consecutive
        for i in range(mesh.n_cells):
            assert mesh.cells[i, 0] == i
            assert mesh.cells[i, 1] == i + 1

    def test_spline_constructed(self):
        """Test conversion of a constructed spline using pv.Spline.

        Create a spline through specific points and verify it converts correctly.
        """
        # Create control points for the spline
        control_points = np.array(
            [
                [0, 0, 0],
                [1, 2, 0],
                [2, 1, 1],
                [3, 0, 2],
            ],
            dtype=np.float32,
        )

        # Create a spline with 20 interpolated points
        pv_mesh = pv.Spline(control_points, n_points=20)

        assert pv_mesh.n_lines == 1  # One continuous curve
        assert pv_mesh.n_points == 20

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 1
        assert mesh.n_points == 20
        assert mesh.n_cells == 19  # 20 points -> 19 segments
        assert mesh.cells.shape == (19, 2)

        # Verify all segments connect consecutively
        for i in range(19):
            assert mesh.cells[i, 0] == i
            assert mesh.cells[i, 1] == i + 1

    def test_polyline_replicates_cell_data_to_segments(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        pv_mesh = pv.PolyData(points, lines=np.array([4, 0, 1, 2, 3]))
        pv_mesh.cell_data["line_id"] = np.array([7])

        mesh = from_pyvista(pv_mesh)

        assert torch.equal(
            mesh.cells,
            torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
        )
        assert torch.equal(mesh.cell_data["line_id"], torch.tensor([7, 7, 7]))

    def test_multiple_polylines_keep_parent_data_order(self, monkeypatch):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        pv_mesh = pv.PolyData(points, lines=np.array([3, 0, 1, 2, 3, 3, 4, 5]))
        pv_mesh.cell_data["line_id"] = np.array([7, 9])
        pv_mesh.cell_data["vector"] = np.array([[1, 2], [3, 4]])

        def fail_if_called(*args, **kwargs):
            raise AssertionError("uniform polylines should use the vectorized path")

        monkeypatch.setattr(pv.UnstructuredGrid, "triangulate", fail_if_called)

        mesh = from_pyvista(pv_mesh)

        assert torch.equal(
            mesh.cells,
            torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5]]),
        )
        assert torch.equal(mesh.cell_data["line_id"], torch.tensor([7, 7, 9, 9]))
        assert torch.equal(
            mesh.cell_data["vector"],
            torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]]),
        )

    def test_unstructured_grid_line_is_not_dropped(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pv_mesh = pv.UnstructuredGrid(
            np.array([2, 0, 1]),
            np.array([pv.CellType.LINE]),
            points,
        )
        pv_mesh.cell_data["line_id"] = np.array([3])

        mesh = from_pyvista(pv_mesh)

        assert torch.equal(mesh.cells, torch.tensor([[0, 1]]))
        assert torch.equal(mesh.cell_data["line_id"], torch.tensor([3]))

    def test_quadratic_edge_uses_vtk_node_order(self):
        # VTK orders a quadratic edge as endpoint, endpoint, midpoint. Pairing
        # consecutive input IDs would incorrectly create [0, 1] and [1, 2].
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        pv_mesh = pv.UnstructuredGrid(
            np.array([3, 0, 1, 2]),
            np.array([pv.CellType.QUADRATIC_EDGE]),
            points,
        )
        pv_mesh.cell_data["edge_id"] = np.array([9])

        mesh = from_pyvista(pv_mesh)

        assert torch.equal(mesh.cells, torch.tensor([[0, 2], [2, 1]]))
        assert torch.equal(mesh.cell_data["edge_id"], torch.tensor([9, 9]))
