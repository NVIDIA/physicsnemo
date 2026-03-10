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

"""Tests for physicsnemo.mesh.io module - round-trip conversion.

Tests validate PyVista ↔ physicsnemo.mesh conversion preserves geometry and data
across spatial dimensions and compute backends.
"""

import numpy as np
import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista  # noqa: E402

### Helper Functions ###


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


class TestRoundTrip:
    """Tests for round-trip conversion: PyVista → Mesh → PyVista."""

    def test_round_trip_2d_airplane(self):
        """Test round-trip conversion preserves geometry for 2D mesh."""
        pv_original = pv.examples.load_airplane()

        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)

        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert pv_reconstructed.n_cells == pv_original.n_cells
        assert np.allclose(pv_reconstructed.points, pv_original.points)

    def test_round_trip_3d_tetbeam(self):
        """Test round-trip conversion preserves geometry for 3D mesh."""
        pv_original = pv.examples.load_tetbeam()

        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)

        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert pv_reconstructed.n_cells == pv_original.n_cells
        assert np.allclose(pv_reconstructed.points, pv_original.points)

        # Verify connectivity is preserved
        assert np.array_equal(
            pv_reconstructed.cells_dict[pv.CellType.TETRA],
            pv_original.cells_dict[pv.CellType.TETRA],
        )

    def test_round_trip_1d_spline(self):
        """Test round-trip conversion for 1D mesh."""
        pv_original = pv.examples.load_spline()

        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)

        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        # Line count matches (spline has 1 polyline, we convert to N-1 segments)
        assert pv_reconstructed.n_lines == mesh.n_cells

    def test_round_trip_0d_pointset(self):
        """Test round-trip conversion for 0D mesh."""
        points_orig = np.random.rand(25, 3).astype(np.float32)
        pv_original = pv.PointSet(points_orig)

        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)

        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert np.allclose(pv_reconstructed.points, pv_original.points)

    def test_round_trip_with_data(self):
        """Test round-trip conversion preserves data arrays."""
        pv_original = pv.Sphere(theta_resolution=10, phi_resolution=10)
        pv_original.clear_data()

        # Add data
        pv_original.point_data["scalars"] = np.random.rand(pv_original.n_points).astype(
            np.float32
        )
        pv_original.cell_data["ids"] = np.arange(pv_original.n_cells, dtype=np.int32)
        pv_original.field_data["metadata"] = np.array([42], dtype=np.int64)

        # Convert to Mesh and back
        mesh = from_pyvista(pv_original)
        pv_reconstructed = to_pyvista(mesh)

        # Verify data is preserved
        assert "scalars" in pv_reconstructed.point_data
        assert "ids" in pv_reconstructed.cell_data
        assert "metadata" in pv_reconstructed.field_data

        # Verify values match
        assert np.allclose(
            pv_reconstructed.point_data["scalars"], pv_original.point_data["scalars"]
        )
        assert np.array_equal(
            pv_reconstructed.cell_data["ids"], pv_original.cell_data["ids"]
        )
        assert np.array_equal(
            pv_reconstructed.field_data["metadata"], pv_original.field_data["metadata"]
        )


### Parametrized Tests for Device Handling ###


class TestRoundTripParametrized:
    """Parametrized tests for round-trip conversion across backends."""

    @pytest.mark.parametrize(
        "pv_example_loader",
        [
            lambda: pv.examples.load_airplane(),  # 2D surface
            lambda: pv.examples.load_tetbeam(),  # 3D volume
        ],
    )
    def test_round_trip_device_parametrized(self, pv_example_loader, device):
        """Test round-trip conversion with device transfer."""
        pv_original = pv_example_loader()

        # Convert to Mesh on specified device
        mesh_cpu = from_pyvista(pv_original)
        mesh = Mesh(
            points=mesh_cpu.points.to(device),
            cells=mesh_cpu.cells.to(device),
            point_data=mesh_cpu.point_data,
            cell_data=mesh_cpu.cell_data,
            global_data=mesh_cpu.global_data,
        )

        # Verify device
        assert_on_device(mesh.points, device)
        assert_on_device(mesh.cells, device)

        # Convert back to PyVista (should move to CPU automatically)
        pv_reconstructed = to_pyvista(mesh)

        # Verify geometry is preserved
        assert pv_reconstructed.n_points == pv_original.n_points
        assert pv_reconstructed.n_cells == pv_original.n_cells

        # Points should match (after moving to CPU)
        expected_points = mesh.points.cpu().numpy()
        assert np.allclose(pv_reconstructed.points, expected_points)

    def test_round_trip_spline_device_parametrized(self, device):
        """Test round-trip with spline (polyline → segments conversion)."""
        pv_original = pv.examples.load_spline()

        # Convert to Mesh
        mesh_cpu = from_pyvista(pv_original)
        mesh = Mesh(
            points=mesh_cpu.points.to(device),
            cells=mesh_cpu.cells.to(device),
        )

        # Verify device
        assert_on_device(mesh.points, device)
        assert_on_device(mesh.cells, device)

        # Convert back
        pv_reconstructed = to_pyvista(mesh)

        # Points should be preserved
        assert pv_reconstructed.n_points == pv_original.n_points, (
            f"Point count mismatch: {pv_reconstructed.n_points=} != {pv_original.n_points=}"
        )

        # Verify points match
        assert np.allclose(pv_reconstructed.points, pv_original.points), (
            "Points should be preserved in round-trip"
        )

        # Cell count: PyVista polyline (1 cell) → physicsnemo.mesh segments (N-1 cells) → PyVista lines (N-1 cells)
        # This is expected: physicsnemo.mesh represents 1D manifolds as individual 1-simplices
        expected_n_cells = pv_original.n_points - 1
        assert pv_reconstructed.n_lines == expected_n_cells, (
            f"Expected {expected_n_cells} line segments, got {pv_reconstructed.n_lines}"
        )

        # The reconstructed mesh should have line segments, not a polyline
        assert mesh.n_manifold_dims == 1, "Should be 1D manifold (edges)"
        assert mesh.n_cells == expected_n_cells, (
            f"Mesh should have {expected_n_cells} line segments"
        )

    def test_device_transfer_preserves_data(self, device):
        """Test that device transfer preserves all data."""
        # Create mesh with data
        pv_mesh = pv.Sphere(theta_resolution=5, phi_resolution=5)
        pv_mesh.point_data["temp"] = np.random.rand(pv_mesh.n_points).astype(np.float32)
        pv_mesh.cell_data["pressure"] = np.random.rand(pv_mesh.n_cells).astype(
            np.float32
        )

        # Convert and transfer to device
        mesh_cpu = from_pyvista(pv_mesh)
        mesh = Mesh(
            points=mesh_cpu.points.to(device),
            cells=mesh_cpu.cells.to(device),
            point_data=mesh_cpu.point_data,
            cell_data=mesh_cpu.cell_data,
        )

        # Verify data is on correct device
        assert_on_device(mesh.points, device)
        assert_on_device(mesh.cells, device)

        # Convert back
        pv_reconstructed = to_pyvista(mesh)

        # Verify data values preserved
        assert "temp" in pv_reconstructed.point_data
        assert "pressure" in pv_reconstructed.cell_data

        expected_temp = mesh.point_data["temp"].cpu().numpy()
        expected_pressure = mesh.cell_data["pressure"].cpu().numpy()

        assert np.allclose(pv_reconstructed.point_data["temp"], expected_temp)
        assert np.allclose(pv_reconstructed.cell_data["pressure"], expected_pressure)
