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

"""Tests for physicsnemo.mesh.io module - data preservation.

Tests validate that all data (point_data, cell_data, field_data) is correctly
preserved during PyVista → physicsnemo.mesh conversion across backends.
"""

import numpy as np
import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402

### Helper Functions ###


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


class TestDataPreservation:
    """Tests for preserving point_data, cell_data, and field_data."""

    def test_point_data_preserved(self):
        """Test that point_data is preserved during conversion."""
        pv_mesh = pv.Sphere()

        # Explicitly create point data
        scalars_data = np.random.rand(pv_mesh.n_points).astype(np.float32)
        vectors_data = np.random.rand(pv_mesh.n_points, 3).astype(np.float32)
        pv_mesh.point_data["scalars"] = scalars_data
        pv_mesh.point_data["vectors"] = vectors_data

        mesh = from_pyvista(pv_mesh)

        # Verify data is preserved
        assert "scalars" in mesh.point_data
        assert "vectors" in mesh.point_data
        assert mesh.point_data["scalars"].shape == (pv_mesh.n_points,)
        assert mesh.point_data["vectors"].shape == (pv_mesh.n_points, 3)
        assert isinstance(mesh.point_data["scalars"], torch.Tensor)
        assert isinstance(mesh.point_data["vectors"], torch.Tensor)

        # Verify values are correct
        assert torch.allclose(
            mesh.point_data["scalars"], torch.from_numpy(scalars_data), atol=1e-6
        )
        assert torch.allclose(
            mesh.point_data["vectors"], torch.from_numpy(vectors_data), atol=1e-6
        )

    def test_cell_data_preserved(self):
        """Test that cell_data is preserved as cell_data."""
        pv_mesh = pv.Sphere()

        # Explicitly create cell data
        cell_ids_data = np.arange(pv_mesh.n_cells, dtype=np.int64)
        quality_data = np.random.rand(pv_mesh.n_cells).astype(np.float32)
        pv_mesh.cell_data["cell_ids"] = cell_ids_data
        pv_mesh.cell_data["quality"] = quality_data

        mesh = from_pyvista(pv_mesh)

        # Verify data is preserved
        assert "cell_ids" in mesh.cell_data
        assert "quality" in mesh.cell_data
        assert mesh.cell_data["cell_ids"].shape == (mesh.n_cells,)
        assert mesh.cell_data["quality"].shape == (mesh.n_cells,)
        assert isinstance(mesh.cell_data["cell_ids"], torch.Tensor)
        assert isinstance(mesh.cell_data["quality"], torch.Tensor)

        # Verify values are correct
        assert torch.equal(mesh.cell_data["cell_ids"], torch.from_numpy(cell_ids_data))
        assert torch.allclose(
            mesh.cell_data["quality"], torch.from_numpy(quality_data), atol=1e-6
        )

    def test_field_data_preserved(self):
        """Test that field_data is preserved as global_data."""
        pv_mesh = pv.Sphere()

        # Explicitly create field data
        metadata_data = np.array([42, 123], dtype=np.int32)
        version_data = np.array([1.0], dtype=np.float32)
        pv_mesh.field_data["metadata"] = metadata_data
        pv_mesh.field_data["version"] = version_data

        mesh = from_pyvista(pv_mesh)

        # Verify data is preserved
        assert "metadata" in mesh.global_data
        assert "version" in mesh.global_data
        assert isinstance(mesh.global_data["metadata"], torch.Tensor)
        assert isinstance(mesh.global_data["version"], torch.Tensor)

        # Verify values are correct
        assert torch.equal(
            mesh.global_data["metadata"], torch.from_numpy(metadata_data)
        )
        assert torch.allclose(
            mesh.global_data["version"], torch.from_numpy(version_data), atol=1e-6
        )

    def test_mesh_with_explicit_normals(self):
        """Test that explicitly added normals are preserved.

        Create a mesh and compute normals explicitly, then verify they're preserved.
        """
        pv_mesh = pv.Sphere(theta_resolution=10, phi_resolution=10)

        # Compute and add normals explicitly
        pv_mesh = pv_mesh.compute_normals(point_normals=True, cell_normals=False)

        # Verify normals exist
        assert "Normals" in pv_mesh.point_data
        normals_data = pv_mesh.point_data["Normals"]

        mesh = from_pyvista(pv_mesh)

        # Verify normals are preserved
        assert "Normals" in mesh.point_data
        normals_tensor = mesh.point_data["Normals"]
        assert isinstance(normals_tensor, torch.Tensor)
        assert normals_tensor.shape == (mesh.n_points, 3)
        assert torch.allclose(normals_tensor, torch.from_numpy(normals_data), atol=1e-6)


### Parametrized Tests for Device Handling ###


class TestDataPreservationParametrized:
    """Parametrized tests for data preservation across backends."""

    def test_data_preservation_with_device_transfer(self, device):
        """Test that data is preserved when transferring to different device."""
        pv_mesh = pv.Sphere(theta_resolution=5, phi_resolution=5)
        pv_mesh.point_data["temp"] = np.random.rand(pv_mesh.n_points).astype(np.float32)
        pv_mesh.cell_data["pressure"] = np.random.rand(pv_mesh.n_cells).astype(
            np.float32
        )
        pv_mesh.field_data["time"] = np.array([1.5], dtype=np.float32)

        # Convert to mesh
        mesh_cpu = from_pyvista(pv_mesh)

        # Transfer to device
        mesh = Mesh(
            points=mesh_cpu.points.to(device),
            cells=mesh_cpu.cells.to(device),
            point_data=mesh_cpu.point_data,
            cell_data=mesh_cpu.cell_data,
            global_data=mesh_cpu.global_data,
        )

        # Verify geometry on device
        assert_on_device(mesh.points, device)
        assert_on_device(mesh.cells, device)

        # Verify all data preserved (as CPU tensors in TensorDict)
        assert "temp" in mesh.point_data
        assert "pressure" in mesh.cell_data
        assert "time" in mesh.global_data

        # Values should match original
        assert torch.allclose(
            mesh.point_data["temp"],
            torch.from_numpy(pv_mesh.point_data["temp"]),
            atol=1e-6,
        )
        assert torch.allclose(
            mesh.cell_data["pressure"],
            torch.from_numpy(pv_mesh.cell_data["pressure"]),
            atol=1e-6,
        )
        assert torch.allclose(
            mesh.global_data["time"],
            torch.from_numpy(pv_mesh.field_data["time"]),
            atol=1e-6,
        )
