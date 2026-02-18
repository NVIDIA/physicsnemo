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

"""Tests for physicsnemo.mesh.io module - data array shapes."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestDataArrayShapes:
    """Tests for various data array shapes (scalars, vectors, matrices, tensors)."""

    def test_scalar_data(self):
        """Test scalar data (1D array per point/cell)."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        # Add scalar data
        point_scalars = np.random.rand(pv_mesh.n_points).astype(np.float32)
        cell_scalars = np.random.rand(pv_mesh.n_cells).astype(np.float32)

        pv_mesh.point_data["temperature"] = point_scalars
        pv_mesh.cell_data["pressure"] = cell_scalars

        mesh = from_pyvista(pv_mesh)

        # Verify scalar data
        assert "temperature" in mesh.point_data
        assert "pressure" in mesh.cell_data
        temp_tensor = mesh.point_data["temperature"]
        assert isinstance(temp_tensor, torch.Tensor)
        assert temp_tensor.shape == (mesh.n_points,)
        assert mesh.cell_data["pressure"].shape == (mesh.n_cells,)
        assert torch.allclose(temp_tensor, torch.from_numpy(point_scalars), atol=1e-6)

    def test_vector_data(self):
        """Test vector data (Nx3 arrays)."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        # Add vector data
        point_vectors = np.random.rand(pv_mesh.n_points, 3).astype(np.float32)
        cell_vectors = np.random.rand(pv_mesh.n_cells, 3).astype(np.float32)

        pv_mesh.point_data["velocity"] = point_vectors
        pv_mesh.cell_data["gradient"] = cell_vectors

        mesh = from_pyvista(pv_mesh)

        # Verify vector data
        assert "velocity" in mesh.point_data
        assert "gradient" in mesh.cell_data
        vel_tensor = mesh.point_data["velocity"]
        assert isinstance(vel_tensor, torch.Tensor)
        assert vel_tensor.shape == (mesh.n_points, 3)
        assert mesh.cell_data["gradient"].shape == (mesh.n_cells, 3)
        assert torch.allclose(vel_tensor, torch.from_numpy(point_vectors), atol=1e-6)

    def test_matrix_data(self):
        """Test matrix/tensor data with 2D arrays (Nx9 for 3x3 tensors).

        NOTE: PyVista only accepts arrays with dimensionality ≤ 2.
        For higher-dimensional data like 3x3 stress tensors, you must
        flatten them to (n, 9) before adding to PyVista.
        """
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        # For tensor data, must be pre-flattened to 2D
        # E.g., 3x3 stress tensor becomes (n, 9) array
        point_tensors_flat = np.random.rand(pv_mesh.n_points, 9).astype(np.float32)
        cell_tensors_flat = np.random.rand(pv_mesh.n_cells, 9).astype(np.float32)

        pv_mesh.point_data["stress"] = point_tensors_flat
        pv_mesh.cell_data["strain"] = cell_tensors_flat

        mesh = from_pyvista(pv_mesh)

        # Verify tensor data is preserved
        assert "stress" in mesh.point_data
        assert "strain" in mesh.cell_data
        stress_tensor = mesh.point_data["stress"]
        assert isinstance(stress_tensor, torch.Tensor)
        assert stress_tensor.shape == (mesh.n_points, 9)
        assert mesh.cell_data["strain"].shape == (mesh.n_cells, 9)

        # Verify values match
        assert torch.allclose(
            stress_tensor, torch.from_numpy(point_tensors_flat), atol=1e-6
        )

    def test_large_2d_array_data(self):
        """Test large 2D arrays (e.g., flattened higher-order tensors).

        NOTE: PyVista only accepts arrays with dimensionality ≤ 2.
        Higher-order tensors must be pre-flattened before adding to PyVista.
        """
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        # For higher-dimensional data, flatten to 2D before adding to PyVista
        # E.g., a 2x3x4 tensor flattened to 24 components
        point_24d = np.random.rand(pv_mesh.n_points, 24).astype(np.float32)
        cell_10d = np.random.rand(pv_mesh.n_cells, 10).astype(np.float32)

        pv_mesh.point_data["tensor_24"] = point_24d
        pv_mesh.cell_data["tensor_10"] = cell_10d

        mesh = from_pyvista(pv_mesh)

        # Verify large 2D arrays are preserved
        assert "tensor_24" in mesh.point_data
        assert "tensor_10" in mesh.cell_data
        tensor_24_result = mesh.point_data["tensor_24"]
        assert isinstance(tensor_24_result, torch.Tensor)
        assert tensor_24_result.shape == (mesh.n_points, 24)
        assert mesh.cell_data["tensor_10"].shape == (mesh.n_cells, 10)
        assert torch.allclose(tensor_24_result, torch.from_numpy(point_24d), atol=1e-6)

    def test_mixed_data_types(self):
        """Test mesh with multiple data arrays of different shapes and types."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        # Clear default data to have a clean slate
        pv_mesh.clear_data()

        # Add various data types (PyVista only accepts arrays with dim ≤ 2)
        pv_mesh.point_data["scalars"] = np.random.rand(pv_mesh.n_points).astype(
            np.float32
        )
        pv_mesh.point_data["vectors"] = np.random.rand(pv_mesh.n_points, 3).astype(
            np.float32
        )
        pv_mesh.point_data["tensors"] = np.random.rand(pv_mesh.n_points, 9).astype(
            np.float32
        )
        pv_mesh.point_data["int_labels"] = np.random.randint(
            0, 10, pv_mesh.n_points, dtype=np.int32
        )

        pv_mesh.cell_data["cell_scalars"] = np.random.rand(pv_mesh.n_cells).astype(
            np.float32
        )
        pv_mesh.cell_data["cell_vectors"] = np.random.rand(pv_mesh.n_cells, 3).astype(
            np.float32
        )

        pv_mesh.field_data["global_int"] = np.array([42], dtype=np.int64)
        pv_mesh.field_data["global_vec"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        mesh = from_pyvista(pv_mesh)

        # Verify all data types are preserved
        assert len(mesh.point_data.keys()) == 4
        assert len(mesh.cell_data.keys()) == 2
        assert len(mesh.global_data.keys()) == 2

        # Verify dtypes are preserved
        assert mesh.point_data["scalars"].dtype == torch.float32
        assert mesh.point_data["int_labels"].dtype == torch.int32
        assert mesh.global_data["global_int"].dtype == torch.int64

        # Verify shapes
        assert mesh.point_data["scalars"].shape == (mesh.n_points,)
        assert mesh.point_data["vectors"].shape == (mesh.n_points, 3)
        assert mesh.point_data["tensors"].shape == (mesh.n_points, 9)

    def test_empty_data_arrays(self):
        """Test mesh with no attached data arrays."""
        pv_mesh = pv.Sphere()

        # Clear any default data
        pv_mesh.clear_data()

        mesh = from_pyvista(pv_mesh)

        # Should have empty data dicts
        assert len(mesh.point_data.keys()) == 0
        assert len(mesh.cell_data.keys()) == 0
        assert len(mesh.global_data.keys()) == 0
