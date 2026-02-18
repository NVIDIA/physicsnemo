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

"""Tests for physicsnemo.mesh.io module - 2D mesh conversion."""

import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestFromPyvista2D:
    """Tests for converting 2D (surface) meshes."""

    def test_airplane_mesh_auto_detection(self):
        """Test automatic detection of 2D manifold from airplane mesh."""
        pv_mesh = pv.examples.load_airplane()

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3  # Triangular cells
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_cells == pv_mesh.n_cells
        assert mesh.points.dtype == torch.float32
        assert mesh.cells.dtype == torch.long

    def test_airplane_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification."""
        pv_mesh = pv.examples.load_airplane()

        mesh = from_pyvista(pv_mesh, manifold_dim=2)

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3

    def test_sphere_mesh(self):
        """Test conversion of sphere mesh."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.cells.shape[1] == 3

    def test_automatic_triangulation(self):
        """Test that non-triangular meshes are automatically triangulated."""
        # Create a plane with quad cells
        pv_mesh = pv.Plane(i_resolution=2, j_resolution=2)
        assert not pv_mesh.is_all_triangles

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        # Should be automatically triangulated
        assert mesh.cells.shape[1] == 3
        assert mesh.n_manifold_dims == 2
