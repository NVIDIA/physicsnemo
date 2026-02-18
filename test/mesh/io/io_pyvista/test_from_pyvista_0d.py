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

"""Tests for physicsnemo.mesh.io module - 0D mesh conversion."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestFromPyvista0D:
    """Tests for converting 0D (point cloud) meshes."""

    def test_pointset_auto_detection(self):
        """Test automatic detection of 0D manifold from PointSet."""
        points = np.random.rand(100, 3).astype(np.float32)
        pv_mesh = pv.PointSet(points)

        # Verify it's just points (no connectivity)
        assert pv_mesh.n_points == 100

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points == 100
        assert mesh.n_cells == 0
        assert mesh.cells.shape == (0, 1)

        # Verify points are preserved correctly
        assert torch.allclose(mesh.points, torch.from_numpy(points).float(), atol=1e-6)

    def test_pointset_explicit_dim(self):
        """Test explicit manifold_dim specification for point cloud."""
        points = np.random.rand(50, 3).astype(np.float32)
        pv_mesh = pv.PointSet(points)

        mesh = from_pyvista(pv_mesh, manifold_dim=0)

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 50
        assert mesh.cells.shape == (0, 1)

    def test_polydata_points_only(self):
        """Test PolyData with only points (no lines or cells).

        PolyData can represent point clouds using vertex cells.
        """
        points = np.random.rand(25, 3).astype(np.float32)
        pv_mesh = pv.PolyData(points)

        # Verify it has vertex cells but no lines or polygon cells
        assert pv_mesh.n_verts == 25
        assert pv_mesh.n_lines == 0

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 25
        assert mesh.n_cells == 0
        assert mesh.cells.shape == (0, 1)
