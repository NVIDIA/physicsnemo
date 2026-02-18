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

"""Tests for physicsnemo.mesh.io module - 3D mesh conversion."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestFromPyvista3D:
    """Tests for converting 3D (volume) meshes."""

    def test_tetbeam_mesh_auto_detection(self):
        """Test automatic detection of 3D manifold from tetbeam mesh."""
        pv_mesh = pv.examples.load_tetbeam()

        # Verify it's all tetrahedral cells
        assert list(pv_mesh.cells_dict.keys()) == [pv.CellType.TETRA]

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4  # Tetrahedral cells
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_cells == pv_mesh.n_cells

    def test_tetbeam_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification for 3D mesh."""
        pv_mesh = pv.examples.load_tetbeam()

        mesh = from_pyvista(pv_mesh, manifold_dim=3)

        assert mesh.n_manifold_dims == 3
        assert mesh.cells.shape[1] == 4

    def test_hexbeam_mesh_tessellation(self):
        """Test automatic tessellation of hexahedral mesh to tetrahedral.

        The hexbeam mesh contains hexahedral cells which must be converted
        to tetrahedral cells for our simplex-based mesh representation.
        """
        pv_mesh = pv.examples.load_hexbeam()

        # Verify it contains hexahedral cells (not tetrahedral)
        assert pv.CellType.HEXAHEDRON in pv_mesh.cells_dict
        assert pv.CellType.TETRA not in pv_mesh.cells_dict
        original_n_points = pv_mesh.n_points

        # Convert - should automatically tessellate
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4  # Tetrahedral cells after tessellation
        # Tessellation may add points at cell centers
        assert mesh.n_points >= original_n_points
        # Each hexahedron is tessellated into at least 5 tetrahedra
        assert mesh.n_cells >= 5 * pv_mesh.n_cells

    def test_simple_tetrahedron(self):
        """Test conversion of a single tetrahedron."""
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.TETRA])

        pv_mesh = pv.UnstructuredGrid(cells, celltypes, points)
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_points == 4
        assert mesh.n_cells == 1
        assert mesh.cells.shape == (1, 4)

        # Verify the face connectivity is correct
        expected_cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        assert torch.equal(mesh.cells, expected_cells)
