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

"""Tests for physicsnemo.mesh.io module - mesh equivalence."""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestMeshEquivalence:
    """Tests that converted meshes are equivalent to direct construction."""

    def test_airplane_equivalence(self):
        """Test that from_pyvista produces same result as direct construction."""
        pv_mesh = pv.examples.load_airplane()

        # Using from_pyvista
        mesh_from_pv = from_pyvista(pv_mesh)

        # Direct construction (as in primitives.py)
        mesh_direct = Mesh(
            points=pv_mesh.points,
            cells=pv_mesh.regular_faces,
            point_data=pv_mesh.point_data,
            cell_data=pv_mesh.cell_data,
            global_data=pv_mesh.field_data,
        )

        assert torch.equal(mesh_from_pv.points, mesh_direct.points)
        assert torch.equal(mesh_from_pv.cells, mesh_direct.cells)

    def test_tetbeam_equivalence(self):
        """Test that from_pyvista produces same result as direct construction for tetbeam."""
        pv_mesh = pv.examples.load_tetbeam()

        # Using from_pyvista
        mesh_from_pv = from_pyvista(pv_mesh)

        # Direct construction (as in primitives.py)
        mesh_direct = Mesh(
            points=pv_mesh.points,
            cells=pv_mesh.cells_dict[pv.CellType.TETRA],
            point_data=pv_mesh.point_data,
            cell_data=pv_mesh.cell_data,
            global_data=pv_mesh.field_data,
        )

        assert torch.equal(mesh_from_pv.points, mesh_direct.points)
        assert torch.equal(mesh_from_pv.cells, mesh_direct.cells)
