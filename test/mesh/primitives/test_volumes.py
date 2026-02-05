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

"""Tests for volume example meshes."""

import pytest

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.mesh import primitives

# Volume primitives that don't require pyvista
PYVISTA_FREE_VOLUMES = ["cube_volume", "tetrahedron_volume"]

# Volume primitives that require pyvista for delaunay_3d
PYVISTA_VOLUMES = ["sphere_volume", "cylinder_volume"]

requires_pyvista = pytest.mark.skipif(
    not check_version_spec("pyvista"),
    reason="pyvista is required for delaunay-based volume meshes",
)


class TestVolumePrimitives:
    """Test all volume example meshes (3D→3D)."""

    @pytest.mark.parametrize("example_name", PYVISTA_FREE_VOLUMES)
    def test_volume_mesh_pyvista_free(self, example_name):
        """Test volume meshes that don't require pyvista."""
        primitives_module = getattr(primitives.volumes, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    @requires_pyvista
    @pytest.mark.parametrize("example_name", PYVISTA_VOLUMES)
    def test_volume_mesh_pyvista(self, example_name):
        """Test volume meshes that require pyvista."""
        primitives_module = getattr(primitives.volumes, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_cube_volume_subdivision(self):
        """Test cube volume subdivision."""
        cube_coarse = primitives.volumes.cube_volume.load(n_subdivisions=2)
        cube_fine = primitives.volumes.cube_volume.load(n_subdivisions=4)

        assert cube_fine.n_cells > cube_coarse.n_cells

    def test_tetrahedron_single_cell(self):
        """Test that single tetrahedron has exactly one cell."""
        tet = primitives.volumes.tetrahedron_volume.load()

        assert tet.n_cells == 1
        assert tet.n_points == 4
        assert tet.cells.shape == (1, 4)

    @requires_pyvista
    @pytest.mark.parametrize("example_name", PYVISTA_VOLUMES)
    def test_delaunay_volumes(self, example_name):
        """Test delaunay-based volume meshes."""
        primitives_module = getattr(primitives.volumes, example_name)
        mesh = primitives_module.load(resolution=15)

        # Should have reasonable number of cells
        assert mesh.n_cells > 10
