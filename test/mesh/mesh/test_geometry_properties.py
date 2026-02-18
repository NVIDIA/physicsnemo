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

"""Tests for Mesh geometry properties against PyVista reference implementations.

Validates cell_centroids, cell_areas, cell_normals, and point_normals by comparing
against PyVista's compute_cell_sizes and compute_normals methods.
"""

import pytest

pytest.importorskip("pyvista")

import numpy as np
import torch

from physicsnemo.mesh.io.io_pyvista import to_pyvista
from physicsnemo.mesh.primitives.pyvista_datasets import bunny
from physicsnemo.mesh.primitives.volumes import sphere_volume

### Constants ###

ATOL = 1e-4
RTOL = 1e-4


### Helper Functions ###


def assert_normals_equal(
    mesh_normals: torch.Tensor,
    pv_normals: np.ndarray,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """Assert normals match, allowing for sign flips.

    Performs two checks:
    1. Component-wise equality allowing sign flip (abs values match)
    2. Alignment via dot product (should be ±1 for unit vectors)

    Isolated vertices (with zero-length normals) are handled separately - both
    implementations should return zero vectors for these.

    Parameters
    ----------
    mesh_normals : torch.Tensor
        Normals from Mesh, shape (n, 3).
    pv_normals : np.ndarray
        Normals from PyVista, shape (n, 3).
    atol : float
        Absolute tolerance for comparison.
    rtol : float
        Relative tolerance for comparison.
    """
    pv_tensor = torch.from_numpy(pv_normals).float()

    ### Identify isolated vertices (zero-length normals in both)
    mesh_norms = mesh_normals.norm(dim=-1)
    pv_norms = pv_tensor.norm(dim=-1)
    isolated_mask = (mesh_norms < atol) & (pv_norms < atol)
    connected_mask = ~isolated_mask

    ### Check that isolated vertices have matching zero normals
    if isolated_mask.any():
        assert torch.allclose(
            mesh_normals[isolated_mask], pv_tensor[isolated_mask], atol=atol, rtol=rtol
        ), "Isolated vertex normals should both be zero vectors"

    ### For connected vertices, check alignment
    if connected_mask.any():
        mesh_connected = mesh_normals[connected_mask]
        pv_connected = pv_tensor[connected_mask]

        ### Check 1: Component-wise equality allowing sign flip
        assert torch.allclose(
            mesh_connected.abs(), pv_connected.abs(), atol=atol, rtol=rtol
        ), (
            f"Normal magnitudes differ.\n"
            f"Max abs diff: {(mesh_connected.abs() - pv_connected.abs()).abs().max()}"
        )

        ### Check 2: Alignment via dot product (should be ±1 for unit vectors)
        dot_products = (mesh_connected * pv_connected).sum(dim=-1)
        assert torch.allclose(
            dot_products.abs(), torch.ones_like(dot_products), atol=atol, rtol=rtol
        ), (
            f"Normals not aligned.\n"
            f"Min |dot|: {dot_products.abs().min()}, Max |dot|: {dot_products.abs().max()}"
        )


### Test Classes ###


class TestCellCentroids:
    """Tests for Mesh.cell_centroids property."""

    def test_2d_manifold_bunny(self):
        """Test cell centroids on 2D manifold (triangular surface mesh)."""
        ### Load bunny mesh and convert to PyVista
        mesh = bunny.load()
        pv_mesh = to_pyvista(mesh)

        ### Compute centroids with both implementations
        mesh_centroids = mesh.cell_centroids  # shape: (n_cells, 3)
        pv_centroids = pv_mesh.cell_centers().points  # shape: (n_cells, 3)

        ### Compare results
        pv_tensor = torch.from_numpy(pv_centroids).float()
        assert torch.allclose(mesh_centroids, pv_tensor, atol=ATOL, rtol=RTOL), (
            f"Cell centroids differ for bunny mesh.\n"
            f"Max diff: {(mesh_centroids - pv_tensor).abs().max()}"
        )

    def test_3d_manifold_sphere_volume(self):
        """Test cell centroids on 3D manifold (tetrahedral volume mesh)."""
        ### Load sphere volume mesh and convert to PyVista
        mesh = sphere_volume.load()
        pv_mesh = to_pyvista(mesh)

        ### Compute centroids with both implementations
        mesh_centroids = mesh.cell_centroids  # shape: (n_cells, 3)
        pv_centroids = pv_mesh.cell_centers().points  # shape: (n_cells, 3)

        ### Compare results
        pv_tensor = torch.from_numpy(pv_centroids).float()
        assert torch.allclose(mesh_centroids, pv_tensor, atol=ATOL, rtol=RTOL), (
            f"Cell centroids differ for sphere volume mesh.\n"
            f"Max diff: {(mesh_centroids - pv_tensor).abs().max()}"
        )


class TestCellAreas:
    """Tests for Mesh.cell_areas property."""

    def test_2d_manifold_bunny(self):
        """Test cell areas on 2D manifold (triangular surface mesh)."""
        ### Load bunny mesh and convert to PyVista
        mesh = bunny.load()
        pv_mesh = to_pyvista(mesh)

        ### Compute areas with both implementations
        mesh_areas = mesh.cell_areas  # shape: (n_cells,)
        pv_sized = pv_mesh.compute_cell_sizes(area=True, volume=False)
        pv_areas = pv_sized.cell_data["Area"]  # shape: (n_cells,)

        ### Compare results
        pv_tensor = torch.from_numpy(pv_areas).float()
        assert torch.allclose(mesh_areas, pv_tensor, atol=ATOL, rtol=RTOL), (
            f"Cell areas differ for bunny mesh.\n"
            f"Max diff: {(mesh_areas - pv_tensor).abs().max()}"
        )

    def test_3d_manifold_sphere_volume(self):
        """Test cell volumes on 3D manifold (tetrahedral volume mesh).

        Note: For 3D manifolds, cell_areas returns the volume of each tetrahedron.
        """
        ### Load sphere volume mesh and convert to PyVista
        mesh = sphere_volume.load()
        pv_mesh = to_pyvista(mesh)

        ### Compute volumes with both implementations
        mesh_volumes = mesh.cell_areas  # shape: (n_cells,)
        pv_sized = pv_mesh.compute_cell_sizes(area=False, volume=True)
        pv_volumes = pv_sized.cell_data["Volume"]  # shape: (n_cells,)

        ### Compare results
        pv_tensor = torch.from_numpy(pv_volumes).float()
        assert torch.allclose(mesh_volumes, pv_tensor, atol=ATOL, rtol=RTOL), (
            f"Cell volumes differ for sphere volume mesh.\n"
            f"Max diff: {(mesh_volumes - pv_tensor).abs().max()}"
        )


class TestCellNormals:
    """Tests for Mesh.cell_normals property."""

    def test_2d_manifold_bunny(self):
        """Test cell normals on 2D manifold (triangular surface mesh).

        Cell normals are only defined for codimension-1 manifolds (e.g., triangles in 3D).
        """
        ### Load bunny mesh and convert to PyVista
        mesh = bunny.load()
        pv_mesh = to_pyvista(mesh)

        ### Compute normals with both implementations
        mesh_normals = mesh.cell_normals  # shape: (n_cells, 3)
        pv_normed = pv_mesh.compute_normals(cell_normals=True, point_normals=False)
        pv_normals = pv_normed.cell_data["Normals"]  # shape: (n_cells, 3)

        ### Compare results (allowing for sign flips)
        assert_normals_equal(mesh_normals, pv_normals)


class TestPointNormals:
    """Tests for Mesh.point_normals property and compute_point_normals method.

    Mesh supports four weighting schemes for point normals:
    - "area": Area-weighted averaging (larger faces have more influence) - default
    - "unweighted": Simple averaging (equal weight per face, matches PyVista/VTK)
    - "angle": Angle-weighted averaging (weight by interior angle at vertex)
    - "angle_area": Combined angle and area weighting (Maya default)

    The point_normals property returns area-weighted normals (canonical default).
    The compute_point_normals() method allows explicit weighting selection.

    Tests use weighting="unweighted" to match PyVista/VTK's compute_normals behavior.
    """

    def test_2d_manifold_bunny(self):
        """Test point normals on 2D manifold (triangular surface mesh).

        Uses unweighted averaging to match PyVista/VTK behavior.
        """
        ### Load bunny mesh and convert to PyVista
        mesh = bunny.load()
        pv_mesh = to_pyvista(mesh)

        ### Compute normals with both implementations
        # Use compute_point_normals with weighting="unweighted" to match PyVista/VTK
        mesh_normals = mesh.compute_point_normals(
            weighting="unweighted"
        )  # (n_points, 3)
        pv_normed = pv_mesh.compute_normals(cell_normals=False, point_normals=True)
        pv_normals = pv_normed.point_data["Normals"]  # shape: (n_points, 3)

        ### Compare results (allowing for sign flips)
        assert_normals_equal(mesh_normals, pv_normals)
