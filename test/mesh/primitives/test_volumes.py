# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import torch

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.mesh import primitives
from physicsnemo.mesh.primitives.procedural import lumpy_ball

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


class TestLumpyBall:
    """Tests for the lumpy_ball procedural volume primitive."""

    def test_basic_instantiation(self):
        """Test basic lumpy_ball creation."""
        mesh = lumpy_ball.load()

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_manifold_dimensions(self):
        """Test that lumpy_ball is a 3D manifold in 3D space."""
        mesh = lumpy_ball.load(n_shells=2, subdivisions=1)

        assert mesh.n_manifold_dims == 3, "Should be 3D manifold (tetrahedra)"
        assert mesh.n_spatial_dims == 3, "Should be in 3D space"
        assert mesh.cells.shape[1] == 4, "Cells should be tetrahedra (4 vertices)"

    @pytest.mark.parametrize(
        "n_shells,subdivisions,expected_cells",
        [
            (1, 0, 20),  # 20 faces * (3*1 - 2) = 20 * 1 = 20
            (2, 0, 80),  # 20 faces * (3*2 - 2) = 20 * 4 = 80
            (2, 1, 320),  # 80 faces * (3*2 - 2) = 80 * 4 = 320
            (3, 1, 560),  # 80 faces * (3*3 - 2) = 80 * 7 = 560
            (3, 2, 2240),  # 320 faces * (3*3 - 2) = 320 * 7 = 2240
        ],
    )
    def test_cell_count_formula(self, n_shells, subdivisions, expected_cells):
        """Verify cell count matches formula: n_faces * (3*n_shells - 2)."""
        mesh = lumpy_ball.load(n_shells=n_shells, subdivisions=subdivisions)

        assert mesh.n_cells == expected_cells, (
            f"Expected {expected_cells} cells for n_shells={n_shells}, "
            f"subdivisions={subdivisions}, got {mesh.n_cells}"
        )

    def test_resolution_scaling(self):
        """Test that more shells/subdivisions = more cells."""
        mesh_coarse = lumpy_ball.load(n_shells=2, subdivisions=1)
        mesh_fine_shells = lumpy_ball.load(n_shells=4, subdivisions=1)
        mesh_fine_subdiv = lumpy_ball.load(n_shells=2, subdivisions=2)

        assert mesh_fine_shells.n_cells > mesh_coarse.n_cells
        assert mesh_fine_subdiv.n_cells > mesh_coarse.n_cells

    def test_noise_reproducibility(self):
        """Test that same seed produces same mesh."""
        mesh1 = lumpy_ball.load(noise_amplitude=0.3, seed=42)
        mesh2 = lumpy_ball.load(noise_amplitude=0.3, seed=42)
        mesh3 = lumpy_ball.load(noise_amplitude=0.3, seed=123)

        # Same seed should produce identical points
        assert torch.allclose(mesh1.points, mesh2.points)
        # Different seed should produce different points
        assert not torch.allclose(mesh1.points, mesh3.points)

    def test_noise_amplitude_effect(self):
        """Test that noise amplitude affects vertex positions."""
        mesh_no_noise = lumpy_ball.load(noise_amplitude=0.0, seed=42)
        mesh_with_noise = lumpy_ball.load(noise_amplitude=0.5, seed=42)

        # With noise, points should differ from no-noise version
        assert not torch.allclose(mesh_no_noise.points, mesh_with_noise.points)

        # No-noise mesh should have points approximately on sphere shells
        # (center point at origin, shell points at expected radii)
        assert torch.allclose(
            mesh_no_noise.points[0], torch.zeros(3), atol=1e-6
        ), "Center point should be at origin"

    def test_center_point(self):
        """Test that center point is at origin."""
        mesh = lumpy_ball.load(noise_amplitude=0.3, seed=42)

        # Even with noise, center point (index 0) should be at origin
        assert torch.allclose(mesh.points[0], torch.zeros(3), atol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device(self, device):
        """Test lumpy_ball on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mesh = lumpy_ball.load(device=device)

        assert mesh.points.device.type == device
        assert mesh.cells.device.type == device

    def test_validation_errors(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="radius must be positive"):
            lumpy_ball.load(radius=-1.0)

        with pytest.raises(ValueError, match="n_shells must be at least 1"):
            lumpy_ball.load(n_shells=0)

        with pytest.raises(ValueError, match="subdivisions must be non-negative"):
            lumpy_ball.load(subdivisions=-1)

        with pytest.raises(ValueError, match="noise_amplitude must be non-negative"):
            lumpy_ball.load(noise_amplitude=-0.1)

    @pytest.mark.parametrize("noise_amplitude", [0.3, 0.5, 0.8])
    def test_high_noise_valid_tetrahedra(self, noise_amplitude):
        """Test that high noise doesn't create degenerate tetrahedra.

        With correlated noise across shells (all shells are scaled versions
        of the same noisy shape), tetrahedra should remain valid regardless
        of noise amplitude.
        """
        mesh = lumpy_ball.load(
            noise_amplitude=noise_amplitude, seed=42, n_shells=3, subdivisions=1
        )

        # Compute signed tetrahedron volumes using scalar triple product.
        # Note: mesh.cell_areas uses Gram determinant which gives unsigned values.
        # For orientation checking, we need signed volumes:
        #   V = (1/6) * (b-a) · ((c-a) × (d-a))
        # Positive V means vertices (a,b,c,d) have consistent right-hand orientation.
        tet_verts = mesh.points[mesh.cells]  # (n_cells, 4, 3)
        a, b, c, d = tet_verts[:, 0], tet_verts[:, 1], tet_verts[:, 2], tet_verts[:, 3]
        ab, ac, ad = b - a, c - a, d - a
        signed_volumes = torch.einsum("ij,ij->i", ab, torch.cross(ac, ad, dim=1)) / 6.0

        # All tetrahedra should have consistent orientation (same sign)
        # If any have opposite sign, the mesh has inverted cells
        assert torch.all(signed_volumes > 0), (
            f"Some tetrahedra have negative volume (inverted) with "
            f"noise_amplitude={noise_amplitude}. "
            f"Min volume: {signed_volumes.min().item():.6f}"
        )
