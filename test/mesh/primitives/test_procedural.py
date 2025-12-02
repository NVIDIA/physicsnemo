"""Tests for procedural example meshes."""

import torch

from physicsnemo.mesh import primitives


class TestProceduralPrimitives:
    """Test all procedural mesh generators."""

    def test_lumpy_sphere(self):
        """Test lumpy sphere generation."""
        mesh = primitives.procedural.lumpy_sphere.load(
            radius=1.0,
            subdivisions=2,
            noise_amplitude=0.1,
            seed=42,
        )

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.n_cells > 0

    def test_lumpy_sphere_reproducibility(self):
        """Test that lumpy sphere is reproducible with same seed."""
        mesh1 = primitives.procedural.lumpy_sphere.load(seed=42, subdivisions=1)
        mesh2 = primitives.procedural.lumpy_sphere.load(seed=42, subdivisions=1)

        assert torch.allclose(mesh1.points, mesh2.points)

    def test_lumpy_sphere_different_seeds(self):
        """Test that lumpy sphere changes with different seeds."""
        mesh1 = primitives.procedural.lumpy_sphere.load(seed=1, subdivisions=1)
        mesh2 = primitives.procedural.lumpy_sphere.load(seed=2, subdivisions=1)

        # Topology should be same
        assert mesh1.n_points == mesh2.n_points
        assert mesh1.n_cells == mesh2.n_cells

        # But geometry should differ
        assert not torch.allclose(mesh1.points, mesh2.points)

    def test_noisy_mesh(self):
        """Test generic noisy mesh function."""
        base_mesh = primitives.surfaces.sphere_icosahedral.load(subdivisions=1)
        noisy_mesh = primitives.procedural.noisy_mesh.load(
            base_mesh=base_mesh,
            noise_scale=0.05,
            seed=42,
        )

        # Topology should be preserved
        assert noisy_mesh.n_points == base_mesh.n_points
        assert noisy_mesh.n_cells == base_mesh.n_cells
        assert torch.equal(noisy_mesh.cells, base_mesh.cells)

        # Geometry should differ
        assert not torch.allclose(noisy_mesh.points, base_mesh.points)

    def test_perturbed_grid(self):
        """Test perturbed grid generation."""
        mesh = primitives.procedural.perturbed_grid.load(
            n_x=5,
            n_y=5,
            perturbation_scale=0.05,
            seed=42,
        )

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 2

        # Check that boundary points are not perturbed
        x_coords = mesh.points[:, 0]
        y_coords = mesh.points[:, 1]

        # Find corner points
        corners = (
            (torch.abs(x_coords) < 1e-6) | (torch.abs(x_coords - 1.0) < 1e-6)
        ) & ((torch.abs(y_coords) < 1e-6) | (torch.abs(y_coords - 1.0) < 1e-6))

        # Corners should still be at expected positions
        assert corners.sum() >= 4
