"""Tests for procedural example meshes."""

import torch

from physicsnemo.mesh import primitives


class TestProceduralPrimitives:
    """Test all procedural mesh generators."""

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
