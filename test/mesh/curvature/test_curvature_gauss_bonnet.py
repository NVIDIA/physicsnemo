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

"""Tests for Gauss-Bonnet theorem and curvature integration convergence.

The Gauss-Bonnet theorem states that for a closed 2D surface M:
    ∫∫_M K dA = 2πχ(M)

where K is Gaussian curvature, dA is area element, and χ(M) is Euler characteristic.

For a sphere (χ=2): ∫∫ K dA = 4π exactly, regardless of:
    - Shape (smooth sphere, lumpy sphere, ellipsoid)
    - Discretization (mesh resolution)
    - Scale (radius)

This is a topological invariant. In the discrete approximation:
    ∫∫ K dA ≈ Σ_i (K_i × A_i)

where K_i is Gaussian curvature at vertex i and A_i is the Voronoi area.
As the mesh is refined, this sum should converge to 4π.
"""

import pytest
import torch

from physicsnemo.mesh.geometry.dual_meshes import (
    compute_dual_volumes_0 as compute_voronoi_areas,
)
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import (
    octahedron_surface,
    sphere_icosahedral,
    tetrahedron_surface,
)

### Helper Functions


def compute_gaussian_curvature_integral(mesh: Mesh) -> torch.Tensor:
    """Compute the discrete integral of Gaussian curvature over the mesh.

    Uses the angle defect formula:
        ∫∫ K dA ≈ Σ_i (K_i × A_i)

    where K_i is Gaussian curvature at vertex i and A_i is the Voronoi area.

    Args:
        mesh: Input mesh (2D manifold)

    Returns:
        Scalar tensor containing the integrated Gaussian curvature
    """
    ### Compute Gaussian curvature at vertices
    K_vertices = mesh.gaussian_curvature_vertices  # (n_points,)

    ### Compute Voronoi areas
    voronoi_areas = compute_voronoi_areas(mesh)  # (n_points,)

    ### Integrate: ∫∫ K dA ≈ Σ K_i * A_i
    total_curvature = (K_vertices * voronoi_areas).sum()

    return total_curvature


### Test Perfect Sphere Convergence


class TestPerfectSphereConvergence:
    """Tests that Gauss-Bonnet theorem holds for perfect spheres with increasing refinement."""

    def test_sphere_gauss_bonnet_convergence(self, device):
        """Test that ∫∫ K dA converges to 4π with subdivision refinement."""
        expected_integral = 4.0 * torch.pi

        integrals = []
        errors = []

        ### Test subdivision levels 0, 1, 2, 3
        for subdivisions in [0, 1, 2, 3]:
            mesh = sphere_icosahedral.load(
                radius=1.0,
                subdivisions=subdivisions,
                device=device,
            )

            integral = compute_gaussian_curvature_integral(mesh)
            error = torch.abs(integral - expected_integral)

            integrals.append(integral.item())
            errors.append(error.item())

        ### Each integral should be close to 4π
        # The Gauss-Bonnet theorem is a topological invariant, so the integral
        # should be very close to 4π at ALL subdivision levels, not just fine ones
        for i, (integral, error) in enumerate(zip(integrals, errors)):
            relative_error = error / expected_integral
            # All levels should be very accurate (topological invariant)
            assert relative_error < 0.002, (
                f"Subdivision level {i}: integral={integral:.6f}, "
                f"expected={expected_integral:.6f}, "
                f"relative_error={relative_error:.1%} exceeds 0.2%"
            )

        ### Verify discretization invariance
        # The integral should be nearly constant across subdivision levels
        # (within numerical precision), not monotonically converging
        max_integral = max(integrals)
        min_integral = min(integrals)
        integral_range = max_integral - min_integral
        relative_variation = integral_range / expected_integral

        assert relative_variation < 0.002, (
            f"Integral variation across subdivision levels too large. "
            f"Min={min_integral:.6f}, Max={max_integral:.6f}, "
            f"Range={integral_range:.6f}, "
            f"Relative variation={relative_variation:.1%} exceeds 0.2%"
        )

    @pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 5.0])
    def test_sphere_gauss_bonnet_scale_invariance(self, device, radius):
        """Test that ∫∫ K dA = 4π regardless of sphere radius (scale invariance)."""
        expected_integral = 4.0 * torch.pi

        ### Create sphere with given radius at moderate refinement
        mesh = sphere_icosahedral.load(
            radius=radius,
            subdivisions=2,
            device=device,
        )

        integral = compute_gaussian_curvature_integral(mesh)
        relative_error = torch.abs(integral - expected_integral) / expected_integral

        ### Should be close to 4π regardless of radius
        assert relative_error < 0.02, (
            f"Scale invariance violated for radius={radius}. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"relative_error={relative_error:.1%} exceeds 2%"
        )

    def test_sphere_gauss_bonnet_absolute_value(self, device):
        """Test that the computed integral is very close to 4π at high refinement."""
        expected_integral = 4.0 * torch.pi

        ### Create highly refined sphere
        mesh = sphere_icosahedral.load(
            radius=1.0,
            subdivisions=3,
            device=device,
        )

        integral = compute_gaussian_curvature_integral(mesh)
        absolute_error = torch.abs(integral - expected_integral)

        ### Should be within tight absolute tolerance
        assert absolute_error < 0.25, (
            f"High-refinement sphere integral far from 4π. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"absolute_error={absolute_error:.6f} exceeds 0.25"
        )

        ### Relative error should be very small
        relative_error = absolute_error / expected_integral
        assert relative_error < 0.02, (
            f"High-refinement sphere relative error too large. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"relative_error={relative_error:.1%} exceeds 2%"
        )


### Test Lumpy Sphere Discretization Invariance


class TestLumpySphereDiscretizationInvariance:
    """Tests that Gauss-Bonnet theorem holds for lumpy spheres across refinement levels."""

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_lumpy_sphere_gauss_bonnet_value(self, device, seed):
        """Test that lumpy sphere has ∫∫ K dA ≈ 4π."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        expected_integral = 4.0 * torch.pi

        ### Create lumpy sphere with moderate perturbation
        mesh = lumpy_sphere.load(
            noise_amplitude=0.2,
            subdivisions=2,
            seed=seed,
            device=device,
        )

        integral = compute_gaussian_curvature_integral(mesh)
        relative_error = torch.abs(integral - expected_integral) / expected_integral

        ### Should be reasonably close to 4π (within ~5%)
        assert relative_error < 0.05, (
            f"Lumpy sphere (seed={seed}) integral far from 4π. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"relative_error={relative_error:.1%} exceeds 5%"
        )

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_lumpy_sphere_discretization_invariance(self, device, seed):
        """Test that ∫∫ K dA is invariant under further mesh refinement.

        This is the key test: after initial subdivision, further refinement
        should not significantly change the integral value.
        """
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        ### Create lumpy sphere at subdivision level 2
        mesh_coarse = lumpy_sphere.load(
            noise_amplitude=0.2,
            subdivisions=2,
            seed=seed,
            device=device,
        )

        integral_coarse = compute_gaussian_curvature_integral(mesh_coarse)

        ### Refine further with one more level of Loop subdivision
        mesh_fine = mesh_coarse.subdivide(levels=1, filter="loop")

        integral_fine = compute_gaussian_curvature_integral(mesh_fine)

        ### Integrals should be very similar (discretization-invariant)
        absolute_difference = torch.abs(integral_fine - integral_coarse)
        relative_difference = absolute_difference / (
            0.5 * (torch.abs(integral_fine) + torch.abs(integral_coarse))
        )

        assert relative_difference < 0.01, (
            f"Discretization variance too high (seed={seed}). "
            f"Coarse integral={integral_coarse:.6f}, "
            f"Fine integral={integral_fine:.6f}, "
            f"relative_difference={relative_difference:.1%} exceeds 1%"
        )

        ### Both should be close to 4π
        expected_integral = 4.0 * torch.pi
        for label, integral in [("coarse", integral_coarse), ("fine", integral_fine)]:
            relative_error = torch.abs(integral - expected_integral) / expected_integral
            assert relative_error < 0.05, (
                f"Lumpy sphere {label} (seed={seed}) integral far from 4π. "
                f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
                f"relative_error={relative_error:.1%} exceeds 5%"
            )


### Test Robustness


class TestGaussBonnetRobustness:
    """Additional robustness tests for various perturbations and base meshes."""

    @pytest.mark.parametrize("amplitude", [0.1, 0.2, 0.4])
    def test_different_perturbation_amplitudes(self, device, amplitude):
        """Test Gauss-Bonnet with different perturbation strengths."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        expected_integral = 4.0 * torch.pi

        ### Create lumpy sphere with given perturbation amplitude
        mesh = lumpy_sphere.load(
            noise_amplitude=amplitude,
            subdivisions=2,
            seed=42,
            device=device,
        )

        integral = compute_gaussian_curvature_integral(mesh)
        relative_error = torch.abs(integral - expected_integral) / expected_integral

        ### Should still be close to 4π (tolerance depends on amplitude)
        # Larger perturbations may need coarser tolerance
        if amplitude <= 0.2:
            tolerance = 0.05
        else:
            tolerance = 0.10

        assert relative_error < tolerance, (
            f"Lumpy sphere (amplitude={amplitude}) integral far from 4π. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"relative_error={relative_error:.1%} exceeds {tolerance * 100}%"
        )

    def test_octahedron_base_mesh(self, device):
        """Test Gauss-Bonnet starting from octahedron instead of icosahedron."""
        expected_integral = 4.0 * torch.pi

        ### Create octahedron
        mesh = octahedron_surface.load(size=1.0, device=device)

        ### Perturb and subdivide
        torch.manual_seed(42)
        radii = (
            torch.rand(mesh.n_points, dtype=torch.float32, device=device) * 0.4 + 0.8
        )
        perturbed_points = mesh.points * radii.unsqueeze(-1)

        mesh = Mesh(
            points=perturbed_points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )

        ### Subdivide
        mesh = mesh.subdivide(levels=2, filter="loop")

        integral = compute_gaussian_curvature_integral(mesh)
        relative_error = torch.abs(integral - expected_integral) / expected_integral

        ### Should still be close to 4π
        assert relative_error < 0.05, (
            f"Octahedron-based lumpy sphere integral far from 4π. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"relative_error={relative_error:.1%} exceeds 5%"
        )

    def test_tetrahedron_base_mesh(self, device):
        """Test Gauss-Bonnet starting from tetrahedron."""
        expected_integral = 4.0 * torch.pi

        ### Create tetrahedron
        mesh = tetrahedron_surface.load(side_length=1.0, device=device)

        ### Perturb and subdivide
        torch.manual_seed(42)
        radii = (
            torch.rand(mesh.n_points, dtype=torch.float32, device=device) * 0.4 + 0.8
        )
        perturbed_points = mesh.points * radii.unsqueeze(-1)

        mesh = Mesh(
            points=perturbed_points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )

        ### Subdivide more aggressively (tetrahedron is coarser)
        mesh = mesh.subdivide(levels=3, filter="loop")

        integral = compute_gaussian_curvature_integral(mesh)
        relative_error = torch.abs(integral - expected_integral) / expected_integral

        ### Should still be close to 4π
        assert relative_error < 0.05, (
            f"Tetrahedron-based lumpy sphere integral far from 4π. "
            f"Integral={integral:.6f}, expected={expected_integral:.6f}, "
            f"relative_error={relative_error:.1%} exceeds 5%"
        )


### Test Edge Cases


class TestGaussBonnetEdgeCases:
    """Tests for edge cases and validation."""

    def test_empty_mesh(self, device):
        """Test that empty mesh gives zero integral."""
        points = torch.empty((0, 3), dtype=torch.float32, device=device)
        cells = torch.empty((0, 3), dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        integral = compute_gaussian_curvature_integral(mesh)

        assert integral == 0.0, f"Empty mesh should give zero integral, got {integral}"

    def test_single_triangle(self, device):
        """Test curvature integral on single triangle."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        ### Should compute without error
        integral = compute_gaussian_curvature_integral(mesh)

        ### Single flat triangle has some curvature at vertices (angle defect)
        # but total should be related to Euler characteristic
        assert not torch.isnan(integral), "Integral should not be NaN"
        assert torch.isfinite(integral), "Integral should be finite"
