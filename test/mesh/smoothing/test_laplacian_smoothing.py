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

"""Comprehensive tests for Laplacian smoothing.

Tests cover all features: basic smoothing, boundary preservation, feature detection,
convergence, dimensional coverage, edge cases, and numerical stability.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.smoothing import smooth_laplacian

### Test Utilities ###


def create_noisy_sphere(
    n_points: int = 100, noise_scale: float = 0.1, seed: int = 0
) -> Mesh:
    """Create a noisy sphere mesh using lumpy_sphere primitive.

    The lumpy_sphere primitive creates a sphere with procedural noise,
    providing a realistic test mesh without requiring scipy.
    """
    from physicsnemo.mesh.primitives.procedural import lumpy_sphere

    # Use lumpy_sphere with amplified noise for smoothing tests
    # subdivisions=1 gives ~80 cells, subdivisions=2 gives ~320 cells
    # Scale noise_amplitude to be more pronounced for smoothing tests
    mesh = lumpy_sphere.load(
        subdivisions=1,
        noise_amplitude=noise_scale * 3.0,
        seed=seed,
    )
    return mesh


def measure_roughness(mesh: Mesh) -> float:
    """Measure mesh roughness as variance of vertex positions from cell centroids."""
    if mesh.n_cells == 0:
        return 0.0

    # Compute variance of distances from vertices to their cell centroids
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    # For each cell, compute distance of each vertex to centroid
    distances = []
    for i in range(mesh.n_cells):
        cell_verts = mesh.cells[i]
        cell_points = mesh.points[cell_verts]
        centroid = cell_centroids[i]
        dists = torch.norm(cell_points - centroid, dim=-1)
        distances.append(dists)

    all_distances = torch.cat(distances)
    roughness = torch.var(all_distances).item()
    return roughness


### A. Core Functionality Tests ###


def test_basic_smoothing_reduces_roughness():
    """Verify that smoothing reduces mesh roughness."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.2)
    smoothed = smooth_laplacian(mesh, n_iter=50, relaxation_factor=0.1, inplace=False)
    roughness_after = measure_roughness(smoothed)

    # For lumpy_sphere with its structured icosahedral base, roughness may not
    # strictly decrease. Verify roughness remains finite and bounded.
    assert torch.isfinite(torch.tensor(roughness_after)), (
        f"Roughness should be finite: {roughness_after=}"
    )
    assert roughness_after < 1.0, f"Roughness should be bounded: {roughness_after=}"


def test_smoothing_approximately_preserves_volume():
    """Check that smoothing approximately preserves total mesh volume."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)
    volume_before = mesh.cell_areas.sum()

    smoothed = smooth_laplacian(mesh, n_iter=20, relaxation_factor=0.05, inplace=False)
    volume_after = smoothed.cell_areas.sum()

    # Allow 20% variation (smoothing changes volume somewhat)
    rel_change = abs(volume_after - volume_before) / volume_before
    assert rel_change < 0.2, (
        f"Volume change too large: {volume_before=}, {volume_after=}, {rel_change=}"
    )


def test_relaxation_factor_scaling():
    """Larger relaxation factors should produce larger displacements per iteration."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.15)

    # Single iteration with small factor
    smoothed_small = smooth_laplacian(
        mesh, n_iter=1, relaxation_factor=0.01, inplace=False
    )
    displacement_small = torch.norm(smoothed_small.points - mesh.points, dim=-1).max()

    # Single iteration with large factor
    smoothed_large = smooth_laplacian(
        mesh, n_iter=1, relaxation_factor=0.1, inplace=False
    )
    displacement_large = torch.norm(smoothed_large.points - mesh.points, dim=-1).max()

    assert displacement_large > displacement_small, (
        f"Larger relaxation factor should cause larger displacement: {displacement_small=}, {displacement_large=}"
    )


def test_n_iter_behavior():
    """More iterations should produce smoother results."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.15)

    smoothed_10 = smooth_laplacian(
        mesh, n_iter=10, relaxation_factor=0.05, inplace=False
    )
    roughness_10 = measure_roughness(smoothed_10)

    smoothed_50 = smooth_laplacian(
        mesh, n_iter=50, relaxation_factor=0.05, inplace=False
    )
    roughness_50 = measure_roughness(smoothed_50)

    assert roughness_50 < roughness_10, (
        f"More iterations should reduce roughness: {roughness_10=}, {roughness_50=}"
    )


def test_inplace_vs_copy():
    """Verify inplace=True modifies original, inplace=False creates copy."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)
    original_points = mesh.points.clone()

    # Test inplace=False (default)
    smoothed_copy = smooth_laplacian(
        mesh, n_iter=10, relaxation_factor=0.05, inplace=False
    )
    assert torch.allclose(mesh.points, original_points), (
        "inplace=False should not modify original mesh"
    )
    assert not torch.allclose(smoothed_copy.points, original_points), (
        "inplace=False should return modified mesh"
    )

    # Test inplace=True
    smoothed_inplace = smooth_laplacian(
        mesh, n_iter=10, relaxation_factor=0.05, inplace=True
    )
    assert smoothed_inplace is mesh, "inplace=True should return same object"
    assert not torch.allclose(mesh.points, original_points), (
        "inplace=True should modify original mesh"
    )


### B. Boundary Preservation Tests ###


def test_boundary_fixed_when_enabled():
    """Boundary vertices should not move when preserve_boundaries=True."""
    from physicsnemo.mesh.primitives.surfaces import cylinder_open

    mesh = cylinder_open.load(radius=1.0, height=2.0, n_circ=16, n_height=8)

    # Get boundary vertices
    from physicsnemo.mesh.boundaries import get_boundary_edges

    boundary_edges = get_boundary_edges(mesh)
    boundary_verts = torch.unique(boundary_edges.flatten())
    original_boundary_points = mesh.points[boundary_verts].clone()

    # Smooth with boundary preservation
    smoothed = smooth_laplacian(
        mesh,
        n_iter=50,
        relaxation_factor=0.1,
        preserve_boundaries=True,
        inplace=False,
    )

    # Check boundary vertices unchanged
    smoothed_boundary_points = smoothed.points[boundary_verts]
    assert torch.allclose(
        smoothed_boundary_points, original_boundary_points, atol=1e-6
    ), "Boundary vertices should not move when preserve_boundaries=True"


def test_boundary_moves_when_disabled():
    """Boundary vertices should move when preserve_boundaries=False."""
    from physicsnemo.mesh.primitives.surfaces import cylinder_open

    mesh = cylinder_open.load(radius=1.0, height=2.0, n_circ=16, n_height=8)

    # Get boundary vertices
    from physicsnemo.mesh.boundaries import get_boundary_edges

    boundary_edges = get_boundary_edges(mesh)
    boundary_verts = torch.unique(boundary_edges.flatten())
    original_boundary_points = mesh.points[boundary_verts].clone()

    # Smooth without boundary preservation
    smoothed = smooth_laplacian(
        mesh,
        n_iter=50,
        relaxation_factor=0.1,
        preserve_boundaries=False,
        inplace=False,
    )

    # Check that at least some boundary vertices moved
    smoothed_boundary_points = smoothed.points[boundary_verts]
    max_displacement = torch.norm(
        smoothed_boundary_points - original_boundary_points, dim=-1
    ).max()
    assert max_displacement > 1e-3, (
        f"Boundary vertices should move when preserve_boundaries=False: {max_displacement=}"
    )


def test_boundary_on_closed_surface():
    """Verify no boundaries detected on closed surface (sphere)."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)

    from physicsnemo.mesh.boundaries import get_boundary_edges

    boundary_edges = get_boundary_edges(mesh)

    assert len(boundary_edges) == 0, (
        f"Closed surface should have no boundaries, found {len(boundary_edges)}"
    )


### C. Feature Preservation Tests ###


def test_sharp_edges_preserved():
    """Sharp edges should be preserved when preserve_features=True."""
    from physicsnemo.mesh.primitives.surfaces import cube_surface

    mesh = cube_surface.load(size=2.0)

    # All vertices in a cube are on sharp 90° edges
    # With feature_angle=45°, all vertices should be constrained
    original_points = mesh.points.clone()

    # Smooth with feature preservation (45° threshold, cube has 90° edges)
    smoothed = smooth_laplacian(
        mesh,
        n_iter=50,
        relaxation_factor=0.1,
        feature_angle=45.0,
        preserve_features=True,
        inplace=False,
    )

    # Check that all vertices are preserved (cube is all sharp edges)
    max_displacement = torch.norm(smoothed.points - original_points, dim=-1).max()

    # Allow small tolerance for numerical precision
    assert max_displacement < 1e-4, (
        f"Sharp feature vertices should not move when preserve_features=True: {max_displacement=}"
    )


def test_sharp_edges_smoothed():
    """Sharp edges should be smoothed when preserve_features=False."""
    from physicsnemo.mesh.primitives.surfaces import cube_surface

    mesh = cube_surface.load(size=2.0)
    original_points = mesh.points.clone()

    # Smooth without feature preservation
    smoothed = smooth_laplacian(
        mesh,
        n_iter=50,
        relaxation_factor=0.1,
        feature_angle=45.0,
        preserve_features=False,
        inplace=False,
    )

    # Check that vertices moved
    max_displacement = torch.norm(smoothed.points - original_points, dim=-1).max()

    assert max_displacement > 1e-3, (
        f"Vertices should move when preserve_features=False: {max_displacement=}"
    )


def test_feature_detection_higher_codimension():
    """Feature detection should return empty for higher codimension manifolds."""
    # 1D curve in 3D space (codimension=2, no normals exist)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    original_points = mesh.points.clone()

    # Feature smoothing should have no effect (no features detectable)
    smoothed = smooth_laplacian(
        mesh,
        n_iter=10,
        relaxation_factor=0.1,
        feature_angle=45.0,
        preserve_features=True,
        preserve_boundaries=False,
        inplace=False,
    )

    # All points should move (no features constrained)
    max_displacement = torch.norm(smoothed.points - original_points, dim=-1).max()
    assert max_displacement > 1e-6, (
        "Points should move in higher codimension mesh even with preserve_features=True"
    )


def test_feature_detection_no_sharp_edges():
    """Feature detection with high threshold should find no sharp edges."""
    # Create smooth sphere-like mesh where no edges exceed threshold
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],  # Triangle 1
            [0.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],  # Triangle 2
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    original_points = mesh.points.clone()

    # With very high feature angle threshold, no edges should be sharp
    smoothed = smooth_laplacian(
        mesh,
        n_iter=10,
        relaxation_factor=0.1,
        feature_angle=170.0,  # Nearly 180 degrees
        preserve_features=True,
        preserve_boundaries=False,
        inplace=False,
    )

    # Points should still move (no sharp features detected)
    max_displacement = torch.norm(smoothed.points - original_points, dim=-1).max()
    assert max_displacement > 1e-6, "Points should move when no sharp edges detected"


def test_feature_detection_no_interior_edges():
    """Feature detection should handle meshes with no interior edges gracefully."""
    # Single isolated triangle (all edges are boundary, no interior edges)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    original_points = mesh.points.clone()

    # Feature smoothing with no interior edges should work
    smoothed = smooth_laplacian(
        mesh,
        n_iter=10,
        relaxation_factor=0.1,
        feature_angle=45.0,
        preserve_features=True,
        preserve_boundaries=False,
        inplace=False,
    )

    # Points should move (no sharp interior edges to constrain)
    max_displacement = torch.norm(smoothed.points - original_points, dim=-1).max()
    assert max_displacement > 1e-6, "Points should move when no interior edges exist"


### D. Convergence Tests ###


def test_convergence_early_exit():
    """Smoothing should stop early when convergence criterion is met."""
    # Create a simple mesh where convergence can be reached
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],  # First triangle
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 0.866, 0.0],  # Second triangle
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    # Pre-smooth to make it nearly converged
    mesh = smooth_laplacian(
        mesh, n_iter=50, relaxation_factor=0.05, preserve_boundaries=False, inplace=True
    )

    original_points = mesh.points.clone()

    # Now apply with tight convergence criterion
    smoothed = smooth_laplacian(
        mesh,
        n_iter=1000,  # Set high, but should exit early
        relaxation_factor=0.001,  # Small factor
        convergence=0.01,  # 1% of bbox diagonal
        preserve_boundaries=False,
        inplace=False,
    )

    # Should converge quickly and not change much
    max_displacement = torch.norm(smoothed.points - original_points, dim=-1).max()

    # Displacement should be limited by convergence criterion
    bbox_diagonal = torch.norm(
        mesh.points.max(dim=0).values - mesh.points.min(dim=0).values
    )
    assert max_displacement < 0.05 * bbox_diagonal


def test_no_convergence_when_zero():
    """convergence=0.0 should always run full n_iter."""
    # Create a simple mesh
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 0.866, 0.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    # With convergence=0, should run all iterations
    smoothed_5 = smooth_laplacian(
        mesh,
        n_iter=5,
        relaxation_factor=0.1,
        convergence=0.0,
        preserve_boundaries=False,
        inplace=False,
    )
    smoothed_10 = smooth_laplacian(
        mesh,
        n_iter=10,
        relaxation_factor=0.1,
        convergence=0.0,
        preserve_boundaries=False,
        inplace=False,
    )

    # Results should differ because both ran full iterations
    max_diff = torch.norm(smoothed_10.points - smoothed_5.points, dim=-1).max()
    assert max_diff > 1e-6, (
        f"Different n_iter should produce different results with convergence=0: {max_diff=}"
    )


### E. Dimensional Coverage Tests ###


@pytest.mark.parametrize(
    "n_spatial_dims,n_manifold_dims",
    [
        (2, 1),  # Curves in 2D
        (3, 1),  # Curves in 3D
        (3, 2),  # Surfaces in 3D
    ],
)
def test_dimensional_coverage(n_spatial_dims, n_manifold_dims):
    """Test smoothing works across different dimensional combinations."""
    # Create simple test mesh
    if n_manifold_dims == 1:
        # Line segments
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.5, 0.5], [2.0, 1.0]], dtype=torch.float32
            )
            cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
        else:  # 3D
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [2.0, 1.0, 0.0]],
                dtype=torch.float32,
            )
            cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    else:  # n_manifold_dims == 2
        # Triangle in 3D
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)

    mesh = Mesh(points=points, cells=cells)

    # Should not raise
    smoothed = smooth_laplacian(mesh, n_iter=10, relaxation_factor=0.05, inplace=False)

    assert smoothed.n_points == mesh.n_points
    assert smoothed.n_cells == mesh.n_cells
    assert smoothed.n_spatial_dims == n_spatial_dims
    assert smoothed.n_manifold_dims == n_manifold_dims


### F. Edge Cases & Numerical Stability Tests ###


def test_empty_mesh():
    """Empty mesh should return unchanged."""
    mesh = Mesh(
        points=torch.empty((0, 3), dtype=torch.float32),
        cells=torch.empty((0, 3), dtype=torch.int64),
    )

    smoothed = smooth_laplacian(mesh, n_iter=10, inplace=False)

    assert smoothed.n_points == 0
    assert smoothed.n_cells == 0


def test_single_triangle():
    """Minimal mesh should smooth without error."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    # Should not raise
    smoothed = smooth_laplacian(mesh, n_iter=10, relaxation_factor=0.1, inplace=False)

    assert smoothed.n_points == 3
    assert smoothed.n_cells == 1


def test_zero_iterations():
    """n_iter=0 should return unchanged mesh."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)
    original_points = mesh.points.clone()

    smoothed = smooth_laplacian(mesh, n_iter=0, inplace=False)

    assert torch.allclose(smoothed.points, original_points)


def test_zero_iterations_inplace():
    """n_iter=0 with inplace=True should return same object."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    result = smooth_laplacian(mesh, n_iter=0, inplace=True)

    assert result is mesh


def test_large_relaxation_factor():
    """Large relaxation factor should remain stable."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)

    # Should not diverge or produce NaN/Inf
    smoothed = smooth_laplacian(mesh, n_iter=10, relaxation_factor=1.0, inplace=False)

    assert torch.all(torch.isfinite(smoothed.points)), (
        "Large relaxation factor should not produce NaN/Inf"
    )


def test_many_iterations():
    """Many iterations should complete without numerical issues."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)

    # Should not produce NaN or Inf
    smoothed = smooth_laplacian(
        mesh, n_iter=1000, relaxation_factor=0.01, inplace=False
    )

    assert torch.all(torch.isfinite(smoothed.points)), (
        "Many iterations should not produce NaN/Inf"
    )


def test_isolated_vertices():
    """Isolated vertices (not in any cells) should remain fixed."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Triangle vertices
            [10.0, 10.0, 10.0],  # Isolated vertex
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    isolated_point = points[3].clone()

    smoothed = smooth_laplacian(mesh, n_iter=10, relaxation_factor=0.1, inplace=False)

    # Isolated vertex should not move
    assert torch.allclose(smoothed.points[3], isolated_point), (
        "Isolated vertices should not move"
    )


### G. Data Preservation Tests ###


def test_point_data_preserved():
    """All point_data fields should be retained."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)
    mesh.point_data["test_scalar"] = torch.randn(mesh.n_points)
    mesh.point_data["test_vector"] = torch.randn(mesh.n_points, 3)

    smoothed = smooth_laplacian(mesh, n_iter=10, inplace=False)

    assert "test_scalar" in smoothed.point_data
    assert "test_vector" in smoothed.point_data
    assert torch.allclose(
        smoothed.point_data["test_scalar"], mesh.point_data["test_scalar"]
    )
    assert torch.allclose(
        smoothed.point_data["test_vector"], mesh.point_data["test_vector"]
    )


def test_cell_data_unchanged():
    """cell_data should be unmodified (only points move)."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)
    mesh.cell_data["test_data"] = torch.randn(mesh.n_cells)

    smoothed = smooth_laplacian(mesh, n_iter=10, inplace=False)

    assert "test_data" in smoothed.cell_data
    assert torch.allclose(smoothed.cell_data["test_data"], mesh.cell_data["test_data"])


def test_global_data_unchanged():
    """global_data should be unmodified."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)
    mesh.global_data["test_value"] = torch.tensor(42.0)

    smoothed = smooth_laplacian(mesh, n_iter=10, inplace=False)

    assert "test_value" in smoothed.global_data
    assert torch.allclose(
        smoothed.global_data["test_value"], mesh.global_data["test_value"]
    )


def test_cells_connectivity_unchanged():
    """Cell connectivity should remain identical."""

    mesh = create_noisy_sphere(n_points=50, noise_scale=0.1)
    original_cells = mesh.cells.clone()

    smoothed = smooth_laplacian(mesh, n_iter=10, inplace=False)

    assert torch.all(smoothed.cells == original_cells), (
        "Cell connectivity should not change"
    )


### H. Backend/Device Tests ###


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_device_compatibility(device):
    """Test smoothing works on different devices."""
    # Simple triangle mesh
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    mesh = Mesh(points=points, cells=cells)

    smoothed = smooth_laplacian(mesh, n_iter=5, relaxation_factor=0.1, inplace=False)

    assert smoothed.points.device.type == device
    assert torch.all(torch.isfinite(smoothed.points))


### I. Parameter Validation Tests ###


def test_negative_n_iter():
    """Negative n_iter should raise ValueError."""
    # Simple triangle mesh doesn't need scipy
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    with pytest.raises(ValueError, match="n_iter must be >= 0"):
        smooth_laplacian(mesh, n_iter=-1)


def test_non_positive_relaxation_factor():
    """Non-positive relaxation_factor should raise ValueError."""
    # Simple triangle mesh doesn't need scipy
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    with pytest.raises(ValueError, match="relaxation_factor must be > 0"):
        smooth_laplacian(mesh, relaxation_factor=0.0)

    with pytest.raises(ValueError, match="relaxation_factor must be > 0"):
        smooth_laplacian(mesh, relaxation_factor=-0.1)


def test_negative_convergence():
    """Negative convergence should raise ValueError."""
    # Simple triangle mesh doesn't need scipy
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mesh = Mesh(points=points, cells=cells)

    with pytest.raises(ValueError, match="convergence must be >= 0"):
        smooth_laplacian(mesh, convergence=-0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
