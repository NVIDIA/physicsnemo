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

"""Comprehensive tests for geometric transformations.

Tests verify correctness of translate, rotate, scale, and general linear transformations
across spatial dimensions, manifold dimensions, and compute backends, with proper cache
invalidation and preservation. Includes error handling, higher-order tensors, and
data transformation.

This module consolidates tests from:
- Core transformation tests with PyVista cross-validation and cache handling
- Comprehensive coverage tests for error paths, edge cases, and data transformation
"""

import numpy as np
import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)

pv = pytest.importorskip("pyvista", minversion="0.46.4")

from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista  # noqa: E402

###############################################################################
# Helper Functions
###############################################################################


def create_mesh_with_caches(
    n_spatial_dims: int, n_manifold_dims: int, device: torch.device | str = "cpu"
):
    """Create a mesh and pre-compute all caches."""
    if n_manifold_dims == 1 and n_spatial_dims == 2:
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]], device=device, dtype=torch.int64
        )
    elif n_manifold_dims == 2 and n_spatial_dims == 2:
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 2 and n_spatial_dims == 3:
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 3 and n_spatial_dims == 3:
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], device=device, dtype=torch.int64
        )
    else:
        raise ValueError(
            f"Unsupported combination: {n_manifold_dims=}, {n_spatial_dims=}"
        )

    mesh = Mesh(points=points, cells=cells)

    # Pre-compute caches
    _ = mesh.cell_areas
    _ = mesh.cell_centroids
    if mesh.codimension == 1:
        _ = mesh.cell_normals

    return mesh


def validate_caches(
    mesh, expected_caches: dict[str, bool], rtol: float = 1e-4, atol: float = 1e-4
) -> None:
    """Validate that caches exist and are correct."""
    for cache_name, should_exist in expected_caches.items():
        if should_exist:
            cached_value = mesh._cache.get(("cell", cache_name), None)
            assert cached_value is not None, (
                f"Cache {cache_name} should exist but is missing"
            )

            mesh_no_cache = Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=mesh.cell_data,
                global_data=mesh.global_data,
            )

            if cache_name == "areas":
                recomputed = mesh_no_cache.cell_areas
            elif cache_name == "centroids":
                recomputed = mesh_no_cache.cell_centroids
            elif cache_name == "normals":
                recomputed = mesh_no_cache.cell_normals
            else:
                raise ValueError(f"Unknown cache: {cache_name}")

            assert torch.allclose(cached_value, recomputed, rtol=rtol, atol=atol), (
                f"Cache {cache_name} has incorrect value.\n"
                f"Max diff: {(cached_value - recomputed).abs().max()}"
            )
        else:
            assert mesh._cache.get(("cell", cache_name), None) is None, (
                f"Cache {cache_name} should not exist but is present"
            )


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


###############################################################################
# Core Transformation Tests
###############################################################################


class TestTranslation:
    """Tests for translate() function."""

    ### Cross-validation against PyVista ###

    def test_translate_against_pyvista(self, device):
        """Cross-validate against PyVista translate."""
        pv_mesh = pv.examples.load_airplane()
        tm_mesh = from_pyvista(pv_mesh)
        tm_mesh = Mesh(
            points=tm_mesh.points.to(device),
            cells=tm_mesh.cells.to(device),
        )

        offset = np.array([10.0, 20.0, 30.0])

        # PyVista translation (on CPU)
        pv_result = pv_mesh.translate(offset, inplace=False)

        # physicsnemo.mesh translation
        tm_result = translate(tm_mesh, offset)

        # Compare points - use rtol for large coordinate values
        tm_as_pv = to_pyvista(tm_result.to("cpu"))
        assert np.allclose(tm_as_pv.points, pv_result.points, rtol=1e-3, atol=1e-3)

    ### Parametrized dimensional tests ###

    @pytest.mark.parametrize("n_spatial_dims", [2, 3])
    def test_translate_simple_parametrized(self, n_spatial_dims, device):
        """Test simple translation across dimensions."""
        n_manifold_dims = n_spatial_dims - 1  # Use triangles in 3D, edges in 2D
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        offset = torch.ones(n_spatial_dims, device=device)
        original_points = mesh.points.clone()

        translated = translate(mesh, offset)

        assert_on_device(translated.points, device)
        expected_points = original_points + offset
        assert torch.allclose(translated.points, expected_points), (
            f"Translation incorrect. Max diff: {(translated.points - expected_points).abs().max()}"
        )

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [(2, 1), (2, 2), (3, 2), (3, 3)],
    )
    def test_translate_preserves_caches(self, n_spatial_dims, n_manifold_dims, device):
        """Verify translation correctly updates caches across dimensions."""
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        original_areas = mesh._cache.get(("cell", "areas"), None).clone()
        original_centroids = mesh._cache.get(("cell", "centroids"), None).clone()

        offset = torch.ones(n_spatial_dims, device=device)
        translated = translate(mesh, offset)

        # Validate caches
        expected_caches = {
            "areas": True,  # Should exist and be unchanged
            "centroids": True,  # Should exist and be translated
        }
        if mesh.codimension == 1:
            original_normals = mesh._cache.get(("cell", "normals"), None).clone()
            expected_caches["normals"] = True  # Should exist and be unchanged

        validate_caches(translated, expected_caches)

        # Verify specific values
        assert torch.allclose(
            translated._cache.get(("cell", "areas"), None), original_areas
        ), "Areas should be unchanged by translation"
        assert torch.allclose(
            translated._cache.get(("cell", "centroids"), None),
            original_centroids + offset,
        ), "Centroids should be translated"

        if mesh.codimension == 1:
            assert torch.allclose(
                translated._cache.get(("cell", "normals"), None), original_normals
            ), "Normals should be unchanged by translation"

    def test_translate_preserves_data(self):
        """Test that translate preserves vector fields unchanged."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Vector field
        mesh.point_data["velocity"] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Translation only affects points, not data (it's affine, not linear)
        translated = translate(mesh, offset=[5.0, 0.0, 0.0])

        # Data should be copied unchanged
        assert torch.allclose(
            translated.point_data["velocity"], mesh.point_data["velocity"]
        )


class TestRotation:
    """Tests for rotate() function."""

    ### Cross-validation against PyVista ###

    @pytest.mark.parametrize("axis_idx,angle", [(0, 45.0), (1, 30.0), (2, 60.0)])
    def test_rotate_against_pyvista(self, axis_idx, angle, device):
        """Cross-validate against PyVista rotation."""
        pv_mesh = pv.examples.load_airplane()
        tm_mesh = from_pyvista(pv_mesh)
        tm_mesh = Mesh(
            points=tm_mesh.points.to(device),
            cells=tm_mesh.cells.to(device),
        )

        # Rotation axis
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axis = axes[axis_idx]

        # PyVista rotation
        if axis_idx == 0:
            pv_result = pv_mesh.rotate_x(angle, inplace=False)
        elif axis_idx == 1:
            pv_result = pv_mesh.rotate_y(angle, inplace=False)
        else:
            pv_result = pv_mesh.rotate_z(angle, inplace=False)

        # physicsnemo.mesh rotation
        tm_result = rotate(tm_mesh, np.radians(angle), axis)

        # Compare points - use rtol for large coordinate values
        tm_as_pv = to_pyvista(tm_result.to("cpu"))
        assert np.allclose(tm_as_pv.points, pv_result.points, rtol=1e-3, atol=1e-3)

    ### Parametrized dimensional tests ###

    def test_rotate_2d_90deg(self, device):
        """Test 2D rotation by 90 degrees."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        rotated = rotate(mesh, np.pi / 2)

        # After 90 degree rotation: [1, 0] -> [0, 1], [0, 1] -> [-1, 0]
        expected = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]],
            device=device,
        )
        assert torch.allclose(rotated.points, expected, atol=1e-6)

    def test_rotate_3d_about_z(self, device):
        """Test 3D rotation about z-axis."""
        points = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        rotated = rotate(mesh, np.pi / 2, [0, 0, 1])

        expected = torch.tensor(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        assert torch.allclose(rotated.points, expected, atol=1e-6)

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", [(2, 1), (3, 2)])
    def test_rotate_preserves_areas_codim1(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify rotation preserves areas but transforms centroids and normals."""
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        original_areas = mesh._cache.get(("cell", "areas"), None).clone()
        original_centroids = mesh._cache.get(("cell", "centroids"), None).clone()
        original_normals = mesh._cache.get(("cell", "normals"), None).clone()

        # Rotate by 45 degrees
        if n_spatial_dims == 2:
            rotated = rotate(mesh, np.pi / 4)
        else:
            rotated = rotate(mesh, np.pi / 4, [1, 0, 0])

        validate_caches(
            rotated,
            {"areas": True, "centroids": True, "normals": True},
        )

        # Areas should be preserved (rotation has det=1)
        assert torch.allclose(
            rotated._cache.get(("cell", "areas"), None), original_areas
        ), "Areas should be preserved by rotation"

        # Centroids and normals should be different (rotated)
        assert not torch.allclose(
            rotated._cache.get(("cell", "centroids"), None), original_centroids
        ), "Centroids should be rotated"
        assert not torch.allclose(
            rotated._cache.get(("cell", "normals"), None), original_normals
        ), "Normals should be rotated"

    def test_rotate_with_vector_data(self):
        """Test rotate with transform_point_data=True rotates vectors."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Vector pointing in x direction
        mesh.point_data["vec"] = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Rotate 90° about z
        rotated = rotate(
            mesh, angle=np.pi / 2, axis=[0.0, 0.0, 1.0], transform_point_data=True
        )

        # Vector should now point in y direction
        expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        assert torch.allclose(rotated.point_data["vec"], expected, atol=1e-5)


class TestScale:
    """Tests for scale() function."""

    ### Cross-validation against PyVista ###

    def test_scale_against_pyvista(self, device):
        """Cross-validate against PyVista scale."""
        pv_mesh = pv.examples.load_airplane()
        tm_mesh = from_pyvista(pv_mesh)
        tm_mesh = Mesh(
            points=tm_mesh.points.to(device),
            cells=tm_mesh.cells.to(device),
        )

        factor = [2.0, 1.5, 0.8]

        # PyVista scaling
        pv_result = pv_mesh.scale(factor, inplace=False, point=[0.0, 0.0, 0.0])

        # physicsnemo.mesh scaling
        tm_result = scale(tm_mesh, factor)

        # Compare points - use rtol for large coordinate values
        tm_as_pv = to_pyvista(tm_result.to("cpu"))
        assert np.allclose(tm_as_pv.points, pv_result.points, rtol=1e-3, atol=1e-3)

    ### Parametrized dimensional tests ###

    @pytest.mark.parametrize("n_spatial_dims", [2, 3])
    def test_scale_uniform_simple(self, n_spatial_dims, device):
        """Test uniform scaling across dimensions."""
        n_manifold_dims = n_spatial_dims - 1
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        factor = 2.0
        original_points = mesh.points.clone()

        scaled = scale(mesh, factor)

        assert_on_device(scaled.points, device)
        expected = original_points * factor
        assert torch.allclose(scaled.points, expected)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [(2, 1), (2, 2), (3, 2), (3, 3)],
    )
    def test_scale_uniform_updates_caches(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify uniform scaling correctly updates all caches."""
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        original_areas = mesh._cache.get(("cell", "areas"), None).clone()
        original_centroids = mesh._cache.get(("cell", "centroids"), None).clone()

        factor = 2.0
        scaled = scale(mesh, factor)

        validate_caches(scaled, {"areas": True, "centroids": True})

        # Areas should scale by factor^n_manifold_dims
        expected_areas = original_areas * (factor**n_manifold_dims)
        assert torch.allclose(
            scaled._cache.get(("cell", "areas"), None), expected_areas
        ), "Areas should scale by factor^n_manifold_dims"

        # Centroids should be scaled
        expected_centroids = original_centroids * factor
        assert torch.allclose(
            scaled._cache.get(("cell", "centroids"), None), expected_centroids
        )

        # For codim-1 and positive uniform scaling, normals should be unchanged
        if mesh.codimension == 1:
            original_normals = mesh._cache.get(("cell", "normals"), None).clone()
            validate_caches(scaled, {"normals": True})
            assert torch.allclose(
                scaled._cache.get(("cell", "normals"), None), original_normals
            )

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", [(2, 1), (3, 2)])
    def test_scale_negative_handles_normals(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify negative scaling correctly handles normals based on manifold dimension."""
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        scaled = scale(mesh, -1.0)

        # Normals should be correct (validated against recomputed values)
        validate_caches(scaled, {"areas": True, "centroids": True, "normals": True})

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", [(2, 1), (3, 2)])
    def test_scale_non_uniform_handles_caches(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify non-uniform scaling correctly computes areas using normals."""
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        factor = torch.ones(n_spatial_dims, device=device)
        factor[0] = 2.0  # Non-uniform

        scaled = scale(mesh, factor)

        # Areas correctly computed using normal-based scaling, normals also correct
        validate_caches(scaled, {"areas": True, "centroids": True, "normals": True})

    def test_scale_with_vector_data(self):
        """Test scale with transform_point_data=True scales vectors."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        mesh.point_data["vec"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Uniform scale by 2
        scaled = scale(mesh, factor=2.0, transform_point_data=True)

        # Vectors should be scaled
        expected = mesh.point_data["vec"] * 2.0
        assert torch.allclose(scaled.point_data["vec"], expected, atol=1e-5)

    def test_scale_changes_areas(self):
        """Test that scaling changes areas by factor squared."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        original_area = mesh.cell_areas

        # Scale by 2
        scaled = scale(mesh, factor=2.0)

        # Area should be 4x (2² for 2D)
        expected_area = original_area * 4.0
        assert torch.allclose(scaled.cell_areas, expected_area, atol=1e-5)

    def test_nonuniform_scale_changes_areas(self):
        """Test that non-uniform scaling changes areas correctly."""
        points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        original_area = mesh.cell_areas

        # Scale by [2, 3]
        scaled = scale(mesh, factor=[2.0, 3.0])

        # Area scales by product = 6
        expected_area = original_area * 6.0
        assert torch.allclose(scaled.cell_areas, expected_area, atol=1e-5)


class TestNonIsotropicAreaScaling:
    """Tests for per-element area scaling under non-isotropic transforms."""

    def test_anisotropic_scale_horizontal_surface_3d(self, device):
        """Test anisotropic scaling of a horizontal surface in 3D."""
        # Triangle in xy-plane (z=0)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Pre-compute caches
        original_area = mesh.cell_areas.clone()
        _ = mesh.cell_normals  # Ensure normals are cached

        # Scale by (2, 3, 5) - non-isotropic
        scaled = scale(mesh, [2.0, 3.0, 5.0])

        # Area should scale by 2 × 3 = 6 (xy-plane is stretched by x and y factors)
        expected_area = original_area * 6.0
        assert torch.allclose(
            scaled._cache.get(("cell", "areas"), None), expected_area, atol=1e-5
        ), (
            f"Expected area {expected_area.item()}, got {scaled._cache.get(('cell', 'areas'), None).item()}"
        )

    def test_anisotropic_scale_vertical_surface_3d(self, device):
        """Test anisotropic scaling of a vertical surface in 3D."""
        # Triangle in xz-plane (y=0)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        original_area = mesh.cell_areas.clone()
        _ = mesh.cell_normals

        # Scale by (2, 3, 5)
        scaled = scale(mesh, [2.0, 3.0, 5.0])

        # Area should scale by 2 × 5 = 10 (xz-plane is stretched by x and z factors)
        expected_area = original_area * 10.0
        assert torch.allclose(
            scaled._cache.get(("cell", "areas"), None), expected_area, atol=1e-5
        )

    def test_anisotropic_scale_diagonal_surface_3d(self, device):
        """Test anisotropic scaling of a diagonal surface in 3D."""
        # Triangle tilted at 45° - points form a surface with normal ≈ (1,1,1)/√3
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        _ = mesh.cell_areas
        _ = mesh.cell_normals

        # Scale by (2, 0.5, 3) - highly anisotropic
        scaled = scale(mesh, [2.0, 0.5, 3.0])

        # Validate against recomputation
        validate_caches(scaled, {"areas": True, "normals": True})

    def test_shear_transform_preserves_area_correctness(self, device):
        """Test that shear transforms correctly compute per-element areas."""
        # Triangle in xy-plane
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        _ = mesh.cell_areas
        _ = mesh.cell_normals

        # Shear in xy plane: [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]
        shear_matrix = torch.tensor(
            [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        sheared = transform(mesh, shear_matrix)

        # Validate against recomputation
        validate_caches(sheared, {"areas": True, "normals": True})

    def test_mixed_orientation_surfaces_3d(self, device):
        """Test mesh with multiple surfaces at different orientations."""
        # Two triangles: one horizontal (z=0), one vertical (y=0)
        points = torch.tensor(
            [
                # Horizontal triangle (xy-plane)
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                # Vertical triangle (xz-plane)
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        original_areas = mesh.cell_areas.clone()
        _ = mesh.cell_normals

        # Scale by (2, 3, 5)
        scaled = scale(mesh, [2.0, 3.0, 5.0])

        # Cell 0 (horizontal): area scales by 2 × 3 = 6
        # Cell 1 (vertical in xz): area scales by 2 × 5 = 10
        expected_areas = original_areas * torch.tensor([6.0, 10.0], device=device)

        assert torch.allclose(
            scaled._cache.get(("cell", "areas"), None), expected_areas, atol=1e-5
        ), (
            f"Expected {expected_areas}, got {scaled._cache.get(('cell', 'areas'), None)}"
        )


class TestTransform:
    """Tests for general linear transform() function."""

    @pytest.mark.parametrize("n_spatial_dims", [2, 3])
    def test_transform_identity(self, n_spatial_dims, device):
        """Test identity transformation leaves mesh unchanged."""
        n_manifold_dims = n_spatial_dims - 1
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        identity_matrix = torch.eye(n_spatial_dims, device=device)
        transformed = transform(mesh, identity_matrix)

        assert torch.allclose(transformed.points, mesh.points)

    def test_transform_shear_2d(self, device):
        """Test shear transformation in 2D."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Shear in x direction
        shear = torch.tensor([[1.0, 0.5], [0.0, 1.0]], device=device)
        sheared = transform(mesh, shear)

        expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], device=device)
        assert torch.allclose(sheared.points, expected)

    def test_transform_projection_3d_to_2d(self, device):
        """Test projection from 3D to 2D."""
        points = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Project onto xy-plane
        proj_xy = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
        projected = transform(mesh, proj_xy)

        expected = torch.tensor([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], device=device)
        assert torch.allclose(projected.points, expected)
        assert projected.n_spatial_dims == 2

    def test_transform_skips_scalar_fields(self):
        """Test that scalar fields are not transformed."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Scalar field
        mesh.point_data["temperature"] = torch.tensor([100.0, 200.0])

        # Transform
        matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        transformed = transform(mesh, matrix, transform_point_data=True)

        # Scalar should be unchanged
        assert torch.allclose(
            transformed.point_data["temperature"], mesh.point_data["temperature"]
        )

    def test_embedding_2d_to_3d(self):
        """Test embedding from 2D to 3D."""
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Embed into 3D (xy-plane at z=0)
        embed_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        embedded = transform(mesh, embed_matrix)

        assert embedded.n_spatial_dims == 3
        assert embedded.points.shape == (2, 3)
        expected = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
        assert torch.allclose(embedded.points, expected)


###############################################################################
# Error Handling Tests
###############################################################################


class TestRotationErrors:
    """Test error handling in rotation."""

    def test_rotate_3d_without_axis_raises(self):
        """Test that 3D rotation without axis raises ValueError."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="implies 2D rotation"):
            rotate(mesh, angle=np.pi / 2, axis=None)

    def test_rotate_3d_with_wrong_axis_shape_raises(self):
        """Test that axis with wrong shape raises NotImplementedError."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(NotImplementedError, match="only supported for 2D.*or 3D"):
            rotate(mesh, angle=np.pi / 2, axis=[1.0, 0.0])  # 2D axis for 3D mesh

    def test_rotate_with_zero_length_axis_raises(self):
        """Test that zero-length axis raises ValueError."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="near-zero length"):
            rotate(mesh, angle=np.pi / 2, axis=[0.0, 0.0, 0.0])

    def test_rotate_4d_raises_error(self):
        """Test that rotation in >3D raises an error."""
        torch.manual_seed(42)
        # 4D mesh
        points = torch.randn(5, 4)
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        # axis=provided implies 3D, so this raises ValueError for dimension mismatch
        with pytest.raises(ValueError, match="implies 3D rotation"):
            rotate(mesh, angle=np.pi / 4, axis=[1.0, 0.0, 0.0, 0.0])


class TestTransformErrors:
    """Test error handling in transform()."""

    def test_transform_with_1d_matrix_raises(self):
        """Test that 1D matrix raises ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="matrix must be 2D"):
            transform(mesh, torch.tensor([1.0, 2.0]))

    def test_transform_with_wrong_input_dims_raises(self):
        """Test that matrix with wrong input dimensions raises ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Matrix expects 3D input, mesh has 2D points
        matrix = torch.eye(3)

        with pytest.raises(ValueError, match="must equal mesh.n_spatial_dims"):
            transform(mesh, matrix)

    def test_transform_incompatible_field_raises(self):
        """Test that incompatible fields raise ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Incompatible tensor (first dim doesn't match n_spatial_dims)
        mesh.point_data["weird_tensor"] = torch.ones(mesh.n_points, 5, 7)  # 5 != 2

        matrix = torch.eye(2)

        # Should raise - incompatible with transformation
        with pytest.raises(ValueError, match="Cannot transform.*First.*dimension"):
            transform(mesh, matrix, transform_point_data=True)


class TestTranslateEdgeCases:
    """Test translate edge cases."""

    def test_translate_with_wrong_offset_dims_raises(self):
        """Test that offset with wrong dimensions raises ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="offset must have shape"):
            translate(mesh, offset=[1.0, 2.0, 3.0])  # 3D offset for 2D mesh


###############################################################################
# Higher-Order Tensor Transformation Tests
###############################################################################


class TestHigherOrderTensorTransformation:
    """Test transformation of rank-2 and higher tensors."""

    def test_transform_rank2_tensor(self):
        """Test transformation of rank-2 tensor (stress tensor)."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Add rank-2 tensor field (e.g., stress tensor)
        stress_tensor = torch.eye(2).unsqueeze(0).expand(mesh.n_points, -1, -1)
        mesh.point_data["stress"] = stress_tensor

        # Rotate by 90 degrees
        angle = np.pi / 2
        rotated = rotate(mesh, angle=angle, transform_point_data=True)

        # Stress tensor should be transformed: T' = R @ T @ R^T
        transformed_stress = rotated.point_data["stress"]

        assert transformed_stress.shape == stress_tensor.shape
        # For identity tensor, rotation shouldn't change it much
        assert torch.allclose(transformed_stress, stress_tensor, atol=1e-5)

    def test_transform_rank3_tensor(self):
        """Test transformation of rank-3 tensor (e.g., piezoelectric tensor)."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Create a rank-3 tensor field
        rank3_tensor = torch.zeros(mesh.n_points, 3, 3, 3)
        for i in range(3):
            rank3_tensor[:, i, i, i] = 1.0

        mesh.point_data["piezo"] = rank3_tensor

        # Rotate 90 degrees about z-axis
        angle = np.pi / 2
        rotated = rotate(
            mesh, angle=angle, axis=[0.0, 0.0, 1.0], transform_point_data=True
        )

        transformed = rotated.point_data["piezo"]

        # Verify shape is preserved
        assert transformed.shape == rank3_tensor.shape

        expected = torch.zeros(mesh.n_points, 3, 3, 3)
        expected[:, 0, 0, 0] = -1.0  # Cube of -1 from R[0,1]=-1
        expected[:, 1, 1, 1] = 1.0  # Cube of 1 from R[1,0]=1
        expected[:, 2, 2, 2] = 1.0  # Cube of 1 from R[2,2]=1

        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_transform_rank4_tensor(self):
        """Test transformation of rank-4 tensor (e.g., elasticity tensor)."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Create a simple rank-4 tensor - identity-like tensor
        rank4_tensor = torch.zeros(mesh.n_points, 2, 2, 2, 2)
        for i in range(2):
            rank4_tensor[:, i, i, i, i] = 1.0

        mesh.point_data["elasticity"] = rank4_tensor

        # Rotate 90 degrees in 2D
        angle = np.pi / 2
        rotated = rotate(mesh, angle=angle, transform_point_data=True)

        transformed = rotated.point_data["elasticity"]

        # Verify shape is preserved
        assert transformed.shape == rank4_tensor.shape

        expected = torch.zeros(mesh.n_points, 2, 2, 2, 2)
        expected[:, 0, 0, 0, 0] = 1.0  # (-1)^4 = 1
        expected[:, 1, 1, 1, 1] = 1.0  # 1^4 = 1

        assert torch.allclose(transformed, expected, atol=1e-5)


###############################################################################
# Data Transformation Tests
###############################################################################


class TestDataTransformation:
    """Test transform_point_data/transform_cell_data/transform_global_data for all types."""

    def test_transform_cell_data_vectors(self):
        """Test that cell_data vectors are also transformed."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Cell vector field
        mesh.cell_data["flux"] = torch.tensor([[1.0, 0.0, 0.0]])

        # Rotate 90° about z
        rotated = rotate(
            mesh, angle=np.pi / 2, axis=[0.0, 0.0, 1.0], transform_cell_data=True
        )

        # Flux should rotate
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(rotated.cell_data["flux"], expected, atol=1e-5)


class TestRotateWithCenter:
    """Test rotation about a custom center point."""

    def test_rotate_about_custom_center(self):
        """Test rotation about a point other than origin."""
        points = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Rotate about center=[1.5, 0, 0] by 180°
        center = [1.5, 0.0, 0.0]
        rotated = rotate(mesh, angle=np.pi, axis=[0.0, 0.0, 1.0], center=center)

        # Points should be reflected about center in xy-plane
        expected = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert torch.allclose(rotated.points, expected, atol=1e-5)


class TestScaleWithCenter:
    """Test scaling about a custom center point."""

    def test_scale_uniform_about_center(self):
        """Test uniform scaling about a custom center."""
        points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Scale by 2 about center=[1, 1]
        center = [1.0, 1.0]
        scaled = scale(mesh, factor=2.0, center=center)

        # Points should be: (p - center) * 2 + center
        expected = (points - torch.tensor(center)) * 2.0 + torch.tensor(center)
        assert torch.allclose(scaled.points, expected, atol=1e-5)

    def test_scale_nonuniform(self):
        """Test non-uniform scaling (anisotropic)."""
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Scale differently in each dimension
        factors = [2.0, 0.5, 3.0]
        scaled = scale(mesh, factor=factors)

        expected = points * torch.tensor(factors)
        assert torch.allclose(scaled.points, expected, atol=1e-5)

    def test_scale_with_center_and_data(self):
        """Test scaling with center and transform_point_data=True."""
        points = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        mesh.point_data["vec"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        scaled = scale(mesh, factor=2.0, center=[1.0, 0.0], transform_point_data=True)

        # Vectors should be scaled
        expected_vec = mesh.point_data["vec"] * 2.0
        assert torch.allclose(scaled.point_data["vec"], expected_vec, atol=1e-5)


###############################################################################
# Cache Invalidation Tests
###############################################################################


class TestCacheInvalidation:
    """Test that cached properties are properly invalidated/preserved."""

    def test_translate_preserves_areas(self):
        """Test that translation preserves cell areas."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Pre-compute area
        original_area = mesh.cell_areas

        # Translate
        translated = translate(mesh, offset=[10.0, 20.0])

        # Area should be preserved
        assert torch.allclose(translated.cell_areas, original_area)

    def test_rotate_preserves_areas(self):
        """Test that rotation preserves cell areas (isometry)."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        original_area = mesh.cell_areas

        # Rotate 45°
        rotated = rotate(mesh, angle=np.pi / 4, axis=[0.0, 0.0, 1.0])

        # Area preserved
        assert torch.allclose(rotated.cell_areas, original_area, atol=1e-5)

    def test_rotate_invalidates_normals(self):
        """Test that rotation invalidates and recomputes normals."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Pre-compute normal
        original_normal = mesh.cell_normals
        assert torch.allclose(original_normal[0], torch.tensor([0.0, 0.0, 1.0]))

        # Rotate 90° about x-axis
        rotated = rotate(mesh, angle=np.pi / 2, axis=[1.0, 0.0, 0.0])

        # Normal should now point in -y direction
        new_normal = rotated.cell_normals
        expected_normal = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(new_normal[0], expected_normal, atol=1e-5)


###############################################################################
# Rotation Composition Tests
###############################################################################


class TestRotationComposition:
    """Test composition of rotations."""

    def test_two_rotations_compose_correctly(self):
        """Test that two consecutive rotations compose correctly."""
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Rotate 90° about z, then 90° about x
        mesh1 = rotate(mesh, angle=np.pi / 2, axis=[0, 0, 1])
        mesh2 = rotate(mesh1, angle=np.pi / 2, axis=[1, 0, 0])

        # First point [1,0,0] -> [0,1,0] -> [0,0,1]
        expected0 = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(mesh2.points[0], expected0, atol=1e-5)

        # Second point [0,1,0] -> [-1,0,0] -> [-1,0,0]
        expected1 = torch.tensor([-1.0, 0.0, 0.0])
        assert torch.allclose(mesh2.points[1], expected1, atol=1e-5)


###############################################################################
# Mesh Method Wrapper Tests
###############################################################################


class TestMeshMethodWrappers:
    """Test that Mesh.rotate(), Mesh.translate(), etc. work correctly."""

    def test_mesh_translate_method(self):
        """Test Mesh.translate() wrapper."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        translated = mesh.translate([5.0, 3.0])

        expected = points + torch.tensor([5.0, 3.0])
        assert torch.allclose(translated.points, expected)

    def test_mesh_rotate_method(self):
        """Test Mesh.rotate() wrapper."""
        points = torch.tensor([[1.0, 0.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        rotated = mesh.rotate(np.pi / 2, [0, 0, 1])

        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(rotated.points, expected, atol=1e-5)

    def test_mesh_scale_method(self):
        """Test Mesh.scale() wrapper."""
        points = torch.tensor([[1.0, 2.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        scaled = mesh.scale(3.0)

        expected = points * 3.0
        assert torch.allclose(scaled.points, expected)

    def test_mesh_transform_method(self):
        """Test Mesh.transform() wrapper."""
        points = torch.tensor([[1.0, 2.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        transformed = mesh.transform(matrix)

        expected = torch.tensor([[2.0, 6.0]])
        assert torch.allclose(transformed.points, expected)


###############################################################################
# Transformation Accuracy Tests
###############################################################################


class TestTransformationAccuracy:
    """Test numerical accuracy of transformations."""

    def test_rotation_orthogonality(self):
        """Test that rotation matrices are orthogonal."""
        points = torch.tensor([[1.0, 0.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        # Multiple rotations should preserve lengths
        for angle in [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]:
            rotated = rotate(mesh, angle=angle, axis=[1, 1, 1])

            # Length should be preserved
            original_length = torch.norm(mesh.points[0])
            rotated_length = torch.norm(rotated.points[0])
            assert torch.allclose(rotated_length, original_length, atol=1e-6)

    def test_rotation_determinant_one(self):
        """Test that rotation preserves orientation (det=1)."""
        # Create a mesh with known volume
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        original_volume = mesh.cell_areas

        # Rotate by arbitrary angle
        rotated = rotate(mesh, angle=0.7, axis=[1, 2, 3])

        # Volume should be preserved (rotation is isometry)
        assert torch.allclose(rotated.cell_areas, original_volume, atol=1e-5)


###############################################################################
# Scale Edge Cases
###############################################################################


class TestScaleEdgeCases:
    """Test scale edge cases."""

    def test_scale_by_zero_allowed(self):
        """Test that scaling by zero is allowed (collapses to point)."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Scaling by zero is mathematically valid (degenerate but allowed)
        scaled = scale(mesh, factor=0.0)

        # All points collapse to origin (or center if specified)
        assert torch.allclose(scaled.points, torch.zeros_like(scaled.points))

    def test_scale_by_negative(self):
        """Test that negative scaling works (reflection)."""
        points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Negative scale causes reflection
        scaled = scale(mesh, factor=-1.0)

        expected = -points
        assert torch.allclose(scaled.points, expected)

        # Volume should be preserved (absolute value)
        assert torch.allclose(scaled.cell_areas, mesh.cell_areas)

    def test_scale_with_mixed_signs(self):
        """Test scaling with mixed positive/negative factors."""
        points = torch.tensor([[1.0, 2.0, 3.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        scaled = scale(mesh, factor=[2.0, -1.0, 0.5])

        expected = torch.tensor([[2.0, -2.0, 1.5]])
        assert torch.allclose(scaled.points, expected)


###############################################################################
# Rotate Data Transform Edge Cases
###############################################################################


class TestRotateDataTransformEdgeCases:
    """Test rotate() with transform_point_data/transform_cell_data covering all code paths."""

    def test_rotate_handles_geometric_caches_separately(self):
        """Test that geometric cached properties are handled by cache handler."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Pre-compute normal
        original_normal = mesh.cell_normals
        assert torch.allclose(original_normal[0], torch.tensor([0.0, 0.0, 1.0]))

        # Rotate - normals should be rotated by cache handler, not transform flags
        rotated = rotate(mesh, angle=np.pi / 2, axis=[1, 0, 0])

        # Normal should still be rotated (handled by internal cache logic)
        new_normal = rotated.cell_normals
        expected = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(new_normal[0], expected, atol=1e-5)

    def test_rotate_with_wrong_dim_field_raises(self):
        """Test that rotate raises for fields with wrong first dimension."""
        points = torch.tensor([[1.0, 0.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        # Field with wrong first dimension
        mesh.point_data["weird"] = torch.ones(mesh.n_points, 5)  # 5 != 3

        with pytest.raises(ValueError, match="Cannot transform.*First.*dimension"):
            rotate(mesh, angle=np.pi / 2, axis=[0, 0, 1], transform_point_data=True)

    def test_rotate_with_incompatible_tensor_raises(self):
        """Test that incompatible tensor raises ValueError."""
        points = torch.tensor([[1.0, 0.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        # Tensor with shape (n_points, 3, 2) - not all dims equal n_spatial_dims
        mesh.point_data["bad"] = torch.ones(mesh.n_points, 3, 2)

        with pytest.raises(ValueError, match="Cannot transform.*field"):
            rotate(mesh, angle=np.pi / 2, axis=[0, 0, 1], transform_point_data=True)

    def test_rotate_cell_data_skips_cached(self):
        """Test that rotate skips cached cell_data fields (under "_cache")."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Cached field (mesh._cache is separate from cell_data; transformations build
        # fresh _cache with only propagated entries, so custom entries are not propagated)
        mesh._cache[("cell", "test_vector")] = torch.ones(mesh.n_cells, 3)

        rotated = rotate(
            mesh, angle=np.pi / 2, axis=[0, 0, 1], transform_cell_data=True
        )

        # Custom cache entry should not be propagated to rotated mesh
        assert rotated._cache.get(("cell", "test_vector"), None) is None

    def test_rotate_cell_data_wrong_shape_raises(self):
        """Test rotate raises for cell_data with wrong shape."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Wrong shape
        mesh.cell_data["weird"] = torch.ones(mesh.n_cells, 5)

        with pytest.raises(ValueError, match="Cannot transform.*First.*dimension"):
            rotate(mesh, angle=np.pi / 2, axis=[0, 0, 1], transform_cell_data=True)

    def test_rotate_cell_data_incompatible_tensor_raises(self):
        """Test rotate with incompatible cell tensor raises."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        mesh.cell_data["bad"] = torch.ones(mesh.n_cells, 3, 2)

        with pytest.raises(ValueError, match="Cannot transform.*field"):
            rotate(mesh, angle=np.pi / 2, axis=[0, 0, 1], transform_cell_data=True)


###############################################################################
# Scale Data Transform Edge Cases
###############################################################################


class TestScaleDataTransformEdgeCases:
    """Test scale() with transform_point_data covering all paths."""

    def test_scale_data_skips_cached(self):
        """Test scale skips cached fields (under "_cache")."""
        points = torch.tensor([[1.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        # Cached field (mesh._cache is separate from point_data; transformations build
        # fresh _cache with only propagated entries, so custom entries are not propagated)
        mesh._cache[("point", "test_vector")] = torch.tensor([[1.0, 2.0]])

        scaled = scale(mesh, factor=2.0, transform_point_data=True)

        # Custom cache entry should not be propagated to scaled mesh
        assert scaled._cache.get(("point", "test_vector"), None) is None

    def test_scale_data_wrong_shape_raises(self):
        """Test scale raises for fields with wrong shape."""
        points = torch.tensor([[1.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        mesh.point_data["weird"] = torch.ones(mesh.n_points, 5)

        with pytest.raises(ValueError, match="Cannot transform.*First.*dimension"):
            scale(mesh, factor=2.0, transform_point_data=True)

    def test_scale_with_incompatible_tensor_raises(self):
        """Test scale with incompatible tensor raises ValueError."""
        points = torch.tensor([[1.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        mesh.point_data["bad"] = torch.ones(mesh.n_points, 2, 3)

        with pytest.raises(ValueError, match="Cannot transform.*field"):
            scale(mesh, factor=2.0, transform_point_data=True)


###############################################################################
# Global Data Transformation Tests
###############################################################################


class TestGlobalDataTransformation:
    """Test global_data transformation."""

    def test_transform_global_data_vector(self):
        """Test that global_data vectors are transformed."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Global vector field (no batch dimension)
        mesh.global_data["reference_direction"] = torch.tensor([1.0, 0.0, 0.0])

        # Rotate 90° about z
        rotated = rotate(
            mesh, angle=np.pi / 2, axis=[0.0, 0.0, 1.0], transform_global_data=True
        )

        # Vector should now point in y direction
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(
            rotated.global_data["reference_direction"], expected, atol=1e-5
        )

    def test_transform_global_data_scalar_unchanged(self):
        """Test that global_data scalars are unchanged."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Global scalar
        mesh.global_data["temperature"] = torch.tensor(300.0)

        # Transform
        matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        transformed = transform(mesh, matrix, transform_global_data=True)

        # Scalar should be unchanged
        assert torch.allclose(
            transformed.global_data["temperature"], mesh.global_data["temperature"]
        )

    def test_transform_global_data_incompatible_raises(self):
        """Test that incompatible global_data raises ValueError."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Incompatible vector (5 != 3)
        mesh.global_data["bad_vector"] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="Cannot transform.*First.*dimension"):
            rotate(
                mesh, angle=np.pi / 2, axis=[0.0, 0.0, 1.0], transform_global_data=True
            )

    def test_scale_global_data(self):
        """Test scale transforms global_data vectors."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        mesh.global_data["force"] = torch.tensor([1.0, 2.0])

        scaled = scale(mesh, factor=3.0, transform_global_data=True)

        expected = torch.tensor([3.0, 6.0])
        assert torch.allclose(scaled.global_data["force"], expected, atol=1e-5)


###############################################################################
# General Edge Cases
###############################################################################


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_mesh(self, device):
        """Test transformations on empty mesh."""
        points = torch.zeros(0, 3, device=device)
        cells = torch.zeros(0, 3, dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # All transformations should work on empty mesh
        translated = translate(mesh, [1, 2, 3])
        assert translated.n_points == 0
        assert_on_device(translated.points, device)

        rotated = rotate(mesh, np.pi / 2, [0, 0, 1])
        assert rotated.n_points == 0

        scaled = scale(mesh, 2.0)
        assert scaled.n_points == 0

    @pytest.mark.parametrize("n_spatial_dims", [2, 3])
    def test_device_preservation(self, n_spatial_dims, device):
        """Test that transformations preserve device."""
        n_manifold_dims = n_spatial_dims - 1
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        # All transformations should preserve device
        translated = mesh.translate(torch.ones(n_spatial_dims, device=device))
        assert_on_device(translated.points, device)
        assert_on_device(translated.cells, device)

        if n_spatial_dims == 3:
            rotated = mesh.rotate(np.pi / 4, [0, 0, 1])
            assert_on_device(rotated.points, device)

        scaled = mesh.scale(2.0)
        assert_on_device(scaled.points, device)

    def test_rotation_axis_normalization(self, device):
        """Test that rotation axis is automatically normalized."""
        mesh = create_mesh_with_caches(3, 2, device=device)

        # Use non-unit axis
        axis_unnormalized = [2.0, 0.0, 0.0]
        axis_normalized = [1.0, 0.0, 0.0]

        result1 = rotate(mesh, np.pi / 4, axis_unnormalized)
        result2 = rotate(mesh, np.pi / 4, axis_normalized)

        assert torch.allclose(result1.points, result2.points, atol=1e-6)

    def test_multiple_transformations_composition(self, device):
        """Test composing multiple transformations with cache tracking."""
        mesh = create_mesh_with_caches(3, 2, device=device)

        # Translate -> Rotate -> Scale
        result = mesh.translate([1, 2, 3])
        validate_caches(result, {"areas": True, "centroids": True, "normals": True})

        result = result.rotate(np.pi / 4, [0, 0, 1])
        validate_caches(result, {"areas": True, "centroids": True, "normals": True})

        result = result.scale(2.0)
        validate_caches(result, {"areas": True, "centroids": True, "normals": True})

        # Final result should have correctly maintained caches
        # Areas should be scaled by 2^2 = 4
        assert torch.allclose(
            result._cache.get(("cell", "areas"), None),
            mesh._cache.get(("cell", "areas"), None) * 4.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
