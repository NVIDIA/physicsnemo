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

"""Tests for geometric transformations with cache handling and PyVista cross-validation.

Tests verify correctness of translate, rotate, scale, and general linear transformations
across spatial dimensions, manifold dimensions, and compute backends, with proper cache
invalidation and preservation.
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
from physicsnemo.mesh.utilities._cache import get_cached

pv = pytest.importorskip("pyvista", minversion="0.46.4")

from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista  # noqa: E402

### Helper Functions ###


def create_mesh_with_caches(
    n_spatial_dims: int, n_manifold_dims: int, device: torch.device | str = "cpu"
):
    """Create a mesh and pre-compute all caches."""
    from physicsnemo.mesh.mesh import Mesh

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
            cached_value = get_cached(mesh.cell_data, cache_name)
            assert cached_value is not None, (
                f"Cache {cache_name} should exist but is missing"
            )

            # Verify cache is correct by creating a fresh mesh without cache
            mesh_no_cache = Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=mesh.cell_data.exclude("_cache"),
                global_data=mesh.global_data,
            )

            # Recompute by accessing property
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
            assert get_cached(mesh.cell_data, cache_name) is None, (
                f"Cache {cache_name} should not exist but is present"
            )


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


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

        original_areas = get_cached(mesh.cell_data, "areas").clone()
        original_centroids = get_cached(mesh.cell_data, "centroids").clone()

        offset = torch.ones(n_spatial_dims, device=device)
        translated = translate(mesh, offset)

        # Validate caches
        expected_caches = {
            "areas": True,  # Should exist and be unchanged
            "centroids": True,  # Should exist and be translated
        }
        if mesh.codimension == 1:
            original_normals = get_cached(mesh.cell_data, "normals").clone()
            expected_caches["normals"] = True  # Should exist and be unchanged

        validate_caches(translated, expected_caches)

        # Verify specific values
        assert torch.allclose(
            get_cached(translated.cell_data, "areas"), original_areas
        ), "Areas should be unchanged by translation"
        assert torch.allclose(
            get_cached(translated.cell_data, "centroids"),
            original_centroids + offset,
        ), "Centroids should be translated"

        if mesh.codimension == 1:
            assert torch.allclose(
                get_cached(translated.cell_data, "normals"), original_normals
            ), "Normals should be unchanged by translation"


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

        original_areas = get_cached(mesh.cell_data, "areas").clone()
        original_centroids = get_cached(mesh.cell_data, "centroids").clone()
        original_normals = get_cached(mesh.cell_data, "normals").clone()

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
        assert torch.allclose(get_cached(rotated.cell_data, "areas"), original_areas), (
            "Areas should be preserved by rotation"
        )

        # Centroids and normals should be different (rotated)
        assert not torch.allclose(
            get_cached(rotated.cell_data, "centroids"), original_centroids
        ), "Centroids should be rotated"
        assert not torch.allclose(
            get_cached(rotated.cell_data, "normals"), original_normals
        ), "Normals should be rotated"


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

        original_areas = get_cached(mesh.cell_data, "areas").clone()
        original_centroids = get_cached(mesh.cell_data, "centroids").clone()

        factor = 2.0
        scaled = scale(mesh, factor)

        validate_caches(scaled, {"areas": True, "centroids": True})

        # Areas should scale by factor^n_manifold_dims
        expected_areas = original_areas * (factor**n_manifold_dims)
        assert torch.allclose(get_cached(scaled.cell_data, "areas"), expected_areas), (
            "Areas should scale by factor^n_manifold_dims"
        )

        # Centroids should be scaled
        expected_centroids = original_centroids * factor
        assert torch.allclose(
            get_cached(scaled.cell_data, "centroids"), expected_centroids
        )

        # For codim-1 and positive uniform scaling, normals should be unchanged
        if mesh.codimension == 1:
            original_normals = get_cached(mesh.cell_data, "normals").clone()
            validate_caches(scaled, {"normals": True})
            assert torch.allclose(
                get_cached(scaled.cell_data, "normals"), original_normals
            )

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", [(2, 1), (3, 2)])
    def test_scale_negative_handles_normals(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify negative scaling correctly handles normals based on manifold dimension.

        The generalized cross product of (n-1) vectors scales by (-1)^(n-1) when negated:
        - n_manifold_dims=1 (odd): normals flip
        - n_manifold_dims=2 (even): normals unchanged
        """
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        scaled = scale(mesh, -1.0)

        # Normals should be correct (validated against recomputed values)
        validate_caches(scaled, {"areas": True, "centroids": True, "normals": True})

    @pytest.mark.parametrize("n_spatial_dims,n_manifold_dims", [(2, 1), (3, 2)])
    def test_scale_non_uniform_handles_caches(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Verify non-uniform scaling correctly computes areas using normals.

        For codimension-1 embedded manifolds, per-element area scaling is computed
        using the formula: area' = area × |det(M)| × ||M^{-T} n||
        where n is the unit normal. This works because the normal encodes the
        tangent space orientation.
        """
        mesh = create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device=device)

        factor = torch.ones(n_spatial_dims, device=device)
        factor[0] = 2.0  # Non-uniform

        scaled = scale(mesh, factor)

        # Areas correctly computed using normal-based scaling, normals also correct
        validate_caches(scaled, {"areas": True, "centroids": True, "normals": True})


class TestNonIsotropicAreaScaling:
    """Tests for per-element area scaling under non-isotropic transforms.

    For codimension-1 manifolds, areas scale by: |det(M)| × ||M^{-T} n||
    where n is the unit normal. This depends on the orientation of each element.
    """

    def test_anisotropic_scale_horizontal_surface_3d(self, device):
        """Test anisotropic scaling of a horizontal surface in 3D.

        For a surface in the xy-plane with normal n=(0,0,1), scaling by (a,b,c)
        should scale the area by |abc| × ||M^{-T} n|| = |abc| × |1/c| = |ab|.
        """
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
            get_cached(scaled.cell_data, "areas"), expected_area, atol=1e-5
        ), (
            f"Expected area {expected_area.item()}, got {get_cached(scaled.cell_data, 'areas').item()}"
        )

    def test_anisotropic_scale_vertical_surface_3d(self, device):
        """Test anisotropic scaling of a vertical surface in 3D.

        For a surface in the xz-plane with normal n=(0,1,0), scaling by (a,b,c)
        should scale the area by |abc| × ||M^{-T} n|| = |abc| × |1/b| = |ac|.
        """
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
            get_cached(scaled.cell_data, "areas"), expected_area, atol=1e-5
        )

    def test_anisotropic_scale_diagonal_surface_3d(self, device):
        """Test anisotropic scaling of a diagonal surface in 3D.

        For a surface at 45° to all axes, the area scaling depends on the normal
        direction and should match the recomputed area exactly.
        """
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
        """Test that shear transforms correctly compute per-element areas.

        Shear transforms have det=1, but the area scaling is orientation-dependent
        for embedded manifolds.
        """
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
        # This is det=1, but non-isotropic
        shear_matrix = torch.tensor(
            [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        sheared = transform(mesh, shear_matrix)

        # For a horizontal surface with normal (0,0,1), shear in xy doesn't change z
        # So M^{-T} n should still have unit length in z-direction, thus area unchanged
        # Validate against recomputation
        validate_caches(sheared, {"areas": True, "normals": True})

    def test_mixed_orientation_surfaces_3d(self, device):
        """Test mesh with multiple surfaces at different orientations.

        Each surface element should have its area scaled according to its own
        normal direction.
        """
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
            get_cached(scaled.cell_data, "areas"), expected_areas, atol=1e-5
        ), f"Expected {expected_areas}, got {get_cached(scaled.cell_data, 'areas')}"


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
            get_cached(result.cell_data, "areas"),
            get_cached(mesh.cell_data, "areas") * 4.0,
            atol=1e-6,
        )
