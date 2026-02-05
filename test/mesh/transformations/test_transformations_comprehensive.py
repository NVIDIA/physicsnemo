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

"""Comprehensive tests for transformations module to achieve 100% coverage.

Tests all error paths, edge cases, higher-order tensors, and data transformation.
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
        # (rotated identity is still identity)
        assert torch.allclose(transformed_stress, stress_tensor, atol=1e-5)

    def test_transform_rank3_tensor(self):
        """Test transformation of rank-3 tensor (e.g., piezoelectric tensor).

        For a rank-3 tensor T with shape (batch, n, n, n), the transformation is:
            T'_ijk = R_il * R_jm * R_kn * T_lmn

        We test with a simple diagonal-like tensor and verify the transformation
        is applied correctly to all three indices.
        """
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Create a rank-3 tensor field
        # Use a tensor where T[i,j,k] = 1 if i==j==k, else 0
        # This makes it easy to verify transformation
        rank3_tensor = torch.zeros(mesh.n_points, 3, 3, 3)
        for i in range(3):
            rank3_tensor[:, i, i, i] = 1.0

        mesh.point_data["piezo"] = rank3_tensor

        # Rotate 90 degrees about z-axis
        # R = [[0, -1, 0],
        #      [1,  0, 0],
        #      [0,  0, 1]]
        # This swaps x->y, y->-x, z->z
        angle = np.pi / 2
        rotated = rotate(
            mesh, angle=angle, axis=[0.0, 0.0, 1.0], transform_point_data=True
        )

        transformed = rotated.point_data["piezo"]

        # Verify shape is preserved
        assert transformed.shape == rank3_tensor.shape

        # Manually compute expected result for verification
        # The original tensor has T[0,0,0]=1, T[1,1,1]=1, T[2,2,2]=1
        # After rotation by 90° about z:
        # T'[i,j,k] = sum over l,m,n of: R[i,l] * R[j,m] * R[k,n] * T[l,m,n]
        #
        # Since original tensor is zero except on diagonal (0,0,0), (1,1,1), (2,2,2):
        # T'[i,j,k] = R[i,0]*R[j,0]*R[k,0]*T[0,0,0] + R[i,1]*R[j,1]*R[k,1]*T[1,1,1] + R[i,2]*R[j,2]*R[k,2]*T[2,2,2]
        #           = R[i,0]*R[j,0]*R[k,0] + R[i,1]*R[j,1]*R[k,1] + R[i,2]*R[j,2]*R[k,2]
        #
        # For 90° rotation about z: R = [[0,-1,0], [1,0,0], [0,0,1]]
        # T'[0,0,0] = 0*0*0 + 1*1*1 + 0*0*0 = 1  (y-component maps to old y-component)
        # T'[1,1,1] = (-1)*(-1)*(-1) + 0*0*0 + 0*0*0 = -1  (x-component maps to old -x)
        # Wait, that's not right. Let me reconsider.
        #
        # Actually, for rotation matrix R about z by 90°:
        # R[0,:] = [0, -1, 0]  (new x = -old y)
        # R[1,:] = [1,  0, 0]  (new y = old x)
        # R[2,:] = [0,  0, 1]  (new z = old z)
        #
        # T'[0,0,0] = R[0,0]*R[0,0]*R[0,0]*1 + R[0,1]*R[0,1]*R[0,1]*1 + R[0,2]*R[0,2]*R[0,2]*1
        #           = 0 + (-1)^3 + 0 = -1
        # T'[1,1,1] = R[1,0]*R[1,0]*R[1,0] + R[1,1]*R[1,1]*R[1,1] + R[1,2]*R[1,2]*R[1,2]
        #           = 1^3 + 0 + 0 = 1
        # T'[2,2,2] = R[2,0]^3 + R[2,1]^3 + R[2,2]^3 = 0 + 0 + 1 = 1

        expected = torch.zeros(mesh.n_points, 3, 3, 3)
        expected[:, 0, 0, 0] = -1.0  # Cube of -1 from R[0,1]=-1
        expected[:, 1, 1, 1] = 1.0  # Cube of 1 from R[1,0]=1
        expected[:, 2, 2, 2] = 1.0  # Cube of 1 from R[2,2]=1

        assert torch.allclose(transformed, expected, atol=1e-5), (
            f"Rank-3 tensor transformation failed.\nExpected:\n{expected[0]}\nGot:\n{transformed[0]}"
        )

    def test_transform_rank4_tensor(self):
        """Test transformation of rank-4 tensor (e.g., elasticity tensor).

        The elasticity tensor (Hooke's law) has 4 indices and transforms as:
            C'_ijkl = R_im * R_jn * R_ko * R_lp * C_mnop
        """
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Create a simple rank-4 tensor - identity-like tensor
        # C[i,j,k,l] = 1 if i==j==k==l, else 0
        rank4_tensor = torch.zeros(mesh.n_points, 2, 2, 2, 2)
        for i in range(2):
            rank4_tensor[:, i, i, i, i] = 1.0

        mesh.point_data["elasticity"] = rank4_tensor

        # Rotate 90 degrees in 2D
        # R = [[0, -1], [1, 0]]
        angle = np.pi / 2
        rotated = rotate(mesh, angle=angle, transform_point_data=True)

        transformed = rotated.point_data["elasticity"]

        # Verify shape is preserved
        assert transformed.shape == rank4_tensor.shape

        # For a 90° rotation in 2D: R = [[0, -1], [1, 0]]
        # C'[0,0,0,0] = R[0,0]^4 * C[0,0,0,0] + R[0,1]^4 * C[1,1,1,1]
        #             = 0^4 * 1 + (-1)^4 * 1 = 1
        # C'[1,1,1,1] = R[1,0]^4 * C[0,0,0,0] + R[1,1]^4 * C[1,1,1,1]
        #             = 1^4 * 1 + 0^4 * 1 = 1

        expected = torch.zeros(mesh.n_points, 2, 2, 2, 2)
        expected[:, 0, 0, 0, 0] = 1.0  # (-1)^4 = 1
        expected[:, 1, 1, 1, 1] = 1.0  # 1^4 = 1

        assert torch.allclose(transformed, expected, atol=1e-5), (
            f"Rank-4 tensor transformation failed.\nExpected diagonal elements [1, 1], got [{transformed[0, 0, 0, 0, 0]}, {transformed[0, 1, 1, 1, 1]}]"
        )


class TestDataTransformation:
    """Test transform_point_data/transform_cell_data/transform_global_data for all types."""

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
        # [1, 0, 0] -> [2, 0, 0] and [2, 0, 0] -> [1, 0, 0]
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


class TestTransformDimensionChange:
    """Test transformations that change spatial dimension."""

    def test_projection_3d_to_2d(self):
        """Test projection from 3D to 2D."""
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        # Project onto xy-plane
        proj_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        projected = transform(mesh, proj_matrix)

        assert projected.n_spatial_dims == 2
        assert projected.points.shape == (2, 2)
        assert torch.allclose(projected.points, points[:, :2])

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


class TestRotateDataTransformEdgeCases:
    """Test rotate() with transform_point_data/transform_cell_data covering all code paths."""

    def test_rotate_handles_geometric_caches_separately(self):
        """Test that geometric cached properties are handled by cache handler, not transform flags."""
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
        from physicsnemo.mesh.utilities._cache import get_cached, set_cached

        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Cached field
        set_cached(mesh.cell_data, "test_vector", torch.ones(mesh.n_cells, 3))

        rotated = rotate(
            mesh, angle=np.pi / 2, axis=[0, 0, 1], transform_cell_data=True
        )

        # Cache should not be transformed (not included in user data transformation)
        # The cache is preserved but not included in the transformation
        assert (
            "_cache" not in rotated.cell_data
            or get_cached(rotated.cell_data, "test_vector") is None
        )

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


class TestScaleDataTransformEdgeCases:
    """Test scale() with transform_point_data covering all paths."""

    def test_scale_data_skips_cached(self):
        """Test scale skips cached fields (under "_cache")."""
        from physicsnemo.mesh.utilities._cache import get_cached, set_cached

        points = torch.tensor([[1.0, 0.0]])
        cells = torch.tensor([[0]])
        mesh = Mesh(points=points, cells=cells)

        set_cached(mesh.point_data, "test_vector", torch.tensor([[1.0, 2.0]]))

        scaled = scale(mesh, factor=2.0, transform_point_data=True)

        # Cache should not be transformed (excluded from user data transformation)
        assert (
            "_cache" not in scaled.point_data
            or get_cached(scaled.point_data, "test_vector") is None
        )

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


class TestTranslateEdgeCases:
    """Test translate edge cases."""

    def test_translate_with_wrong_offset_dims_raises(self):
        """Test that offset with wrong dimensions raises ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="offset must have shape"):
            translate(mesh, offset=[1.0, 2.0, 3.0])  # 3D offset for 2D mesh


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
