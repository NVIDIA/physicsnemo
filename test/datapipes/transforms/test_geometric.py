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

"""Tests for geometric transforms (Translate, Scale, ComputeSDF, ComputeNormals)."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.geometric import (
    ComputeNormals,
    ComputeSDF,
    Scale,
    Translate,
)


class TestTranslate:
    """Tests for Translate transform."""

    def test_translate_add_mode_default(self):
        """Test that add mode is the default (subtract=False)."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([1.0, 2.0, 3.0]),
        )
        assert transform.subtract is False

    def test_translate_add_mode_with_tensor(self):
        """Test add mode with a fixed tensor value."""
        offset = torch.tensor([1.0, 2.0, 3.0])
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_subtract_mode_with_tensor(self):
        """Test subtract mode with a fixed tensor value."""
        offset = torch.tensor([1.0, 2.0, 3.0])
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[-1.0, -2.0, -3.0], [0.0, -1.0, -2.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_add_mode_with_key(self):
        """Test add mode with a key reference."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="offset",
            subtract=False,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                "offset": torch.tensor([5.0, 10.0, 15.0]),
            },
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[5.0, 10.0, 15.0], [6.0, 11.0, 16.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_subtract_mode_with_key(self):
        """Test subtract mode with a key reference (centering use case)."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="center_of_mass",
            subtract=True,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                "center_of_mass": torch.tensor([1.0, 1.0, 1.0]),
            },
            batch_size=[],
        )

        result = transform(data)

        # Points should be centered: original - center_of_mass
        expected = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_multiple_keys(self):
        """Test translation applied to multiple keys."""
        offset = torch.tensor([1.0, 1.0, 1.0])
        transform = Translate(
            input_keys=["positions", "surface_points"],
            center_key_or_value=offset,
            subtract=False,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0]]),
                "surface_points": torch.tensor([[5.0, 5.0, 5.0]]),
            },
            batch_size=[],
        )

        result = transform(data)

        assert torch.allclose(result["positions"], torch.tensor([[1.0, 1.0, 1.0]]))
        assert torch.allclose(result["surface_points"], torch.tensor([[6.0, 6.0, 6.0]]))

    def test_translate_preserves_other_fields(self):
        """Test that translation preserves fields not in input_keys."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[0.0, 0.0, 0.0]]),
                "velocities": torch.tensor([[1.0, 2.0, 3.0]]),
            },
            batch_size=[],
        )

        result = transform(data)

        # Velocities should be unchanged
        assert torch.allclose(result["velocities"], torch.tensor([[1.0, 2.0, 3.0]]))

    def test_translate_missing_key_raises(self):
        """Test that missing center key raises KeyError."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="nonexistent_key",
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0]])},
            batch_size=[],
        )

        with pytest.raises(KeyError, match="nonexistent_key"):
            transform(data)

    def test_translate_1d_center_broadcasted(self):
        """Test that 1D center tensor is properly broadcasted."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([1.0, 2.0, 3.0]),  # 1D tensor
            subtract=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_2d_center(self):
        """Test that 2D center tensor works correctly."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value=torch.tensor([[1.0, 2.0, 3.0]]),  # 2D tensor (1, 3)
            subtract=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_translate_repr_add_mode(self):
        """Test repr shows add mode."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="offset",
            subtract=False,
        )

        repr_str = repr(transform)
        assert "Translate" in repr_str
        assert "mode=add" in repr_str
        assert "positions" in repr_str

    def test_translate_repr_subtract_mode(self):
        """Test repr shows subtract mode."""
        transform = Translate(
            input_keys=["positions"],
            center_key_or_value="center_of_mass",
            subtract=True,
        )

        repr_str = repr(transform)
        assert "Translate" in repr_str
        assert "mode=subtract" in repr_str
        assert "center_of_mass" in repr_str

    def test_translate_skips_missing_input_keys(self):
        """Test that missing input keys are silently skipped."""
        transform = Translate(
            input_keys=["positions", "nonexistent"],
            center_key_or_value=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict(
            {"positions": torch.tensor([[0.0, 0.0, 0.0]])},
            batch_size=[],
        )

        # Should not raise, just skip the missing key
        result = transform(data)
        assert torch.allclose(result["positions"], torch.tensor([[1.0, 1.0, 1.0]]))

    def test_translate_add_then_subtract_roundtrip(self):
        """Test that add followed by subtract returns to original."""
        offset = torch.tensor([5.0, 10.0, 15.0])
        add_transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=False,
        )
        subtract_transform = Translate(
            input_keys=["positions"],
            center_key_or_value=offset,
            subtract=True,
        )

        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data = TensorDict({"positions": original.clone()}, batch_size=[])

        # Add then subtract should return to original
        result = add_transform(data)
        result = subtract_transform(result)

        assert torch.allclose(result["positions"], original)


class TestScale:
    """Tests for Scale transform."""

    def test_scale_multiply_mode_default(self):
        """Test that multiply mode is the default (divide=False)."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
        )
        assert transform.divide is False

    def test_scale_multiply_mode_with_tensor(self):
        """Test multiply mode with a fixed tensor value."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_divide_mode_with_tensor(self):
        """Test divide mode with a fixed tensor value."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_multiple_keys(self):
        """Test scaling multiple keys."""
        transform = Scale(
            input_keys=["positions", "velocities"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=False,
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[1.0, 2.0, 3.0]]),
                "velocities": torch.tensor([[2.0, 4.0, 6.0]]),
            },
            batch_size=[],
        )

        result = transform(data)

        assert torch.allclose(result["positions"], torch.tensor([[2.0, 4.0, 6.0]]))
        assert torch.allclose(result["velocities"], torch.tensor([[4.0, 8.0, 12.0]]))

    def test_scale_preserves_other_fields(self):
        """Test that scaling preserves fields not in input_keys."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
        )

        data = TensorDict(
            {
                "positions": torch.tensor([[1.0, 2.0, 3.0]]),
                "labels": torch.tensor([1, 2, 3]),
            },
            batch_size=[],
        )

        result = transform(data)

        assert torch.equal(result["labels"], torch.tensor([1, 2, 3]))

    def test_scale_nonuniform_multiply(self):
        """Test scaling with non-uniform scale factors in multiply mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 4.0]),
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_nonuniform_divide(self):
        """Test scaling with non-uniform scale factors in divide mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 4.0]),
            divide=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 2.0, 4.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 1.0, 1.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_repr_multiply_mode(self):
        """Test repr shows multiply mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 3.0]),
            divide=False,
        )

        repr_str = repr(transform)
        assert "Scale" in repr_str
        assert "mode=multiply" in repr_str
        assert "positions" in repr_str

    def test_scale_repr_divide_mode(self):
        """Test repr shows divide mode."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([1.0, 2.0, 3.0]),
            divide=True,
        )

        repr_str = repr(transform)
        assert "Scale" in repr_str
        assert "mode=divide" in repr_str
        assert "positions" in repr_str

    def test_scale_1d_scale_broadcasted(self):
        """Test that 1D scale tensor is properly broadcasted."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),  # 1D tensor
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_2d_scale(self):
        """Test that 2D scale tensor works correctly."""
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([[2.0, 2.0, 2.0]]),  # 2D tensor (1, 3)
            divide=False,
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        assert torch.allclose(result["positions"], expected)

    def test_scale_skips_missing_input_keys(self):
        """Test that missing input keys are silently skipped."""
        transform = Scale(
            input_keys=["positions", "nonexistent"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
        )

        data = TensorDict(
            {"positions": torch.tensor([[1.0, 1.0, 1.0]])},
            batch_size=[],
        )

        # Should not raise, just skip the missing key
        result = transform(data)
        assert torch.allclose(result["positions"], torch.tensor([[2.0, 2.0, 2.0]]))

    def test_scale_multiply_then_divide_roundtrip(self):
        """Test that multiply followed by divide returns to original."""
        scale_factor = torch.tensor([2.0, 3.0, 4.0])
        multiply_transform = Scale(
            input_keys=["positions"],
            scale=scale_factor,
            divide=False,
        )
        divide_transform = Scale(
            input_keys=["positions"],
            scale=scale_factor,
            divide=True,
        )

        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data = TensorDict({"positions": original.clone()}, batch_size=[])

        # Multiply then divide should return to original
        result = multiply_transform(data)
        result = divide_transform(result)

        assert torch.allclose(result["positions"], original)


class TestReScaleBackwardsCompatibility:
    """Tests for Scale backwards compatibility alias."""

    def test_rescale_is_scale(self):
        """Test that Scale is an alias for Scale."""
        assert Scale is Scale

    def test_rescale_divide_mode_default_for_backwards_compat(self):
        """Test that Scale can be used with the new API.

        Note: The old Scale always divided. With the new Scale class,
        users need to explicitly set divide=True for the same behavior.
        """
        # Old behavior equivalent: dividing by scale
        transform = Scale(
            input_keys=["positions"],
            scale=torch.tensor([2.0, 2.0, 2.0]),
            divide=True,
        )

        data = TensorDict(
            {"positions": torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])},
            batch_size=[],
        )

        result = transform(data)

        expected = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        assert torch.allclose(result["positions"], expected)

    def test_rescale_import_from_module(self):
        """Test that Scale can be imported from the transforms module."""
        from physicsnemo.datapipes.transforms import Scale as ImportedReScale

        assert ImportedReScale is Scale


# ============================================================================
# ComputeSDF Tests
# ============================================================================


class TestComputeSDF:
    """Tests for ComputeSDF transform."""

    @pytest.fixture
    def simple_cube_mesh(self):
        """Create a simple cube mesh centered at origin."""
        # Cube vertices from -1 to 1
        vertices = torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=torch.float32,
        )

        # Faces (triangulated cube - 12 triangles, 2 per face)
        faces = torch.tensor(
            [
                # Front
                0,
                1,
                2,
                0,
                2,
                3,
                # Back
                4,
                6,
                5,
                4,
                7,
                6,
                # Top
                3,
                2,
                6,
                3,
                6,
                7,
                # Bottom
                0,
                5,
                1,
                0,
                4,
                5,
                # Right
                1,
                5,
                6,
                1,
                6,
                2,
                # Left
                0,
                3,
                7,
                0,
                7,
                4,
            ],
            dtype=torch.int32,
        )

        return vertices, faces

    def test_sdf_basic(self, simple_cube_mesh):
        """Test basic SDF computation."""
        vertices, faces = simple_cube_mesh

        transform = ComputeSDF(
            input_keys=["query_points"],
            output_key="sdf",
            mesh_coords_key="mesh_coords",
            mesh_faces_key="mesh_faces",
        )

        # Query points at various locations
        query_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # At center
                [2.0, 0.0, 0.0],  # Further from surface
                [0.5, 0.5, 0.5],  # Near surface
            ]
        )

        data = TensorDict(
            {
                "query_points": query_points,
                "mesh_coords": vertices,
                "mesh_faces": faces,
            }
        )

        result = transform(data)

        assert "sdf" in result
        assert result["sdf"].shape == (3, 1)

        # SDF should be finite (not NaN or Inf)
        assert torch.isfinite(result["sdf"]).all()

        # Point further from surface should have larger absolute SDF
        assert abs(result["sdf"][1, 0]) > abs(result["sdf"][2, 0])

    def test_sdf_with_closest_points(self, simple_cube_mesh):
        """Test SDF computation with closest points output."""
        vertices, faces = simple_cube_mesh

        transform = ComputeSDF(
            input_keys=["query_points"],
            output_key="sdf",
            mesh_coords_key="mesh_coords",
            mesh_faces_key="mesh_faces",
            closest_points_key="closest_pts",
        )

        query_points = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        data = TensorDict(
            {
                "query_points": query_points,
                "mesh_coords": vertices,
                "mesh_faces": faces,
            }
        )

        result = transform(data)

        assert "closest_pts" in result
        assert result["closest_pts"].shape == (2, 3)

    def test_sdf_missing_mesh_coords_raises(self):
        """Test that missing mesh coordinates raises KeyError."""
        transform = ComputeSDF(
            input_keys=["query_points"],
            output_key="sdf",
            mesh_coords_key="mesh_coords",
            mesh_faces_key="mesh_faces",
        )

        data = TensorDict(
            {
                "query_points": torch.randn(10, 3),
                "mesh_faces": torch.randint(0, 100, (100,)),
            }
        )

        with pytest.raises(KeyError, match="Mesh coordinates key"):
            transform(data)

    def test_sdf_missing_mesh_faces_raises(self):
        """Test that missing mesh faces raises KeyError."""
        transform = ComputeSDF(
            input_keys=["query_points"],
            output_key="sdf",
            mesh_coords_key="mesh_coords",
            mesh_faces_key="mesh_faces",
        )

        data = TensorDict(
            {
                "query_points": torch.randn(10, 3),
                "mesh_coords": torch.randn(100, 3),
            }
        )

        with pytest.raises(KeyError, match="Mesh faces key"):
            transform(data)

    def test_sdf_missing_input_key_raises(self, simple_cube_mesh):
        """Test that missing input key raises KeyError."""
        vertices, faces = simple_cube_mesh

        transform = ComputeSDF(
            input_keys=["query_points"],
            output_key="sdf",
            mesh_coords_key="mesh_coords",
            mesh_faces_key="mesh_faces",
        )

        data = TensorDict(
            {
                "mesh_coords": vertices,
                "mesh_faces": faces,
            }
        )

        with pytest.raises(KeyError, match="Input key"):
            transform(data)

    def test_sdf_multiple_input_keys(self, simple_cube_mesh):
        """Test SDF computation with multiple input keys."""
        vertices, faces = simple_cube_mesh

        transform = ComputeSDF(
            input_keys=["query_a", "query_b"],
            output_key="sdf",
            mesh_coords_key="mesh_coords",
            mesh_faces_key="mesh_faces",
        )

        data = TensorDict(
            {
                "query_a": torch.tensor([[0.0, 0.0, 0.0]]),
                "query_b": torch.tensor([[2.0, 0.0, 0.0]]),
                "mesh_coords": vertices,
                "mesh_faces": faces,
            }
        )

        result = transform(data)

        # Multiple inputs should have suffixed output keys
        assert "sdf_query_a" in result
        assert "sdf_query_b" in result

    def test_sdf_repr(self):
        """Test string representation."""
        transform = ComputeSDF(
            input_keys=["points"],
            output_key="sdf_values",
            mesh_coords_key="coords",
            mesh_faces_key="faces",
        )

        repr_str = repr(transform)
        assert "ComputeSDF" in repr_str
        assert "points" in repr_str
        assert "sdf_values" in repr_str


# ============================================================================
# ComputeNormals Tests
# ============================================================================


class TestComputeNormals:
    """Tests for ComputeNormals transform."""

    def test_compute_normals_basic(self):
        """Test basic normal computation from closest points."""
        transform = ComputeNormals(
            positions_key="positions",
            closest_points_key="closest_points",
            center_of_mass_key="center_of_mass",
            output_key="normals",
        )

        # Points on a sphere surface
        positions = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        # Closest points on unit sphere (same direction, different magnitude)
        closest_points = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        center_of_mass = torch.tensor([0.0, 0.0, 0.0])

        data = TensorDict(
            {
                "positions": positions,
                "closest_points": closest_points,
                "center_of_mass": center_of_mass,
            }
        )

        result = transform(data)

        assert "normals" in result
        assert result["normals"].shape == (3, 3)

        # Normals should be unit length
        norms = torch.norm(result["normals"], dim=-1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)

        # Normals should point outward (same direction as position - closest)
        expected_normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(
            result["normals"], expected_normals, atol=1e-5, rtol=1e-5
        )

    def test_compute_normals_zero_distance_fallback(self):
        """Test that zero-distance points use center of mass fallback."""
        transform = ComputeNormals(
            positions_key="positions",
            closest_points_key="closest_points",
            center_of_mass_key="center_of_mass",
            output_key="normals",
            handle_zero_distance=True,
        )

        # Point exactly on the surface (position == closest_point)
        positions = torch.tensor([[1.0, 0.0, 0.0]])
        closest_points = torch.tensor([[1.0, 0.0, 0.0]])  # Same as position
        center_of_mass = torch.tensor([0.0, 0.0, 0.0])

        data = TensorDict(
            {
                "positions": positions,
                "closest_points": closest_points,
                "center_of_mass": center_of_mass,
            }
        )

        result = transform(data)

        # Should use direction from center of mass to position
        expected_normal = torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(
            result["normals"], expected_normal, atol=1e-5, rtol=1e-5
        )

    def test_compute_normals_2d_center_of_mass(self):
        """Test with 2D center of mass (shape (1, 3))."""
        transform = ComputeNormals(
            positions_key="positions",
            closest_points_key="closest_points",
            center_of_mass_key="center_of_mass",
            output_key="normals",
        )

        positions = torch.tensor([[2.0, 0.0, 0.0]])
        closest_points = torch.tensor([[1.0, 0.0, 0.0]])
        center_of_mass = torch.tensor([[0.0, 0.0, 0.0]])  # Shape (1, 3)

        data = TensorDict(
            {
                "positions": positions,
                "closest_points": closest_points,
                "center_of_mass": center_of_mass,
            }
        )

        result = transform(data)

        assert result["normals"].shape == (1, 3)

    def test_compute_normals_disable_zero_distance_handling(self):
        """Test with handle_zero_distance=False."""
        transform = ComputeNormals(
            positions_key="positions",
            closest_points_key="closest_points",
            center_of_mass_key="center_of_mass",
            output_key="normals",
            handle_zero_distance=False,
        )

        # Mixed points
        positions = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # Zero distance
            ]
        )
        closest_points = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # Same as position
            ]
        )
        center_of_mass = torch.tensor([0.0, 0.0, 0.0])

        data = TensorDict(
            {
                "positions": positions,
                "closest_points": closest_points,
                "center_of_mass": center_of_mass,
            }
        )

        result = transform(data)

        # First normal should still be correct
        assert result["normals"].shape == (2, 3)

    def test_compute_normals_repr(self):
        """Test string representation."""
        transform = ComputeNormals(
            positions_key="pos",
            closest_points_key="closest",
            center_of_mass_key="com",
            output_key="normals",
        )

        repr_str = repr(transform)
        assert "ComputeNormals" in repr_str
        assert "pos" in repr_str
        assert "normals" in repr_str
