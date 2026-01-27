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

"""Tests for spatial transforms (BoundingBoxFilter, CreateGrid, KNearestNeighbors, CenterOfMass)."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.spatial import (
    BoundingBoxFilter,
    CenterOfMass,
    CreateGrid,
    KNearestNeighbors,
)

# ============================================================================
# BoundingBoxFilter Tests
# ============================================================================


class TestBoundingBoxFilter:
    """Tests for BoundingBoxFilter transform."""

    def test_basic_filtering(self):
        """Test that points outside bbox are filtered out."""
        transform = BoundingBoxFilter(
            input_keys=["coords"],
            bbox_min=torch.tensor([-1.0, -1.0, -1.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        # Create points, some inside and some outside bbox
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # inside
                [0.5, 0.5, 0.5],  # inside
                [2.0, 0.0, 0.0],  # outside (x > 1)
                [-2.0, 0.0, 0.0],  # outside (x < -1)
                [0.0, 2.0, 0.0],  # outside (y > 1)
            ]
        )
        data = TensorDict({"coords": coords})

        result = transform(data)

        # Only first two points should remain
        assert result["coords"].shape[0] == 2
        torch.testing.assert_close(
            result["coords"],
            torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        )

    def test_filtering_with_dependent_keys(self):
        """Test that dependent keys are filtered with the same mask."""
        transform = BoundingBoxFilter(
            input_keys=["coords"],
            bbox_min=torch.tensor([-1.0, -1.0, -1.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
            dependent_keys=["values", "normals"],
        )

        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # inside
                [2.0, 0.0, 0.0],  # outside
                [0.5, 0.5, 0.5],  # inside
            ]
        )
        values = torch.tensor([1.0, 2.0, 3.0])
        normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        data = TensorDict({"coords": coords, "values": values, "normals": normals})

        result = transform(data)

        assert result["coords"].shape[0] == 2
        assert result["values"].shape[0] == 2
        assert result["normals"].shape[0] == 2

        torch.testing.assert_close(result["values"], torch.tensor([1.0, 3.0]))

    def test_all_points_inside(self):
        """Test when all points are inside bbox."""
        transform = BoundingBoxFilter(
            input_keys=["coords"],
            bbox_min=torch.tensor([-10.0, -10.0, -10.0]),
            bbox_max=torch.tensor([10.0, 10.0, 10.0]),
        )

        coords = torch.randn(100, 3)  # Should all be inside
        data = TensorDict({"coords": coords})

        result = transform(data)

        assert result["coords"].shape[0] == 100

    def test_all_points_outside(self):
        """Test when all points are outside bbox."""
        transform = BoundingBoxFilter(
            input_keys=["coords"],
            bbox_min=torch.tensor([100.0, 100.0, 100.0]),
            bbox_max=torch.tensor([200.0, 200.0, 200.0]),
        )

        coords = torch.randn(50, 3)  # All should be outside
        data = TensorDict({"coords": coords})

        result = transform(data)

        assert result["coords"].shape[0] == 0

    def test_multiple_input_keys(self):
        """Test filtering multiple coordinate keys."""
        transform = BoundingBoxFilter(
            input_keys=["coords1", "coords2"],
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        coords1 = torch.tensor(
            [
                [0.5, 0.5, 0.5],  # inside
                [-0.5, 0.5, 0.5],  # outside
            ]
        )
        coords2 = torch.tensor(
            [
                [0.2, 0.2, 0.2],  # inside
                [0.8, 0.8, 0.8],  # inside
                [1.5, 0.5, 0.5],  # outside
            ]
        )

        data = TensorDict({"coords1": coords1, "coords2": coords2})

        result = transform(data)

        assert result["coords1"].shape[0] == 1
        assert result["coords2"].shape[0] == 2

    def test_missing_input_key_skipped(self):
        """Test that missing input keys are skipped without error."""
        transform = BoundingBoxFilter(
            input_keys=["coords", "missing_key"],
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        coords = torch.tensor([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
        data = TensorDict({"coords": coords})

        # Should not raise, just skip missing key
        result = transform(data)
        assert result["coords"].shape[0] == 2

    def test_missing_dependent_key_skipped(self):
        """Test that missing dependent keys are skipped without error."""
        transform = BoundingBoxFilter(
            input_keys=["coords"],
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
            dependent_keys=["values", "missing_key"],
        )

        coords = torch.tensor([[0.5, 0.5, 0.5]])
        values = torch.tensor([1.0])
        data = TensorDict({"coords": coords, "values": values})

        result = transform(data)
        assert "values" in result
        assert result["values"].shape[0] == 1

    def test_repr(self):
        """Test string representation."""
        transform = BoundingBoxFilter(
            input_keys=["coords"],
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
            dependent_keys=["values"],
        )

        repr_str = repr(transform)
        assert "BoundingBoxFilter" in repr_str
        assert "coords" in repr_str
        assert "values" in repr_str


# ============================================================================
# CreateGrid Tests
# ============================================================================


class TestCreateGrid:
    """Tests for CreateGrid transform."""

    def test_basic_grid_creation(self):
        """Test creating a basic 3D grid."""
        transform = CreateGrid(
            output_key="grid",
            resolution=(4, 4, 4),
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict({})
        result = transform(data)

        assert "grid" in result
        # 4 * 4 * 4 = 64 points
        assert result["grid"].shape == (64, 3)

    def test_grid_bounds(self):
        """Test that grid points are within bounds."""
        transform = CreateGrid(
            output_key="grid",
            resolution=(10, 10, 10),
            bbox_min=torch.tensor([-1.0, -1.0, -1.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict({})
        result = transform(data)

        grid = result["grid"]

        # All coordinates should be within bounds
        assert (grid[:, 0] >= -1.0).all()
        assert (grid[:, 0] <= 1.0).all()
        assert (grid[:, 1] >= -1.0).all()
        assert (grid[:, 1] <= 1.0).all()
        assert (grid[:, 2] >= -1.0).all()
        assert (grid[:, 2] <= 1.0).all()

    def test_grid_corners(self):
        """Test that grid includes corner points."""
        transform = CreateGrid(
            output_key="grid",
            resolution=(2, 2, 2),
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict({})
        result = transform(data)

        grid = result["grid"]

        # Check corners are present
        assert grid.shape == (8, 3)

        # Min corner
        assert any(
            torch.allclose(grid[i], torch.tensor([0.0, 0.0, 0.0])) for i in range(8)
        )
        # Max corner
        assert any(
            torch.allclose(grid[i], torch.tensor([1.0, 1.0, 1.0])) for i in range(8)
        )

    def test_non_uniform_resolution(self):
        """Test grid with different resolutions per dimension."""
        transform = CreateGrid(
            output_key="grid",
            resolution=(2, 3, 4),
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict({})
        result = transform(data)

        # 2 * 3 * 4 = 24 points
        assert result["grid"].shape == (24, 3)

    def test_preserves_existing_data(self):
        """Test that existing data is preserved."""
        transform = CreateGrid(
            output_key="grid",
            resolution=(2, 2, 2),
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        existing = torch.randn(10, 5)
        data = TensorDict({"existing": existing})

        result = transform(data)

        assert "existing" in result
        torch.testing.assert_close(result["existing"], existing)
        assert "grid" in result

    def test_repr(self):
        """Test string representation."""
        transform = CreateGrid(
            output_key="my_grid",
            resolution=(8, 8, 8),
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        repr_str = repr(transform)
        assert "CreateGrid" in repr_str
        assert "my_grid" in repr_str
        assert "(8, 8, 8)" in repr_str


# ============================================================================
# KNearestNeighbors Tests
# ============================================================================


class TestKNearestNeighbors:
    """Tests for KNearestNeighbors transform."""

    def test_basic_knn(self):
        """Test basic k-NN computation."""
        transform = KNearestNeighbors(
            points_key="points",
            queries_key="queries",
            k=3,
            output_prefix="neighbors",
            drop_first_neighbor=False,
        )

        # Create structured points for predictable results
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        queries = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        data = TensorDict({"points": points, "queries": queries})

        result = transform(data)

        # Check output keys exist
        assert "neighbors_indices" in result
        assert "neighbors_distances" in result
        assert "neighbors_coords" in result

        # Check shapes
        assert result["neighbors_indices"].shape == (2, 3)
        assert result["neighbors_distances"].shape == (2, 3)
        # coords excludes self, so k-1 neighbors
        assert result["neighbors_coords"].shape == (2, 3, 3)

    def test_knn_with_extract_keys(self):
        """Test k-NN with additional feature extraction."""
        transform = KNearestNeighbors(
            points_key="points",
            queries_key="queries",
            k=4,
            output_prefix="nn",
            extract_keys=["normals", "values"],
        )

        n_points = 20
        points = torch.randn(n_points, 3)
        queries = torch.randn(5, 3)
        normals = torch.randn(n_points, 3)
        values = torch.randn(n_points, 1)

        data = TensorDict(
            {"points": points, "queries": queries, "normals": normals, "values": values}
        )

        result = transform(data)

        # Check extracted features
        assert "nn_normals" in result
        assert "nn_values" in result

        # Shape should be (n_queries, k-1, feature_dim) since self is excluded
        assert result["nn_normals"].shape == (5, 4, 3)
        assert result["nn_values"].shape == (5, 4, 1)

    def test_knn_missing_points_raises(self):
        """Test that missing points key raises KeyError."""
        transform = KNearestNeighbors(
            points_key="points",
            queries_key="queries",
            k=3,
        )

        data = TensorDict({"queries": torch.randn(5, 3)})

        with pytest.raises(KeyError, match="Points key"):
            transform(data)

    def test_knn_missing_queries_raises(self):
        """Test that missing queries key raises KeyError."""
        transform = KNearestNeighbors(
            points_key="points",
            queries_key="queries",
            k=3,
        )

        data = TensorDict({"points": torch.randn(10, 3)})

        with pytest.raises(KeyError, match="Queries key"):
            transform(data)

    def test_knn_self_query(self):
        """Test k-NN where queries are a subset of points."""
        transform = KNearestNeighbors(
            points_key="points",
            queries_key="queries",
            k=5,
            output_prefix="neighbors",
        )

        points = torch.randn(50, 3)
        queries = points[:10]  # First 10 points as queries

        data = TensorDict({"points": points, "queries": queries})

        result = transform(data)

        # First neighbor should be self (distance ~0)
        assert result["neighbors_distances"][:, 0].max() < 1e-5

    def test_repr(self):
        """Test string representation."""
        transform = KNearestNeighbors(
            points_key="surface_points",
            queries_key="query_points",
            k=11,
        )

        repr_str = repr(transform)
        assert "KNearestNeighbors" in repr_str
        assert "surface_points" in repr_str
        assert "query_points" in repr_str
        assert "11" in repr_str


# ============================================================================
# CenterOfMass Tests
# ============================================================================


class TestCenterOfMass:
    """Tests for CenterOfMass transform."""

    def test_basic_center_of_mass(self):
        """Test basic weighted center of mass computation."""
        transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        # Symmetric case: center should be at origin
        coords = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        areas = torch.tensor([1.0, 1.0, 1.0, 1.0])

        data = TensorDict({"coords": coords, "areas": areas})

        result = transform(data)

        assert "com" in result
        assert result["com"].shape == torch.Size(
            [
                3,
            ]
        )
        torch.testing.assert_close(
            result["com"], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
        )

    def test_weighted_center_of_mass(self):
        """Test center of mass with non-uniform weights."""
        transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        # Two points, one with much larger area
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        areas = torch.tensor([9.0, 1.0])  # First point has 9x weight

        data = TensorDict({"coords": coords, "areas": areas})

        result = transform(data)

        # Center should be closer to first point
        expected_x = (0.0 * 9.0 + 10.0 * 1.0) / 10.0  # = 1.0
        torch.testing.assert_close(
            result["com"], torch.tensor([expected_x, 0.0, 0.0]), atol=1e-6, rtol=1e-6
        )

    def test_single_point(self):
        """Test center of mass with single point."""
        transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        coords = torch.tensor([[5.0, 3.0, 2.0]])
        areas = torch.tensor([1.0])

        data = TensorDict({"coords": coords, "areas": areas})

        result = transform(data)

        torch.testing.assert_close(result["com"], torch.tensor([5.0, 3.0, 2.0]))

    def test_missing_coords_raises(self):
        """Test that missing coords key raises KeyError."""
        transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        data = TensorDict({"areas": torch.tensor([1.0, 1.0])})

        with pytest.raises(KeyError, match="Coordinates key"):
            transform(data)

    def test_missing_areas_raises(self):
        """Test that missing areas key raises KeyError."""
        transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        data = TensorDict({"coords": torch.randn(10, 3)})

        with pytest.raises(KeyError, match="Areas key"):
            transform(data)

    def test_preserves_existing_data(self):
        """Test that existing data is preserved."""
        transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        coords = torch.randn(50, 3)
        areas = torch.rand(50)
        other_data = torch.randn(10, 5)

        data = TensorDict({"coords": coords, "areas": areas, "other": other_data})

        result = transform(data)

        assert "other" in result
        torch.testing.assert_close(result["other"], other_data)

    def test_repr(self):
        """Test string representation."""
        transform = CenterOfMass(
            coords_key="stl_centers",
            areas_key="stl_areas",
            output_key="center_of_mass",
        )

        repr_str = repr(transform)
        assert "CenterOfMass" in repr_str
        assert "stl_centers" in repr_str
        assert "center_of_mass" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestSpatialTransformIntegration:
    """Integration tests combining multiple spatial transforms."""

    def test_center_of_mass_then_translate(self):
        """Test computing CoM and using it for translation."""
        from physicsnemo.datapipes.transforms.geometric import Translate

        # First compute center of mass
        com_transform = CenterOfMass(
            coords_key="coords",
            areas_key="areas",
            output_key="com",
        )

        # Then translate to center at origin
        translate_transform = Translate(
            input_keys=["coords"],
            center_key_or_value="com",
            subtract=True,
        )

        coords = torch.tensor(
            [
                [10.0, 10.0, 10.0],
                [12.0, 10.0, 10.0],
                [10.0, 12.0, 10.0],
                [10.0, 10.0, 12.0],
            ]
        )
        areas = torch.tensor([1.0, 1.0, 1.0, 1.0])

        data = TensorDict({"coords": coords, "areas": areas})

        # Apply transforms
        data = com_transform(data)
        data = translate_transform(data)

        # Center of mass should now be at origin
        new_com = (data["coords"] * areas.unsqueeze(-1)).sum(dim=0) / areas.sum()
        torch.testing.assert_close(
            new_com, torch.tensor([0.0, 0.0, 0.0]), atol=1e-5, rtol=1e-5
        )

    def test_create_grid_then_filter(self):
        """Test creating grid and filtering to subregion."""
        grid_transform = CreateGrid(
            output_key="grid",
            resolution=(10, 10, 10),
            bbox_min=torch.tensor([-1.0, -1.0, -1.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        filter_transform = BoundingBoxFilter(
            input_keys=["grid"],
            bbox_min=torch.tensor([0.0, 0.0, 0.0]),
            bbox_max=torch.tensor([1.0, 1.0, 1.0]),
        )

        data = TensorDict({})

        # Create full grid
        data = grid_transform(data)
        full_grid_size = data["grid"].shape[0]

        # Filter to positive octant
        data = filter_transform(data)

        # Should have roughly 1/8 of points (plus some edge cases)
        assert data["grid"].shape[0] < full_grid_size
        assert data["grid"].shape[0] > 0

        # All remaining points should be in positive octant
        assert (data["grid"] > 0).all()
