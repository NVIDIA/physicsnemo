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

"""Tests for converting between cell data and point data.

Tests validate data conversion across spatial dimensions, manifold dimensions,
and compute backends, ensuring correct averaging and preservation of data types.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities._cache import get_cached, set_cached

### Helper Functions ###


def create_simple_mesh(
    n_spatial_dims: int, n_manifold_dims: int, device: torch.device | str = "cpu"
):
    """Create a simple mesh for testing."""
    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 1:
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [0.5, 1.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 2:
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.5]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 3:
        if n_spatial_dims == 3:
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
            raise ValueError("3-simplices require 3D embedding space")
    else:
        raise ValueError(f"Unsupported {n_manifold_dims=}")

    return Mesh(points=points, cells=cells)


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


class TestCellDataToPointData:
    """Tests for cell_data_to_point_data method."""

    def test_simple_triangle_mesh(self):
        """Test cell to point conversion on a simple triangle mesh."""
        ### Create mesh with two triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0, 200.0])},
        )

        ### Convert
        result = mesh.cell_data_to_point_data()

        ### Check that both cell and point data exist
        assert "temperature" in result.cell_data
        assert "temperature" in result.point_data

        ### Check point data values
        # Point 0: only in cell 0 -> 100.0
        assert torch.allclose(result.point_data["temperature"][0], torch.tensor(100.0))
        # Point 1: in cells 0 and 1 -> (100 + 200) / 2 = 150.0
        assert torch.allclose(result.point_data["temperature"][1], torch.tensor(150.0))
        # Point 2: in cells 0 and 1 -> 150.0
        assert torch.allclose(result.point_data["temperature"][2], torch.tensor(150.0))
        # Point 3: only in cell 1 -> 200.0
        assert torch.allclose(result.point_data["temperature"][3], torch.tensor(200.0))

    def test_multidimensional_data(self):
        """Test conversion of multi-dimensional cell data."""
        ### Create mesh with vector cell data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"velocity": torch.tensor([[1.0, 2.0, 3.0]])},
        )

        ### Convert
        result = mesh.cell_data_to_point_data()

        ### All points should get the same vector
        assert result.point_data["velocity"].shape == (3, 3)
        for i in range(3):
            assert torch.allclose(
                result.point_data["velocity"][i],
                torch.tensor([1.0, 2.0, 3.0]),
            )

    def test_preserves_original_data(self):
        """Test that original cell data is preserved."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        original_value = torch.tensor([42.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": original_value.clone()},
        )

        result = mesh.cell_data_to_point_data()

        ### Original cell data unchanged
        assert torch.allclose(result.cell_data["value"], original_value)

    def test_key_conflict_raises_error(self):
        """Test that duplicate keys raise error by default."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0])},
            cell_data={"value": torch.tensor([10.0])},
        )

        ### Should raise error
        with pytest.raises(ValueError):
            mesh.cell_data_to_point_data()

    def test_overwrite_keys(self):
        """Test that overwrite_keys=True allows overwriting."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0])},
            cell_data={"value": torch.tensor([100.0])},
        )

        ### Should not raise error
        result = mesh.cell_data_to_point_data(overwrite_keys=True)

        ### Point data should be overwritten
        assert torch.allclose(
            result.point_data["value"], torch.tensor([100.0, 100.0, 100.0])
        )

    def test_skips_cached_properties(self):
        """Test that cached properties (under "_cache") are skipped."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Access a cached property
        _ = mesh.cell_centroids  # This creates cache

        ### Convert
        result = mesh.cell_data_to_point_data()

        ### Cached property should not be in point_data (should not leak from cell_data)
        assert get_cached(result.point_data, "centroids") is None


class TestPointDataToCellData:
    """Tests for point_data_to_cell_data method."""

    def test_simple_triangle_mesh(self):
        """Test point to cell conversion on a simple triangle mesh."""
        ### Create mesh with point data
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"temperature": torch.tensor([100.0, 200.0, 300.0, 400.0])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Check that both point and cell data exist
        assert "temperature" in result.point_data
        assert "temperature" in result.cell_data

        ### Check cell data values
        # Cell 0: vertices [0, 1, 2] -> (100 + 200 + 300) / 3 = 200.0
        assert torch.allclose(result.cell_data["temperature"][0], torch.tensor(200.0))
        # Cell 1: vertices [1, 3, 2] -> (200 + 400 + 300) / 3 = 300.0
        assert torch.allclose(result.cell_data["temperature"][1], torch.tensor(300.0))

    def test_multidimensional_data(self):
        """Test conversion of multi-dimensional point data."""
        ### Create mesh with vector point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"velocity": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Cell should get average of vertex vectors
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).mean(dim=0)
        assert torch.allclose(result.cell_data["velocity"][0], expected)

    def test_preserves_original_data(self):
        """Test that original point data is preserved."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        original_value = torch.tensor([1.0, 2.0, 3.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": original_value.clone()},
        )

        result = mesh.point_data_to_cell_data()

        ### Original point data unchanged
        assert torch.allclose(result.point_data["value"], original_value)

    def test_key_conflict_raises_error(self):
        """Test that duplicate keys raise error by default."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0])},
            cell_data={"value": torch.tensor([10.0])},
        )

        ### Should raise error
        with pytest.raises(ValueError):
            mesh.point_data_to_cell_data()

    def test_overwrite_keys(self):
        """Test that overwrite_keys=True allows overwriting."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([10.0, 20.0, 30.0])},
            cell_data={"value": torch.tensor([999.0])},
        )

        ### Should not raise error
        result = mesh.point_data_to_cell_data(overwrite_keys=True)

        ### Cell data should be overwritten with average of point data
        expected = torch.tensor([10.0, 20.0, 30.0]).mean()
        assert torch.allclose(result.cell_data["value"], expected)

    def test_skips_cached_properties(self):
        """Test that cached properties (under "_cache") are skipped."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        set_cached(mesh.point_data, "test_cached_value", torch.tensor([1.0, 2.0, 3.0]))

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Cached property should not be converted to cell_data
        assert get_cached(result.cell_data, "test_cached_value") is None

    def test_3d_tetrahedral_mesh(self):
        """Test on 3D tetrahedral mesh."""
        ### Create tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        )

        ### Convert
        result = mesh.point_data_to_cell_data()

        ### Cell value should be average of vertex values
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).mean()
        assert torch.allclose(result.cell_data["value"][0], expected)


class TestRoundTripConversion:
    """Test round-trip conversion between cell and point data."""

    def test_cell_to_point_to_cell(self):
        """Test converting cell -> point -> cell."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        original_value = torch.tensor([42.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": original_value.clone()},
        )

        ### Convert cell -> point -> cell
        result = mesh.cell_data_to_point_data()
        result = result.point_data_to_cell_data(overwrite_keys=True)

        ### For single cell mesh, should recover original value
        assert torch.allclose(result.cell_data["value"], original_value)

    def test_point_to_cell_to_point(self):
        """Test converting point -> cell -> point."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        original_values = torch.tensor([10.0, 20.0, 30.0])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": original_values.clone()},
        )

        ### Convert point -> cell -> point
        result = mesh.point_data_to_cell_data()
        result = result.cell_data_to_point_data(overwrite_keys=True)

        ### For single cell mesh, all points should get the average value
        avg = original_values.mean()
        assert torch.allclose(result.point_data["value"], torch.tensor([avg, avg, avg]))


### Parametrized Tests for Exhaustive Dimensional Coverage ###


class TestDataConversionParametrized:
    """Parametrized tests for data conversion across all dimensions and backends."""

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_cell_to_point_basic_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test basic cell-to-point conversion across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add scalar cell data
        cell_values = torch.arange(mesh.n_cells, dtype=torch.float32, device=device)
        mesh.cell_data["value"] = cell_values

        result = mesh.cell_data_to_point_data()

        # Verify data was converted
        assert "value" in result.point_data, "Point data should contain 'value'"
        assert result.point_data["value"].shape[0] == mesh.n_points

        # Verify device consistency
        assert_on_device(result.point_data["value"], device)

        # Verify original data preserved
        assert torch.equal(result.cell_data["value"], cell_values)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_point_to_cell_basic_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test basic point-to-cell conversion across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add scalar point data
        point_values = torch.arange(mesh.n_points, dtype=torch.float32, device=device)
        mesh.point_data["value"] = point_values

        result = mesh.point_data_to_cell_data()

        # Verify data was converted
        assert "value" in result.cell_data, "Cell data should contain 'value'"
        assert result.cell_data["value"].shape[0] == mesh.n_cells

        # Verify device consistency
        assert_on_device(result.cell_data["value"], device)

        # Verify original data preserved
        assert torch.equal(result.point_data["value"], point_values)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 2),
            (3, 2),
            (3, 3),
        ],
    )
    def test_multidimensional_cell_to_point_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test multidimensional data conversion (vectors) across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add vector cell data
        vectors = torch.randn(mesh.n_cells, n_spatial_dims, device=device)
        mesh.cell_data["velocity"] = vectors

        result = mesh.cell_data_to_point_data()

        # Verify shape
        assert result.point_data["velocity"].shape == (mesh.n_points, n_spatial_dims)

        # Verify device
        assert_on_device(result.point_data["velocity"], device)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 2),
            (3, 2),
            (3, 3),
        ],
    )
    def test_multidimensional_point_to_cell_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test multidimensional data conversion (vectors) across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add vector point data
        vectors = torch.randn(mesh.n_points, n_spatial_dims, device=device)
        mesh.point_data["velocity"] = vectors

        result = mesh.point_data_to_cell_data()

        # Verify shape
        assert result.cell_data["velocity"].shape == (mesh.n_cells, n_spatial_dims)

        # Verify device
        assert_on_device(result.cell_data["velocity"], device)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_cached_properties_skipped_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test that cached properties are skipped across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Access cached properties to populate them
        _ = mesh.cell_centroids
        _ = mesh.cell_areas

        # Convert cell to point
        result = mesh.cell_data_to_point_data()

        # Cached properties should not be converted
        assert get_cached(result.point_data, "centroids") is None
        assert get_cached(result.point_data, "areas") is None

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_round_trip_consistency_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test round-trip conversion consistency across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Add cell data
        cell_values = (
            torch.arange(mesh.n_cells, dtype=torch.float32, device=device) * 10.0
        )
        mesh.cell_data["value"] = cell_values

        # Round trip: cell → point → cell
        intermediate = mesh.cell_data_to_point_data()
        result = intermediate.point_data_to_cell_data(overwrite_keys=True)

        # Values should be approximately the same (averaging may introduce small changes)
        # But device should be preserved
        assert_on_device(result.cell_data["value"], device)
        assert result.cell_data["value"].shape[0] == mesh.n_cells

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 2),
            (3, 2),
            (3, 3),
        ],
    )
    def test_empty_data_dict_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test conversion with no data across dimensions."""
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # No data to convert
        result1 = mesh.cell_data_to_point_data()
        result2 = mesh.point_data_to_cell_data()

        # Should work without errors
        assert result1.n_points == mesh.n_points
        assert result2.n_cells == mesh.n_cells

        # Devices should be preserved
        assert_on_device(result1.points, device)
        assert_on_device(result2.points, device)
