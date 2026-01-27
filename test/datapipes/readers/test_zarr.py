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

"""Extended tests for ZarrReader to improve coverage."""

import numpy as np
import pytest
import torch

import physicsnemo.datapipes as dp

# ============================================================================
# ZarrReader Coordinated Subsampling Tests
# ============================================================================


class TestZarrReaderCoordinatedSubsampling:
    """Tests for ZarrReader coordinated subsampling feature."""

    @pytest.fixture
    def zarr_large_data_dir(self, tmp_path):
        """Create Zarr data with larger arrays for subsampling tests."""
        zarr = pytest.importorskip("zarr")

        for i in range(3):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            root = zarr.open(group_path, mode="w")
            root.create_array(
                "volume_coords", data=np.random.randn(1000, 3).astype(np.float32)
            )
            root.create_array(
                "volume_fields", data=np.random.randn(1000, 8).astype(np.float32)
            )
            root.create_array("metadata_scalar", data=np.array([i], dtype=np.float32))

        return tmp_path

    def test_coordinated_subsampling_basic(self, zarr_large_data_dir):
        """Test basic coordinated subsampling."""
        reader = dp.ZarrReader(
            zarr_large_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={
                "n_points": 100,
                "target_keys": ["volume_coords", "volume_fields"],
            },
        )

        data, metadata = reader[0]

        # Subsampled arrays should have 100 points
        assert data["volume_coords"].shape == (100, 3)
        assert data["volume_fields"].shape == (100, 8)

        # Non-target arrays should have original size
        assert data["metadata_scalar"].shape == (1,)

    def test_coordinated_subsampling_consistency(self, zarr_large_data_dir):
        """Test that coordinated subsampling is consistent across target keys."""
        reader = dp.ZarrReader(
            zarr_large_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={
                "n_points": 200,
                "target_keys": ["volume_coords", "volume_fields"],
            },
        )

        # Load same sample multiple times - subsampling is random but should
        # be applied consistently within a single __getitem__ call
        data, _ = reader[0]

        # Both arrays should have same number of points
        assert data["volume_coords"].shape[0] == data["volume_fields"].shape[0]

    def test_coordinated_subsampling_too_few_points(self, zarr_large_data_dir):
        """Test that requesting more points than available raises error."""
        reader = dp.ZarrReader(
            zarr_large_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={
                "n_points": 5000,  # More than 1000 available
                "target_keys": ["volume_coords"],
            },
        )

        with pytest.raises(ValueError, match="less than"):
            reader[0]

    def test_coordinated_subsampling_single_group_mode(self, tmp_path):
        """Test coordinated subsampling in single group mode."""
        zarr = pytest.importorskip("zarr")

        # Create single group with samples indexed along first dimension
        path = tmp_path / "data.zarr"
        root = zarr.open(path, mode="w")
        root.create_array("coords", data=np.random.randn(10, 500, 3).astype(np.float32))
        root.create_array(
            "features", data=np.random.randn(10, 500, 8).astype(np.float32)
        )

        reader = dp.ZarrReader(
            path,
            coordinated_subsampling={
                "n_points": 100,
                "target_keys": ["coords", "features"],
            },
        )

        assert len(reader) == 10

        data, _ = reader[0]

        # Should be subsampled along second dimension
        assert data["coords"].shape == (100, 3)
        assert data["features"].shape == (100, 8)

    def test_supports_coordinated_subsampling_property(self, zarr_large_data_dir):
        """Test _supports_coordinated_subsampling property."""
        reader = dp.ZarrReader(zarr_large_data_dir, group_pattern="sample_*.zarr")

        assert reader._supports_coordinated_subsampling is True


# ============================================================================
# ZarrReader Attribute Loading Tests
# ============================================================================


class TestZarrReaderAttributes:
    """Tests for ZarrReader attribute loading feature."""

    @pytest.fixture
    def zarr_with_attrs(self, tmp_path):
        """Create Zarr data with various attributes."""
        zarr = pytest.importorskip("zarr")

        for i in range(3):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            root = zarr.open(group_path, mode="w")

            # Create arrays
            root.create_array("data", data=np.random.randn(50, 3).astype(np.float32))

            # Add various attribute types (zarr v3 requires JSON-serializable values)
            root.attrs["sample_id"] = i
            root.attrs["scale_factor"] = float(i + 1) * 0.5
            root.attrs["is_valid"] = True
            root.attrs["dimensions"] = [10, 20, 30]
            root.attrs["center"] = [0.0, 0.0, 0.0]  # Use list instead of ndarray

        return tmp_path

    def test_load_scalar_int_attribute(self, zarr_with_attrs):
        """Test loading integer scalar attribute."""
        reader = dp.ZarrReader(
            zarr_with_attrs,
            fields=["data", "sample_id"],
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[1]

        assert "sample_id" in data
        assert data["sample_id"].item() == 1

    def test_load_scalar_float_attribute(self, zarr_with_attrs):
        """Test loading float scalar attribute."""
        reader = dp.ZarrReader(
            zarr_with_attrs,
            fields=["data", "scale_factor"],
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[0]

        assert "scale_factor" in data
        torch.testing.assert_close(
            data["scale_factor"], torch.tensor(0.5), atol=1e-6, rtol=1e-6
        )

    def test_load_bool_attribute(self, zarr_with_attrs):
        """Test loading boolean attribute."""
        reader = dp.ZarrReader(
            zarr_with_attrs,
            fields=["data", "is_valid"],
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[0]

        assert "is_valid" in data
        assert data["is_valid"].item() == True  # noqa: E712

    def test_load_list_attribute(self, zarr_with_attrs):
        """Test loading list attribute."""
        reader = dp.ZarrReader(
            zarr_with_attrs,
            fields=["data", "dimensions"],
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[0]

        assert "dimensions" in data
        torch.testing.assert_close(data["dimensions"], torch.tensor([10, 20, 30]))

    def test_load_list_as_array_attribute(self, zarr_with_attrs):
        """Test loading list attribute that becomes a tensor."""
        reader = dp.ZarrReader(
            zarr_with_attrs,
            fields=["data", "center"],
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[0]

        assert "center" in data
        torch.testing.assert_close(data["center"], torch.tensor([0.0, 0.0, 0.0]))

    def test_string_attribute_raises(self, tmp_path):
        """Test that string attributes raise TypeError."""
        zarr = pytest.importorskip("zarr")

        group_path = tmp_path / "sample.zarr"
        root = zarr.open(group_path, mode="w")
        root.create_array("data", data=np.random.randn(10, 3).astype(np.float32))
        root.attrs["name"] = "test_sample"

        reader = dp.ZarrReader(
            tmp_path,
            fields=["data", "name"],
            group_pattern="*.zarr",
        )

        with pytest.raises(TypeError, match="string"):
            reader[0]


# ============================================================================
# ZarrReader Default Values Tests
# ============================================================================


class TestZarrReaderDefaultValues:
    """Tests for ZarrReader default values feature."""

    @pytest.fixture
    def zarr_data_dir(self, tmp_path):
        """Create basic Zarr test data."""
        zarr = pytest.importorskip("zarr")

        for i in range(3):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            root = zarr.open(group_path, mode="w")
            root.create_array(
                "positions", data=np.random.randn(50, 3).astype(np.float32)
            )

        return tmp_path

    def test_default_value_for_missing_field(self, zarr_data_dir):
        """Test that default values are used for missing fields."""
        default_tensor = torch.ones(10, 5)

        reader = dp.ZarrReader(
            zarr_data_dir,
            fields=["positions", "missing_field"],
            default_values={"missing_field": default_tensor},
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[0]

        assert "missing_field" in data
        torch.testing.assert_close(data["missing_field"], default_tensor)

    def test_missing_field_without_default_raises(self, zarr_data_dir):
        """Test that missing field without default raises KeyError."""
        reader = dp.ZarrReader(
            zarr_data_dir,
            fields=["positions", "nonexistent"],
            group_pattern="sample_*.zarr",
        )

        with pytest.raises(KeyError, match="nonexistent"):
            reader[0]

    def test_default_value_not_used_when_field_exists(self, zarr_data_dir):
        """Test that default is not used when field exists."""
        default_tensor = torch.zeros(50, 3)

        reader = dp.ZarrReader(
            zarr_data_dir,
            fields=["positions"],
            default_values={"positions": default_tensor},
            group_pattern="sample_*.zarr",
        )

        data, _ = reader[0]

        # Should have actual data, not zeros
        assert not torch.allclose(data["positions"], default_tensor)


# ============================================================================
# ZarrReader Cache and Close Tests
# ============================================================================


class TestZarrReaderCacheAndClose:
    """Tests for ZarrReader caching and close functionality."""

    @pytest.fixture
    def zarr_data_dir(self, tmp_path):
        """Create basic Zarr test data."""
        zarr = pytest.importorskip("zarr")

        for i in range(3):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            root = zarr.open(group_path, mode="w")
            root.create_array("data", data=np.random.randn(50, 3).astype(np.float32))

        return tmp_path

    def test_cache_stores_enabled_by_default(self, zarr_data_dir):
        """Test that store caching is enabled by default."""
        reader = dp.ZarrReader(zarr_data_dir, group_pattern="sample_*.zarr")

        assert reader._cache_stores is True

    def test_cache_stores_disabled(self, zarr_data_dir):
        """Test that store caching can be disabled."""
        reader = dp.ZarrReader(
            zarr_data_dir,
            group_pattern="sample_*.zarr",
            cache_stores=False,
        )

        assert reader._cache_stores is False

        # Should still work
        data, _ = reader[0]
        assert "data" in data

    def test_close_clears_cache(self, zarr_data_dir):
        """Test that close() clears cached stores."""
        reader = dp.ZarrReader(zarr_data_dir, group_pattern="sample_*.zarr")

        # Access data to populate cache
        _ = reader[0]
        _ = reader[1]

        # Cache should have entries
        assert len(reader._cached_stores) > 0

        reader.close()

        # Cache should be cleared
        assert len(reader._cached_stores) == 0

    def test_context_manager_closes(self, zarr_data_dir):
        """Test that context manager calls close()."""
        with dp.ZarrReader(zarr_data_dir, group_pattern="sample_*.zarr") as reader:
            _ = reader[0]
            assert len(reader._cached_stores) > 0

        # After context exit, cache should be cleared
        assert len(reader._cached_stores) == 0


# ============================================================================
# ZarrReader Single Group Mode Tests
# ============================================================================


class TestZarrReaderSingleGroupMode:
    """Tests for ZarrReader single group mode."""

    @pytest.fixture
    def single_zarr_group(self, tmp_path):
        """Create a single Zarr group with samples along first dimension."""
        zarr = pytest.importorskip("zarr")

        path = tmp_path / "data.zarr"
        root = zarr.open(path, mode="w")
        root.create_array("inputs", data=np.random.randn(20, 64).astype(np.float32))
        root.create_array("targets", data=np.random.randn(20, 10).astype(np.float32))
        root.attrs["dataset_name"] = "test"  # Will be ignored (string)

        return path

    def test_single_group_mode_detection(self, single_zarr_group):
        """Test that single group mode is detected."""
        reader = dp.ZarrReader(single_zarr_group)

        assert reader._single_group_mode is True
        assert len(reader) == 20

    def test_single_group_mode_indexing(self, single_zarr_group):
        """Test indexing in single group mode."""
        reader = dp.ZarrReader(single_zarr_group)

        data, metadata = reader[5]

        assert data["inputs"].shape == (64,)
        assert data["targets"].shape == (10,)
        assert metadata["sample_index"] == 5

    def test_single_group_mode_metadata(self, single_zarr_group):
        """Test metadata in single group mode."""
        reader = dp.ZarrReader(single_zarr_group)

        _, metadata = reader[0]

        assert "source_file" in metadata
        assert "sample_index" in metadata
        assert metadata["sample_index"] == 0

    def test_single_group_vs_directory_mode(self, tmp_path):
        """Test that directory mode is used when appropriate."""
        zarr = pytest.importorskip("zarr")

        # Create directory with multiple groups
        for i in range(3):
            group_path = tmp_path / f"sample_{i}.zarr"
            root = zarr.open(group_path, mode="w")
            root.create_array("data", data=np.random.randn(50).astype(np.float32))

        reader = dp.ZarrReader(tmp_path, group_pattern="sample_*.zarr")

        assert reader._single_group_mode is False
        assert len(reader) == 3


# ============================================================================
# ZarrReader Repr Tests
# ============================================================================


class TestZarrReaderRepr:
    """Tests for ZarrReader string representation."""

    @pytest.fixture
    def zarr_data_dir(self, tmp_path):
        """Create basic Zarr test data."""
        zarr = pytest.importorskip("zarr")

        for i in range(5):
            group_path = tmp_path / f"sample_{i:03d}.zarr"
            root = zarr.open(group_path, mode="w")
            root.create_array("data", data=np.random.randn(50, 3).astype(np.float32))

        return tmp_path

    def test_repr_basic(self, zarr_data_dir):
        """Test basic repr output."""
        reader = dp.ZarrReader(zarr_data_dir, group_pattern="sample_*.zarr")

        repr_str = repr(reader)

        assert "ZarrReader" in repr_str
        assert "len=5" in repr_str
        assert "cache_stores=True" in repr_str

    def test_repr_with_fields(self, zarr_data_dir):
        """Test repr with field selection."""
        reader = dp.ZarrReader(
            zarr_data_dir,
            fields=["data"],
            group_pattern="sample_*.zarr",
        )

        repr_str = repr(reader)

        assert "data" in repr_str

    def test_repr_with_subsampling(self, zarr_data_dir):
        """Test repr with coordinated subsampling."""
        reader = dp.ZarrReader(
            zarr_data_dir,
            group_pattern="sample_*.zarr",
            coordinated_subsampling={"n_points": 25, "target_keys": ["data"]},
        )

        repr_str = repr(reader)

        assert "subsampling=25" in repr_str
