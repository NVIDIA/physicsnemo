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


"""
Tests for the NumpyReader.

Tests reading from .npz files, directories, and coordinated subsampling.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from physicsnemo.datapipes.readers import NumpyReader


class TestNumpyReaderBasic:
    """Basic functionality tests for NumpyReader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_npz_file(self):
        """Test reading from a single .npz file."""
        # Create test data
        coords = np.random.randn(20, 3).astype(np.float32)
        features = np.random.randn(20, 5).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords, features=features)

        # Create reader
        reader = NumpyReader(npz_path, fields=["coords", "features"])

        # Check properties
        assert len(reader) == 20
        assert set(reader.field_names) == {"coords", "features"}

        # Load sample
        data, metadata = reader[0]
        assert "coords" in data
        assert "features" in data
        assert data["coords"].shape == (3,)
        assert data["features"].shape == (5,)

    def test_single_npz_file_load_all_fields(self):
        """Test reading all fields from a single .npz file when fields=None."""
        # Create test data
        coords = np.random.randn(20, 3).astype(np.float32)
        features = np.random.randn(20, 5).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords, features=features)

        # Create reader without specifying fields
        reader = NumpyReader(npz_path)

        # Should load all fields
        assert set(reader.field_names) == {"coords", "features"}

        data, metadata = reader[0]
        assert "coords" in data
        assert "features" in data

    def test_directory_of_npz_files(self):
        """Test reading from a directory of .npz files."""
        # Create test data
        for i in range(5):
            coords = np.random.randn(100, 3).astype(np.float32)
            features = np.random.randn(100, 2).astype(np.float32)

            npz_path = self.temp_path / f"sample_{i:03d}.npz"
            np.savez(npz_path, coords=coords, features=features)

        # Create reader
        reader = NumpyReader(
            self.temp_path, file_pattern="sample_*.npz", fields=["coords", "features"]
        )

        # Check properties
        assert len(reader) == 5
        assert set(reader.field_names) == {"coords", "features"}

        # Load sample
        data, metadata = reader[0]
        assert data["coords"].shape == (100, 3)
        assert data["features"].shape == (100, 2)

    def test_directory_load_all_fields(self):
        """Test reading all fields from directory when fields=None."""
        # Create test data
        for i in range(3):
            coords = np.random.randn(50, 3).astype(np.float32)
            features = np.random.randn(50, 2).astype(np.float32)

            npz_path = self.temp_path / f"sample_{i:03d}.npz"
            np.savez(npz_path, coords=coords, features=features)

        # Create reader without specifying fields
        reader = NumpyReader(self.temp_path, file_pattern="sample_*.npz")

        # Should load all fields
        assert set(reader.field_names) == {"coords", "features"}

        data, metadata = reader[0]
        assert "coords" in data
        assert "features" in data

    def test_default_values(self):
        """Test optional keys with default values."""
        # Create test data with only some keys
        coords = np.random.randn(10, 100, 3).astype(np.float32)
        features = np.random.randn(10, 100, 2).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords, features=features)
        # Note: no "normals" key

        # Create reader with optional key
        default_normals = torch.zeros(100, 3)
        reader = NumpyReader(
            npz_path,
            fields=["coords", "features", "normals"],
            default_values={"normals": default_normals},
        )

        # Load sample
        data, metadata = reader[0]
        assert "coords" in data
        assert "features" in data
        assert "normals" in data

        # Check that default was used
        assert torch.allclose(data["normals"], default_normals)

    def test_unsupported_file_type(self):
        """Test that .npy files raise an error."""
        npy_path = self.temp_path / "data.npy"
        np.save(npy_path, np.random.randn(10, 3, 4))

        with pytest.raises(ValueError, match="Unsupported file type"):
            NumpyReader(npy_path)


class TestNumpyReaderCoordinatedSubsampling:
    """Test coordinated subsampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_coordinated_subsampling_directory_npz(self):
        """Test coordinated subsampling in directory mode."""
        # Create test data with large arrays
        n_samples = 5
        n_points = 100000
        subsample_points = 10000

        for i in range(n_samples):
            coords = np.random.randn(n_points, 3).astype(np.float32)
            features = np.random.randn(n_points, 4).astype(np.float32)
            areas = np.random.rand(n_points).astype(np.float32)

            npz_path = self.temp_path / f"sample_{i:03d}.npz"
            np.savez(npz_path, coords=coords, features=features, areas=areas)

        # Create reader with coordinated subsampling
        reader = NumpyReader(
            self.temp_path,
            file_pattern="sample_*.npz",
            fields=["coords", "features", "areas"],
            coordinated_subsampling={
                "n_points": subsample_points,
                "target_keys": ["coords", "features"],
            },
        )

        # Load sample
        data, metadata = reader[0]

        # Check that subsampled arrays have correct size
        assert data["coords"].shape == (subsample_points, 3)
        assert data["features"].shape == (subsample_points, 4)

        # Non-target keys should be full size
        assert data["areas"].shape == (n_points,)

    def test_supports_coordinated_subsampling(self):
        """Test that coordinated subsampling is only supported in directory mode."""
        # Directory mode: supported
        npz_path = self.temp_path / "sample_000.npz"
        np.savez(npz_path, coords=np.random.randn(100, 3))

        reader_dir = NumpyReader(self.temp_path, file_pattern="sample_*.npz")
        assert reader_dir._supports_coordinated_subsampling is True

        # Single .npz file mode: not supported
        single_npz_path = self.temp_path / "single.npz"
        np.savez(single_npz_path, coords=np.random.randn(10, 100, 3))

        reader_single = NumpyReader(single_npz_path)
        assert reader_single._supports_coordinated_subsampling is False

        # Config is ignored for readers that don't support it
        reader_with_config = NumpyReader(
            single_npz_path,
            coordinated_subsampling={"n_points": 50, "target_keys": ["coords"]},
        )
        # Config is stored but will be ignored during loading
        assert reader_with_config._coordinated_subsampling_config is not None


class TestNumpyReaderMemoryManagement:
    """Test memory management and cleanup."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pin_memory(self):
        """Test pin_memory functionality."""
        coords = np.random.randn(10, 3, 4).astype(np.float32)
        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        # Create reader with pin_memory
        reader = NumpyReader(npz_path, pin_memory=True)
        data, metadata = reader[0]

        # Check that tensor is pinned
        assert data["coords"].is_pinned()

    def test_close_handles(self):
        """Test that file handles are properly closed."""
        coords = np.random.randn(20, 3).astype(np.float32)
        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        reader = NumpyReader(npz_path)
        _ = reader[0]

        # Close should not raise
        reader.close()

        # Should be able to open again
        reader2 = NumpyReader(npz_path)
        _ = reader2[0]
        reader2.close()


class TestNumpyReaderPreload:
    """Tests for preload_to_cpu functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preload_basic(self):
        """Test that preload_to_cpu loads data into RAM and closes the file."""
        coords = np.random.randn(15, 3).astype(np.float32)
        features = np.random.randn(15, 4).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords, features=features)

        reader = NumpyReader(
            npz_path, fields=["coords", "features"], preload_to_cpu=True
        )

        assert reader._data is None
        assert reader._preloaded is not None
        assert "coords" in reader._preloaded
        assert "features" in reader._preloaded
        assert len(reader) == 15

        data, metadata = reader[0]
        assert data["coords"].shape == (3,)
        assert data["features"].shape == (4,)
        torch.testing.assert_close(
            data["coords"], torch.from_numpy(coords[0]), atol=1e-6, rtol=1e-6
        )

    def test_preload_matches_non_preloaded(self):
        """Test that preloaded data matches non-preloaded data."""
        np.random.seed(42)
        coords = np.random.randn(10, 3).astype(np.float32)
        features = np.random.randn(10, 5).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords, features=features)

        reader_disk = NumpyReader(
            npz_path, fields=["coords", "features"], preload_to_cpu=False
        )
        reader_ram = NumpyReader(
            npz_path, fields=["coords", "features"], preload_to_cpu=True
        )

        for i in range(len(reader_disk)):
            data_disk, _ = reader_disk[i]
            data_ram, _ = reader_ram[i]
            torch.testing.assert_close(data_disk["coords"], data_ram["coords"])
            torch.testing.assert_close(data_disk["features"], data_ram["features"])

        reader_disk.close()
        reader_ram.close()

    def test_preload_with_default_values(self):
        """Test preload with default values for missing fields."""
        coords = np.random.randn(10, 100, 3).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        default_normals = torch.ones(100, 3, dtype=torch.float64)
        reader = NumpyReader(
            npz_path,
            fields=["coords", "normals"],
            default_values={"normals": default_normals},
            preload_to_cpu=True,
        )

        data, _ = reader[0]
        assert "normals" in data
        assert data["normals"].dtype == torch.float32
        reader.close()

    def test_preload_missing_required_field_raises(self):
        """Test that preload raises KeyError for missing required fields."""
        coords = np.random.randn(10, 3).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        with pytest.raises(KeyError, match="Required fields"):
            NumpyReader(
                npz_path,
                fields=["coords", "missing_field"],
                preload_to_cpu=True,
            )

    def test_preload_ignored_in_directory_mode(self):
        """Test that preload_to_cpu is ignored in directory mode."""
        for i in range(3):
            coords = np.random.randn(50, 3).astype(np.float32)
            npz_path = self.temp_path / f"sample_{i:03d}.npz"
            np.savez(npz_path, coords=coords)

        reader = NumpyReader(
            self.temp_path,
            file_pattern="sample_*.npz",
            fields=["coords"],
            preload_to_cpu=True,
        )

        assert reader._preloaded is None
        assert len(reader) == 3

        data, _ = reader[0]
        assert data["coords"].shape == (50, 3)
        reader.close()

    def test_preload_with_coordinated_subsampling(self):
        """Test preloaded reader with coordinated subsampling."""
        n_samples = 5
        n_points = 1000
        subsample_points = 100

        coords = np.random.randn(n_samples, n_points, 3).astype(np.float32)
        features = np.random.randn(n_samples, n_points, 4).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords, features=features)

        reader = NumpyReader(
            npz_path,
            fields=["coords", "features"],
            preload_to_cpu=True,
            coordinated_subsampling={
                "n_points": subsample_points,
                "target_keys": ["coords", "features"],
            },
        )

        assert reader._preloaded is not None
        data, _ = reader[0]
        assert data["coords"].shape == (subsample_points, 3)
        assert data["features"].shape == (subsample_points, 4)
        reader.close()

    def test_preload_close_releases_memory(self):
        """Test that close() releases preloaded arrays."""
        coords = np.random.randn(10, 3).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        reader = NumpyReader(npz_path, fields=["coords"], preload_to_cpu=True)
        assert reader._preloaded is not None

        reader.close()
        assert reader._preloaded is None

    def test_preload_repr(self):
        """Test that repr includes preload_to_cpu info."""
        coords = np.random.randn(10, 3).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        reader = NumpyReader(npz_path, fields=["coords"], preload_to_cpu=True)
        assert "preload_to_cpu=True" in repr(reader)
        reader.close()


class TestNumpyReaderFloat32:
    """Tests for float32 conversion behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_float64_converted_to_float32(self):
        """Test that float64 numpy arrays are returned as float32 tensors."""
        coords = np.random.randn(10, 3).astype(np.float64)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        reader = NumpyReader(npz_path, fields=["coords"])
        data, _ = reader[0]
        assert data["coords"].dtype == torch.float32
        reader.close()

    def test_float64_converted_to_float32_directory_mode(self):
        """Test float64 conversion in directory mode."""
        for i in range(3):
            coords = np.random.randn(50, 3).astype(np.float64)
            npz_path = self.temp_path / f"sample_{i:03d}.npz"
            np.savez(npz_path, coords=coords)

        reader = NumpyReader(
            self.temp_path, file_pattern="sample_*.npz", fields=["coords"]
        )
        data, _ = reader[0]
        assert data["coords"].dtype == torch.float32
        reader.close()

    def test_default_values_converted_to_float32(self):
        """Test that default values are returned as float32."""
        coords = np.random.randn(10, 100, 3).astype(np.float32)

        npz_path = self.temp_path / "data.npz"
        np.savez(npz_path, coords=coords)

        default_normals = torch.zeros(100, 3, dtype=torch.float64)
        reader = NumpyReader(
            npz_path,
            fields=["coords", "normals"],
            default_values={"normals": default_normals},
        )

        data, _ = reader[0]
        assert data["normals"].dtype == torch.float32
        reader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
