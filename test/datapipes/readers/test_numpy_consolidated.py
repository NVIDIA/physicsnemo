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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
