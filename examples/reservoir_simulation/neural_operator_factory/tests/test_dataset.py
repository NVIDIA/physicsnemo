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

"""Unit tests for the unified ReservoirDataset and data loading utilities."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataloader import (
    ReservoirDataset,
    collate_fn,
    create_dataloaders,
    get_dataset_info,
)

# =============================================================================
# Fixtures for Creating Test Data
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_3d_data(temp_data_dir):
    """Create mock 3D dataset files (CO2 format)."""
    # 3D: (N, H, W, T, C) input, (N, H, W, T) output
    N, H, W, T, C = 10, 32, 64, 16, 12

    for mode in ["train", "val", "test"]:
        input_data = torch.randn(N, H, W, T, C)
        output_data = torch.randn(N, H, W, T)

        torch.save(input_data, temp_data_dir / f"dP_{mode}_a.pt")
        torch.save(output_data, temp_data_dir / f"dP_{mode}_u.pt")

    return temp_data_dir, {"N": N, "H": H, "W": W, "T": T, "C": C}


@pytest.fixture
def mock_4d_data(temp_data_dir):
    """Create mock 4D dataset files (Norne format)."""
    # 4D: (N, X, Y, Z, T, C) input, (N, X, Y, Z, T) output
    N, X, Y, Z, T, C = 8, 16, 24, 12, 10, 8

    for mode in ["train", "val", "test"]:
        input_data = torch.randn(N, X, Y, Z, T, C)
        output_data = torch.randn(N, X, Y, Z, T)

        torch.save(input_data, temp_data_dir / f"norne_{mode}_input.pt")
        torch.save(output_data, temp_data_dir / f"norne_{mode}_output.pt")

    return temp_data_dir, {"N": N, "X": X, "Y": Y, "Z": Z, "T": T, "C": C}


@pytest.fixture
def mock_generic_data(temp_data_dir):
    """Create mock generic dataset files."""
    N, H, W, T, C = 6, 24, 48, 12, 6

    for mode in ["train", "val", "test"]:
        input_data = torch.randn(N, H, W, T, C)
        output_data = torch.randn(N, H, W, T)

        torch.save(input_data, temp_data_dir / f"{mode}_input.pt")
        torch.save(output_data, temp_data_dir / f"{mode}_output.pt")

    return temp_data_dir, {"N": N, "H": H, "W": W, "T": T, "C": C}


# =============================================================================
# Test ReservoirDataset - 3D Data
# =============================================================================


class TestReservoirDataset3D:
    """Tests for ReservoirDataset with 3D data."""

    def test_load_3d_with_variable(self, mock_3d_data):
        """Test loading 3D data using variable name (CO2 convention)."""
        data_path, dims = mock_3d_data

        ds = ReservoirDataset(data_path, mode="train", variable="pressure")

        assert len(ds) == dims["N"]
        assert ds.dimensions == "3d"
        assert ds.spatial_shape == (dims["H"], dims["W"])
        assert ds.time_steps == dims["T"]
        assert ds.num_channels == dims["C"]

    def test_getitem_3d(self, mock_3d_data):
        """Test __getitem__ returns correct shapes for 3D data."""
        data_path, dims = mock_3d_data

        ds = ReservoirDataset(
            data_path, mode="train", variable="pressure", normalize=False
        )
        x, y = ds[0]

        # Single sample shapes (no batch dimension)
        assert x.shape == (dims["H"], dims["W"], dims["T"], dims["C"])
        assert y.shape == (dims["H"], dims["W"], dims["T"])

    def test_normalization_3d(self, mock_3d_data):
        """Test normalization for 3D data."""
        data_path, dims = mock_3d_data

        ds_norm = ReservoirDataset(
            data_path, mode="train", variable="pressure", normalize=True
        )
        ds_raw = ReservoirDataset(
            data_path, mode="train", variable="pressure", normalize=False
        )

        x_norm, y_norm = ds_norm[0]
        x_raw, y_raw = ds_raw[0]

        # Normalized data should be different from raw
        assert not torch.allclose(x_norm, x_raw)

    def test_normalization_stats_sharing(self, mock_3d_data):
        """Test that normalization stats can be shared across datasets."""
        data_path, _ = mock_3d_data

        train_ds = ReservoirDataset(
            data_path, mode="train", variable="pressure", normalize=True
        )
        val_ds = ReservoirDataset(
            data_path, mode="val", variable="pressure", normalize=True
        )

        # Share normalization from train to val
        norm_stats = train_ds.get_normalization_stats()
        val_ds.set_normalization(*norm_stats)

        # Check that val_ds now has train's normalization
        assert torch.allclose(val_ds.input_mean, train_ds.input_mean)
        assert torch.allclose(val_ds.input_std, train_ds.input_std)


# =============================================================================
# Test ReservoirDataset - 4D Data
# =============================================================================


class TestReservoirDataset4D:
    """Tests for ReservoirDataset with 4D data."""

    def test_load_4d_with_explicit_files(self, mock_4d_data):
        """Test loading 4D data using explicit file patterns."""
        data_path, dims = mock_4d_data

        ds = ReservoirDataset(
            data_path,
            mode="train",
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
        )

        assert len(ds) == dims["N"]
        assert ds.dimensions == "4d"
        assert ds.spatial_shape == (dims["X"], dims["Y"], dims["Z"])
        assert ds.time_steps == dims["T"]
        assert ds.num_channels == dims["C"]

    def test_getitem_4d(self, mock_4d_data):
        """Test __getitem__ returns correct shapes for 4D data."""
        data_path, dims = mock_4d_data

        ds = ReservoirDataset(
            data_path,
            mode="train",
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
            normalize=False,
        )
        x, y = ds[0]

        # Single sample shapes (no batch dimension)
        assert x.shape == (dims["X"], dims["Y"], dims["Z"], dims["T"], dims["C"])
        assert y.shape == (dims["X"], dims["Y"], dims["Z"], dims["T"])

    def test_normalization_4d(self, mock_4d_data):
        """Test normalization for 4D data."""
        data_path, dims = mock_4d_data

        ds_norm = ReservoirDataset(
            data_path,
            mode="train",
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
            normalize=True,
        )

        # Check normalization stats have correct shape
        # Input mean should be (1, 1, 1, 1, 1, C) for 6D data
        assert ds_norm.input_mean.shape[-1] == dims["C"]


# =============================================================================
# Test Dimension Validation
# =============================================================================


class TestDimensionValidation:
    """Tests for dimension validation against config."""

    def test_expected_dimensions_match(self, mock_3d_data):
        """Test that matching expected dimensions passes."""
        data_path, _ = mock_3d_data

        # Should not raise - dimensions match
        ds = ReservoirDataset(
            data_path, mode="train", variable="pressure", expected_dimensions="3d"
        )
        assert ds.dimensions == "3d"

    def test_expected_dimensions_mismatch(self, mock_3d_data):
        """Test that mismatched expected dimensions raises error."""
        data_path, _ = mock_3d_data

        # Should raise - expecting 4d but data is 3d
        with pytest.raises(ValueError) as excinfo:
            ReservoirDataset(
                data_path, mode="train", variable="pressure", expected_dimensions="4d"
            )

        assert "Dimension mismatch" in str(excinfo.value)
        assert "4d" in str(excinfo.value)
        assert "3d" in str(excinfo.value)

    def test_expected_dimensions_4d(self, mock_4d_data):
        """Test expected dimensions validation for 4D data."""
        data_path, _ = mock_4d_data

        # Correct expectation
        ds = ReservoirDataset(
            data_path,
            mode="train",
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
            expected_dimensions="4d",
        )
        assert ds.dimensions == "4d"

        # Wrong expectation
        with pytest.raises(ValueError):
            ReservoirDataset(
                data_path,
                mode="train",
                input_file="norne_{mode}_input.pt",
                output_file="norne_{mode}_output.pt",
                expected_dimensions="3d",
            )


# =============================================================================
# Test Auto-Detection
# =============================================================================


class TestAutoDetection:
    """Tests for file auto-detection."""

    def test_auto_detect_generic_files(self, mock_generic_data):
        """Test auto-detection of generic file naming pattern."""
        data_path, dims = mock_generic_data

        ds = ReservoirDataset(data_path, mode="train")

        assert len(ds) == dims["N"]
        assert ds.dimensions == "3d"

    def test_auto_detect_co2_files(self, mock_3d_data):
        """Test auto-detection when variable is specified."""
        data_path, dims = mock_3d_data

        ds = ReservoirDataset(data_path, mode="train", variable="pressure")
        assert len(ds) == dims["N"]

    def test_auto_detect_fails_gracefully(self, temp_data_dir):
        """Test that auto-detection gives helpful error for unknown files."""
        # Create files with unusual naming
        torch.save(torch.randn(5, 10, 10, 5, 3), temp_data_dir / "weird_input.pt")
        torch.save(torch.randn(5, 10, 10, 5), temp_data_dir / "weird_output.pt")

        with pytest.raises(FileNotFoundError) as excinfo:
            ReservoirDataset(temp_data_dir, mode="train")

        assert "weird_input.pt" in str(excinfo.value) or "weird_output.pt" in str(
            excinfo.value
        )


# =============================================================================
# Test Collate Functions
# =============================================================================


class TestCollateFunctions:
    """Tests for collate functions."""

    def test_collate_fn(self, mock_3d_data):
        """Test collate function for 3D data."""
        data_path, dims = mock_3d_data
        ds = ReservoirDataset(
            data_path, mode="train", variable="pressure", normalize=False
        )

        batch = [ds[i] for i in range(3)]
        inputs, targets = collate_fn(batch)

        assert inputs.shape == (3, dims["H"], dims["W"], dims["T"], dims["C"])
        assert targets.shape == (3, dims["H"], dims["W"], dims["T"])

    def test_collate_fn_4d(self, mock_4d_data):
        """Test collate function for 4D data."""
        data_path, dims = mock_4d_data
        ds = ReservoirDataset(
            data_path,
            mode="train",
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
            normalize=False,
        )

        batch = [ds[i] for i in range(2)]
        inputs, targets = collate_fn(batch)

        assert inputs.shape == (
            2,
            dims["X"],
            dims["Y"],
            dims["Z"],
            dims["T"],
            dims["C"],
        )
        assert targets.shape == (2, dims["X"], dims["Y"], dims["Z"], dims["T"])


# =============================================================================
# Test Dataloaders
# =============================================================================


class TestDataloaders:
    """Tests for create_dataloaders function."""

    def test_create_dataloaders_3d(self, mock_3d_data):
        """Test creating dataloaders for 3D data."""
        data_path, dims = mock_3d_data

        train, val, test = create_dataloaders(
            data_path, variable="pressure", batch_size=2, num_workers=0, normalize=False
        )

        assert len(train.dataset) == dims["N"]
        assert len(val.dataset) == dims["N"]
        assert len(test.dataset) == dims["N"]

        # Check batch dimensions
        inputs, targets = next(iter(train))
        assert inputs.shape[0] == 2  # batch size
        assert inputs.dim() == 5  # (B, H, W, T, C)

    def test_create_dataloaders_4d(self, mock_4d_data):
        """Test creating dataloaders for 4D data."""
        data_path, dims = mock_4d_data

        train, val, test = create_dataloaders(
            data_path,
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
            batch_size=2,
            num_workers=0,
            normalize=False,
        )

        inputs, targets = next(iter(train))
        assert inputs.dim() == 6  # (B, X, Y, Z, T, C)
        assert targets.dim() == 5  # (B, X, Y, Z, T)

    def test_create_dataloaders_with_dimension_validation(self, mock_3d_data):
        """Test dataloaders with expected_dimensions validation."""
        data_path, _ = mock_3d_data

        # Should work
        train, _, _ = create_dataloaders(
            data_path,
            variable="pressure",
            batch_size=2,
            num_workers=0,
            expected_dimensions="3d",
        )
        assert len(train.dataset) > 0

        # Should fail
        with pytest.raises(ValueError):
            create_dataloaders(
                data_path,
                variable="pressure",
                batch_size=2,
                num_workers=0,
                expected_dimensions="4d",
            )


# =============================================================================
# Test get_dataset_info Utility
# =============================================================================


class TestDatasetInfo:
    """Tests for get_dataset_info utility function."""

    def test_get_dataset_info_3d(self, mock_3d_data):
        """Test getting dataset info for 3D data."""
        data_path, dims = mock_3d_data

        info = get_dataset_info(data_path, variable="pressure")

        assert info["dimensions"] == "3d"
        assert info["spatial_shape"] == (dims["H"], dims["W"])
        assert info["time_steps"] == dims["T"]
        assert info["num_channels"] == dims["C"]
        assert info["num_samples"]["train"] == dims["N"]

    def test_get_dataset_info_4d(self, mock_4d_data):
        """Test getting dataset info for 4D data."""
        data_path, dims = mock_4d_data

        info = get_dataset_info(
            data_path,
            input_file="norne_{mode}_input.pt",
            output_file="norne_{mode}_output.pt",
        )

        assert info["dimensions"] == "4d"
        assert info["spatial_shape"] == (dims["X"], dims["Y"], dims["Z"])
        assert info["time_steps"] == dims["T"]
        assert info["num_channels"] == dims["C"]


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_mode(self, mock_3d_data):
        """Test that invalid mode raises error."""
        data_path, _ = mock_3d_data

        with pytest.raises(ValueError) as excinfo:
            ReservoirDataset(data_path, mode="invalid", variable="pressure")

        assert (
            "train" in str(excinfo.value).lower() or "val" in str(excinfo.value).lower()
        )

    def test_missing_files(self, temp_data_dir):
        """Test error when data files don't exist."""
        with pytest.raises(FileNotFoundError):
            ReservoirDataset(
                temp_data_dir,
                mode="train",
                input_file="nonexistent_input.pt",
                output_file="nonexistent_output.pt",
            )

    def test_invalid_data_dimensions(self, temp_data_dir):
        """Test error for unsupported data dimensions."""
        # Create 3D input (wrong)
        torch.save(torch.randn(10, 32, 64, 12), temp_data_dir / "train_input.pt")
        torch.save(torch.randn(10, 32, 64), temp_data_dir / "train_output.pt")

        with pytest.raises(ValueError) as excinfo:
            ReservoirDataset(temp_data_dir, mode="train")

        assert "Unsupported data dimensions" in str(excinfo.value)
