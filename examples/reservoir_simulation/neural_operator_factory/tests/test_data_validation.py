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

"""Unit tests for data validation utilities."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.validation import (
    detect_dimensions,
    get_dimension_info,
    print_validation_summary,
    validate_batch_dimensions,
    validate_sample_dimensions,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def batch_3d_input():
    """3D batched input tensor: (B, H, W, T, C)"""
    return torch.randn(4, 32, 64, 16, 12)


@pytest.fixture
def batch_3d_target():
    """3D batched target tensor: (B, H, W, T)"""
    return torch.randn(4, 32, 64, 16)


@pytest.fixture
def batch_4d_input():
    """4D batched input tensor: (B, X, Y, Z, T, C)"""
    return torch.randn(2, 16, 24, 12, 10, 8)


@pytest.fixture
def batch_4d_target():
    """4D batched target tensor: (B, X, Y, Z, T)"""
    return torch.randn(2, 16, 24, 12, 10)


@pytest.fixture
def sample_3d_input():
    """3D single sample input: (H, W, T, C)"""
    return torch.randn(32, 64, 16, 12)


@pytest.fixture
def sample_3d_target():
    """3D single sample target: (H, W, T)"""
    return torch.randn(32, 64, 16)


@pytest.fixture
def sample_4d_input():
    """4D single sample input: (X, Y, Z, T, C)"""
    return torch.randn(16, 24, 12, 10, 8)


@pytest.fixture
def sample_4d_target():
    """4D single sample target: (X, Y, Z, T)"""
    return torch.randn(16, 24, 12, 10)


# =============================================================================
# Test detect_dimensions
# =============================================================================


class TestDetectDimensions:
    """Tests for detect_dimensions function."""

    def test_detect_3d_from_batch(self, batch_3d_input):
        """Test detecting 3D from batched input."""
        assert detect_dimensions(batch_3d_input) == "3d"

    def test_detect_4d_from_batch(self, batch_4d_input):
        """Test detecting 4D from batched input."""
        assert detect_dimensions(batch_4d_input) == "4d"

    def test_detect_invalid_dimensions(self):
        """Test error for unsupported dimensions."""
        # 3D tensor (not 5D or 6D)
        invalid_tensor = torch.randn(10, 32, 64)

        with pytest.raises(ValueError) as excinfo:
            detect_dimensions(invalid_tensor)

        assert "Cannot detect dimensions" in str(excinfo.value)

    def test_detect_from_7d_tensor(self):
        """Test error for too many dimensions."""
        tensor_7d = torch.randn(2, 8, 8, 8, 8, 8, 4)

        with pytest.raises(ValueError):
            detect_dimensions(tensor_7d)


# =============================================================================
# Test validate_batch_dimensions
# =============================================================================


class TestValidateBatchDimensions:
    """Tests for validate_batch_dimensions function."""

    def test_validate_3d_batch(self, batch_3d_input, batch_3d_target):
        """Test validation of 3D batched data."""
        result = validate_batch_dimensions(batch_3d_input, batch_3d_target, "pressure")

        assert result["dimensions"] == "3d"
        assert result["batch_size"] == 4
        assert result["spatial_shape"] == (32, 64)
        assert result["time_steps"] == 16
        assert result["num_channels"] == 12

    def test_validate_4d_batch(self, batch_4d_input, batch_4d_target):
        """Test validation of 4D batched data."""
        result = validate_batch_dimensions(
            batch_4d_input, batch_4d_target, "saturation"
        )

        assert result["dimensions"] == "4d"
        assert result["batch_size"] == 2
        assert result["spatial_shape"] == (16, 24, 12)
        assert result["time_steps"] == 10
        assert result["num_channels"] == 8

    def test_batch_size_mismatch(self, batch_3d_input):
        """Test error when batch sizes don't match."""
        wrong_target = torch.randn(8, 32, 64, 16)  # Different batch size

        with pytest.raises(ValueError) as excinfo:
            validate_batch_dimensions(batch_3d_input, wrong_target)

        assert "Batch size mismatch" in str(excinfo.value)

    def test_spatial_mismatch(self, batch_3d_input):
        """Test error when spatial dimensions don't match."""
        wrong_target = torch.randn(4, 48, 64, 16)  # Different H

        with pytest.raises(ValueError) as excinfo:
            validate_batch_dimensions(batch_3d_input, wrong_target)

        assert "mismatch" in str(excinfo.value).lower()

    def test_wrong_input_ndim(self):
        """Test error for wrong input dimensions."""
        wrong_input = torch.randn(4, 32, 64)  # 3D instead of 5D/6D
        target = torch.randn(4, 32, 64)

        with pytest.raises(ValueError) as excinfo:
            validate_batch_dimensions(wrong_input, target)

        assert "Invalid input shape" in str(excinfo.value)

    def test_wrong_target_ndim_for_3d(self, batch_3d_input):
        """Test error when target has wrong dimensions for 3D data."""
        wrong_target = torch.randn(4, 32, 64, 16, 1)  # 5D instead of 4D

        with pytest.raises(ValueError) as excinfo:
            validate_batch_dimensions(batch_3d_input, wrong_target)

        assert "Invalid target shape" in str(excinfo.value)


# =============================================================================
# Test validate_sample_dimensions
# =============================================================================


class TestValidateSampleDimensions:
    """Tests for validate_sample_dimensions function."""

    def test_validate_3d_sample(self, sample_3d_input, sample_3d_target):
        """Test validation of 3D single sample."""
        result = validate_sample_dimensions(sample_3d_input, sample_3d_target)

        assert result["dimensions"] == "3d"
        assert result["spatial_shape"] == (32, 64)
        assert result["time_steps"] == 16
        assert result["num_channels"] == 12

    def test_validate_4d_sample(self, sample_4d_input, sample_4d_target):
        """Test validation of 4D single sample."""
        result = validate_sample_dimensions(sample_4d_input, sample_4d_target)

        assert result["dimensions"] == "4d"
        assert result["spatial_shape"] == (16, 24, 12)
        assert result["time_steps"] == 10
        assert result["num_channels"] == 8

    def test_sample_spatial_mismatch(self, sample_3d_input):
        """Test error when sample spatial dimensions don't match."""
        wrong_target = torch.randn(48, 64, 16)  # Different H

        with pytest.raises(ValueError) as excinfo:
            validate_sample_dimensions(sample_3d_input, wrong_target)

        assert "mismatch" in str(excinfo.value).lower()

    def test_wrong_sample_input_ndim(self):
        """Test error for wrong sample input dimensions."""
        wrong_input = torch.randn(32, 64, 16)  # 3D instead of 4D/5D
        target = torch.randn(32, 64, 16)

        with pytest.raises(ValueError) as excinfo:
            validate_sample_dimensions(wrong_input, target)

        assert "Invalid input shape" in str(excinfo.value)


# =============================================================================
# Test print_validation_summary
# =============================================================================


class TestPrintValidationSummary:
    """Tests for print_validation_summary function."""

    def test_print_3d_batch_summary(self, capsys):
        """Test printing summary for 3D batched data."""
        input_shape = (4, 32, 64, 16, 12)
        target_shape = (4, 32, 64, 16)

        print_validation_summary(input_shape, target_shape, "pressure", is_batch=True)

        captured = capsys.readouterr()
        assert "validation passed" in captured.out.lower()
        assert "3D" in captured.out
        assert "32" in captured.out  # H
        assert "64" in captured.out  # W

    def test_print_4d_batch_summary(self, capsys):
        """Test printing summary for 4D batched data."""
        input_shape = (2, 16, 24, 12, 10, 8)
        target_shape = (2, 16, 24, 12, 10)

        print_validation_summary(input_shape, target_shape, "saturation", is_batch=True)

        captured = capsys.readouterr()
        assert "4D" in captured.out
        assert "X" in captured.out or "Y" in captured.out or "Z" in captured.out

    def test_print_3d_sample_summary(self, capsys):
        """Test printing summary for 3D single sample."""
        input_shape = (32, 64, 16, 12)
        target_shape = (32, 64, 16)

        print_validation_summary(input_shape, target_shape, "pressure", is_batch=False)

        captured = capsys.readouterr()
        assert "validation passed" in captured.out.lower()

    def test_print_with_logger(self):
        """Test printing with custom logger."""

        class MockLogger:
            """Minimal logger stub for testing print_validation_summary."""

            def __init__(self):
                self.messages = []

            def success(self, msg):
                """Record a success-level message."""
                self.messages.append(("success", msg))

            def info(self, msg):
                """Record an info-level message."""
                self.messages.append(("info", msg))

        logger = MockLogger()
        input_shape = (4, 32, 64, 16, 12)
        target_shape = (4, 32, 64, 16)

        print_validation_summary(
            input_shape, target_shape, "pressure", is_batch=True, logger=logger
        )

        assert len(logger.messages) > 0
        assert any("success" in msg[0] for msg in logger.messages)


# =============================================================================
# Test get_dimension_info
# =============================================================================


class TestGetDimensionInfo:
    """Tests for get_dimension_info function."""

    def test_get_info_3d_batch_input(self, batch_3d_input):
        """Test getting info from 3D batched input."""
        info = get_dimension_info(batch_3d_input, is_batch=True)

        assert info["dimensions"] == "3d"
        assert info["batch_size"] == 4
        assert info["spatial_shape"] == (32, 64)
        assert info["time_steps"] == 16
        assert info["num_channels"] == 12

    def test_get_info_4d_batch_input(self, batch_4d_input):
        """Test getting info from 4D batched input."""
        info = get_dimension_info(batch_4d_input, is_batch=True)

        assert info["dimensions"] == "4d"
        assert info["batch_size"] == 2
        assert info["spatial_shape"] == (16, 24, 12)
        assert info["time_steps"] == 10
        assert info["num_channels"] == 8

    def test_get_info_3d_sample(self, sample_3d_input):
        """Test getting info from 3D single sample."""
        info = get_dimension_info(sample_3d_input, is_batch=False)

        assert info["dimensions"] == "3d"
        assert "batch_size" not in info
        assert info["spatial_shape"] == (32, 64)

    def test_get_info_4d_sample(self, sample_4d_input):
        """Test getting info from 4D single sample."""
        info = get_dimension_info(sample_4d_input, is_batch=False)

        assert info["dimensions"] == "4d"
        assert info["spatial_shape"] == (16, 24, 12)

    def test_get_info_invalid_tensor(self):
        """Test error for invalid tensor dimensions."""
        invalid_tensor = torch.randn(10, 32)  # 2D tensor

        with pytest.raises(ValueError):
            get_dimension_info(invalid_tensor, is_batch=True)


# =============================================================================
# Test Integration with Dataset
# =============================================================================


class TestValidationIntegration:
    """Integration tests with actual data tensors."""

    def test_full_validation_pipeline_3d(self, batch_3d_input, batch_3d_target):
        """Test full validation pipeline for 3D data."""
        # Detect dimensions
        dims = detect_dimensions(batch_3d_input)
        assert dims == "3d"

        # Validate
        info = validate_batch_dimensions(batch_3d_input, batch_3d_target)
        assert info["dimensions"] == "3d"

        # Get detailed info
        detailed = get_dimension_info(batch_3d_input, is_batch=True)
        assert detailed["dimensions"] == info["dimensions"]
        assert detailed["batch_size"] == info["batch_size"]

    def test_full_validation_pipeline_4d(self, batch_4d_input, batch_4d_target):
        """Test full validation pipeline for 4D data."""
        dims = detect_dimensions(batch_4d_input)
        assert dims == "4d"

        info = validate_batch_dimensions(batch_4d_input, batch_4d_target)
        assert info["dimensions"] == "4d"

        detailed = get_dimension_info(batch_4d_input, is_batch=True)
        assert detailed["dimensions"] == "4d"

    def test_validation_preserves_tensor_data(self, batch_3d_input, batch_3d_target):
        """Test that validation doesn't modify the tensors."""
        original_input = batch_3d_input.clone()
        original_target = batch_3d_target.clone()

        validate_batch_dimensions(batch_3d_input, batch_3d_target)

        assert torch.equal(batch_3d_input, original_input)
        assert torch.equal(batch_3d_target, original_target)
