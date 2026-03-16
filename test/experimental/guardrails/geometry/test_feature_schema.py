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
# ruff: noqa: E402

"""Tests for geometry guardrail feature schema validation."""

import numpy as np
import pytest

pytest.importorskip("pyvista")

from physicsnemo.experimental.guardrails.geometry import (
    FEATURE_NAMES,
    FeatureSchema,
)


def test_feature_schema_attributes():
    """Test that FeatureSchema has correct attributes."""
    assert hasattr(FeatureSchema, "names")
    assert hasattr(FeatureSchema, "version")
    assert hasattr(FeatureSchema, "hash")
    assert hasattr(FeatureSchema, "dim")

    assert FeatureSchema.names == FEATURE_NAMES
    assert FeatureSchema.dim == len(FEATURE_NAMES)
    assert FeatureSchema.dim == 22


def test_feature_schema_validate_array_valid():
    """Test validation of valid feature arrays."""
    # Single sample
    X = np.random.randn(1, 22)
    FeatureSchema.validate_array(X)  # Should not raise

    # Multiple samples
    X = np.random.randn(100, 22)
    FeatureSchema.validate_array(X)  # Should not raise


def test_feature_schema_validate_array_wrong_dimensions():
    """Test rejection of arrays with wrong dimensions."""
    # 1D array
    X = np.random.randn(22)
    with pytest.raises(ValueError, match="Feature array must be 2D"):
        FeatureSchema.validate_array(X)

    # 3D array
    X = np.random.randn(10, 22, 5)
    with pytest.raises(ValueError, match="Feature array must be 2D"):
        FeatureSchema.validate_array(X)


def test_feature_schema_validate_array_wrong_features():
    """Test rejection of arrays with wrong number of features."""
    # Too few features
    X = np.random.randn(10, 10)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        FeatureSchema.validate_array(X)

    # Too many features
    X = np.random.randn(10, 30)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        FeatureSchema.validate_array(X)


def test_feature_schema_hash_consistent():
    """Test that schema hash is consistent."""
    hash1 = FeatureSchema.hash
    hash2 = FeatureSchema.hash

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256


def test_feature_schema_version():
    """Test that schema version is set."""
    assert isinstance(FeatureSchema.version, str)
    assert len(FeatureSchema.version) > 0
