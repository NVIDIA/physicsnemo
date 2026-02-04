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

"""Tests for geometry guardrail feature extraction."""

import numpy as np
import pytest
import trimesh

from physicsnemo.experimental.guardrails.geometry import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_hash,
)


def test_extract_features_basic():
    """Test basic feature extraction from a simple mesh."""
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    features = extract_features(mesh)
    
    # Check output shape
    assert features.shape == (len(FEATURE_NAMES),)
    assert features.shape[0] == 22
    
    # Check all values are finite
    assert np.isfinite(features).all()


def test_extract_features_centroid():
    """Test that centroid features are correct."""
    # Create box centered at origin
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    features = extract_features(mesh)
    
    # First 3 features are centroid
    centroid = features[:3]
    # Should be near zero (box is centered)
    assert np.allclose(centroid, [0, 0, 0], atol=1e-6)


def test_extract_features_translated_mesh():
    """Test feature extraction on translated mesh."""
    # Create box at different position
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.apply_translation([10, 20, 30])
    
    features = extract_features(mesh)
    centroid = features[:3]
    
    # Centroid should reflect translation
    assert np.allclose(centroid, [10, 20, 30], atol=1e-6)


def test_extract_features_area():
    """Test that surface area feature is correct."""
    # Unit cube has surface area of 6
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    features = extract_features(mesh)
    
    # Feature index 18 is total_area
    total_area = features[18]
    assert np.isclose(total_area, 6.0, rtol=0.01)


def test_extract_features_deterministic():
    """Test that feature extraction is deterministic."""
    mesh = trimesh.creation.sphere(radius=1.0, subdivisions=3)
    
    features1 = extract_features(mesh)
    features2 = extract_features(mesh)
    
    assert np.allclose(features1, features2)


def test_extract_features_invalid_mesh():
    """Test that invalid meshes raise errors."""
    # Too few vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    with pytest.raises(ValueError, match="Too few vertices"):
        extract_features(mesh)


def test_extract_features_insufficient_pca():
    """Test error on insufficient points for PCA."""
    # Create mesh with exactly 9 vertices (less than 10 required for PCA)
    vertices = np.random.randn(9, 3) * 10
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    with pytest.raises(ValueError, match="Insufficient points for PCA"):
        extract_features(mesh)


@pytest.mark.parametrize("shape", ["box", "sphere", "cylinder"])
def test_extract_features_various_shapes(shape):
    """Test feature extraction on various primitive shapes."""
    if shape == "box":
        mesh = trimesh.creation.box(extents=[1, 2, 3])
    elif shape == "sphere":
        mesh = trimesh.creation.sphere(radius=2.0)
    elif shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=1.0, height=3.0)
    
    features = extract_features(mesh)
    
    # Check shape and finiteness
    assert features.shape == (22,)
    assert np.isfinite(features).all()
    
    # Check that features are non-trivial (not all zeros)
    assert not np.allclose(features, 0.0)


def test_feature_hash_deterministic():
    """Test that feature hash is deterministic."""
    names = ["feat1", "feat2", "feat3"]
    
    hash1 = feature_hash(names)
    hash2 = feature_hash(names)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 produces 64 hex characters


def test_feature_hash_sensitive():
    """Test that feature hash is sensitive to changes."""
    names1 = ["feat1", "feat2", "feat3"]
    names2 = ["feat1", "feat2", "feat4"]  # Changed last element
    names3 = ["feat1", "feat3", "feat2"]  # Reordered
    
    hash1 = feature_hash(names1)
    hash2 = feature_hash(names2)
    hash3 = feature_hash(names3)
    
    # All should be different
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3


def test_feature_names_constant():
    """Test that FEATURE_NAMES is correct."""
    assert len(FEATURE_NAMES) == 22
    assert "centroid_x" in FEATURE_NAMES
    assert "total_area" in FEATURE_NAMES
    assert "pca_eig1" in FEATURE_NAMES


def test_feature_version_constant():
    """Test that FEATURE_VERSION is set."""
    assert isinstance(FEATURE_VERSION, str)
    assert FEATURE_VERSION.startswith("v")
