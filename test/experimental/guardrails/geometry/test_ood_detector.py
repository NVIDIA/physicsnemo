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

"""Tests for GeometryGuardrail OOD detector main API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from physicsnemo.experimental.guardrails import GeometryGuardrail
from physicsnemo.experimental.guardrails.geometry import FEATURE_NAMES, FEATURE_VERSION


def test_guardrail_constructor():
    """Test GuardRail constructor with various parameters."""
    guardrail = GeometryGuardrail(
        n_components=2,
        warn_pct=95.0,
        reject_pct=99.0,
        covariance_type="full",
        random_state=42,
    )
    
    assert guardrail.warn_pct == 95.0
    assert guardrail.reject_pct == 99.0
    assert guardrail.feature_names == FEATURE_NAMES
    assert guardrail.feature_version == FEATURE_VERSION


def test_guardrail_constructor_invalid_thresholds():
    """Test that invalid thresholds raise errors."""
    # warn_pct > reject_pct
    with pytest.raises(ValueError, match="warn_pct"):
        GeometryGuardrail(warn_pct=99.0, reject_pct=95.0)
    
    # Out of range
    with pytest.raises(ValueError, match="warn_pct must be in"):
        GeometryGuardrail(warn_pct=150.0)
    
    with pytest.raises(ValueError, match="reject_pct must be in"):
        GeometryGuardrail(reject_pct=-10.0)


def test_guardrail_fit():
    """Test fitting guardrail on mesh objects."""
    # Create training meshes
    train_meshes = [
        trimesh.creation.box(extents=[1, 1, 1]),
        trimesh.creation.box(extents=[1.5, 1.5, 1.5]),
        trimesh.creation.box(extents=[0.8, 0.8, 0.8]),
    ]
    
    guardrail = GeometryGuardrail(n_components=1, random_state=42)
    guardrail.fit(train_meshes)
    
    # Check that density model is fitted
    assert guardrail.density.ref_scores is not None


def test_guardrail_query():
    """Test querying guardrail with new meshes."""
    # Create and fit guardrail
    train_meshes = [trimesh.creation.box(extents=[1, 1, 1]) for _ in range(10)]
    
    guardrail = GeometryGuardrail(
        n_components=1,
        warn_pct=80.0,
        reject_pct=95.0,
        random_state=42,
    )
    guardrail.fit(train_meshes)
    
    # Query similar and dissimilar meshes
    test_meshes = [
        trimesh.creation.box(extents=[1, 1, 1]),  # Similar
        trimesh.creation.sphere(radius=100.0),  # Very different
    ]
    
    results = guardrail.query(test_meshes)
    
    assert len(results) == 2
    assert all("percentile" in r for r in results)
    assert all("status" in r for r in results)
    assert all(r["status"] in ["OK", "WARN", "REJECT"] for r in results)


def test_guardrail_classification():
    """Test that classification logic works correctly."""
    guardrail = GeometryGuardrail(warn_pct=90.0, reject_pct=95.0)
    
    assert guardrail._classify(50.0) == "OK"
    assert guardrail._classify(89.9) == "OK"
    assert guardrail._classify(90.0) == "WARN"
    assert guardrail._classify(94.9) == "WARN"
    assert guardrail._classify(95.0) == "REJECT"
    assert guardrail._classify(99.9) == "REJECT"


def test_guardrail_save_load():
    """Test saving and loading guardrail."""
    # Create and fit guardrail
    train_meshes = [trimesh.creation.box() for _ in range(10)]
    
    guardrail = GeometryGuardrail(
        n_components=1,
        warn_pct=95.0,
        reject_pct=99.0,
        random_state=42,
    )
    guardrail.fit(train_meshes)
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "guardrail.npz"
        guardrail.save(save_path)
        
        # Load and verify
        loaded = GeometryGuardrail.load(save_path)
        
        assert loaded.warn_pct == guardrail.warn_pct
        assert loaded.reject_pct == guardrail.reject_pct
        assert loaded.feature_names == guardrail.feature_names
        assert loaded.feature_version == guardrail.feature_version
        
        # Test that loaded model gives same results
        test_mesh = [trimesh.creation.box()]
        results_orig = guardrail.query(test_mesh)
        results_loaded = loaded.query(test_mesh)
        
        assert np.isclose(
            results_orig[0]["percentile"],
            results_loaded[0]["percentile"],
        )
        assert results_orig[0]["status"] == results_loaded[0]["status"]


def test_guardrail_save_before_fit():
    """Test that saving before fit raises error."""
    guardrail = GeometryGuardrail()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "guardrail.npz"
        with pytest.raises(RuntimeError, match="Guardrail not fitted"):
            guardrail.save(save_path)


def test_guardrail_fit_from_dir():
    """Test fitting from STL directory."""
    # Create temporary directory with STL files
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_dir = Path(tmpdir)
        
        # Save some test meshes
        for i in range(5):
            mesh = trimesh.creation.box(extents=[1 + i * 0.1] * 3)
            mesh.export(stl_dir / f"mesh_{i:03d}.stl")
        
        # Fit guardrail
        guardrail = GeometryGuardrail(random_state=42)
        guardrail.fit_from_dir(stl_dir, n_workers=2)
        
        assert guardrail.density.ref_scores is not None
        assert len(guardrail.density.ref_scores) == 5


def test_guardrail_query_from_dir():
    """Test querying from STL directory."""
    # Fit guardrail on some meshes
    train_meshes = [trimesh.creation.box() for _ in range(10)]
    guardrail = GeometryGuardrail(random_state=42)
    guardrail.fit(train_meshes)
    
    # Create temporary directory with test STL files
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_dir = Path(tmpdir)
        
        # Save test meshes
        for i in range(3):
            mesh = trimesh.creation.box(extents=[1 + i * 0.5] * 3)
            mesh.export(stl_dir / f"test_{i:03d}.stl")
        
        # Query directory
        results = guardrail.query_from_dir(stl_dir, n_workers=2)
        
        assert len(results) == 3
        assert all("name" in r for r in results)
        assert all("percentile" in r for r in results)
        assert all("status" in r for r in results)
        assert all(r["name"].endswith(".stl") for r in results)


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("cov_type", ["full", "diag"])
def test_guardrail_various_configs(n_components, cov_type):
    """Test guardrail with various configurations."""
    train_meshes = [trimesh.creation.box() for _ in range(10)]
    
    guardrail = GeometryGuardrail(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=42,
    )
    guardrail.fit(train_meshes)
    
    test_meshes = [trimesh.creation.sphere()]
    results = guardrail.query(test_meshes)
    
    assert len(results) == 1
    assert "status" in results[0]


def test_guardrail_outlier_detection():
    """Test that guardrail correctly identifies outliers."""
    # Train on unit cubes
    train_meshes = [
        trimesh.creation.box(extents=[1 + 0.1 * i] * 3) for i in range(20)
    ]
    
    guardrail = GeometryGuardrail(
        n_components=1,
        warn_pct=95.0,
        reject_pct=99.0,
        random_state=42,
    )
    guardrail.fit(train_meshes)
    
    # Test with inlier and outlier
    inlier = trimesh.creation.box(extents=[1.5, 1.5, 1.5])
    outlier = trimesh.creation.sphere(radius=100.0)  # Very different
    
    results = guardrail.query([inlier, outlier])
    
    # Outlier should have higher percentile than inlier
    assert results[1]["percentile"] > results[0]["percentile"]
    
    # Outlier should be flagged (at least WARN)
    assert results[1]["status"] in ["WARN", "REJECT"]
