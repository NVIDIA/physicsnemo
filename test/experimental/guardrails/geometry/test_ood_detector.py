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

"""Tests for GeometryGuardrail OOD detector main API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyvista")

import pyvista as pv

from physicsnemo.experimental.guardrails import GeometryGuardrail
from physicsnemo.experimental.guardrails.geometry import FEATURE_NAMES, FEATURE_VERSION
from physicsnemo.mesh.io.io_pyvista import from_pyvista


def test_guardrail_constructor():
    """Test GuardRail constructor with various parameters."""
    guardrail = GeometryGuardrail(
        method="gmm",
        gmm_components=2,
        warn_pct=95.0,
        reject_pct=99.0,
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
        GeometryGuardrail(method="gmm", warn_pct=99.0, reject_pct=95.0)

    # Out of range
    with pytest.raises(ValueError, match="warn_pct must be in"):
        GeometryGuardrail(method="gmm", warn_pct=150.0)

    with pytest.raises(ValueError, match="reject_pct must be in"):
        GeometryGuardrail(method="gmm", reject_pct=-10.0)


@pytest.mark.parametrize(
    "method,components",
    [
        ("gmm", {"gmm_components": 1}),
        ("pce", {"pce_components": 5}),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
            id="cuda",
        ),
    ],
)
def test_guardrail_fit_and_query(method, components, device):
    """Test fitting and querying guardrail with various configurations on CPU and GPU."""
    # Create training meshes (PCE requires at least 10 samples)
    train_meshes = [from_pyvista(pv.Cube()) for _ in range(10)]

    guardrail = GeometryGuardrail(
        method=method,
        warn_pct=80.0,
        reject_pct=95.0,
        device=device,
        random_state=42,
        **components,
    )
    guardrail.fit(train_meshes)

    # Query similar and dissimilar meshes
    test_meshes = [
        from_pyvista(pv.Cube()),  # Similar
        from_pyvista(pv.Sphere(radius=100.0)),  # Very different
    ]

    results = guardrail.query(test_meshes)

    assert len(results) == 2
    assert all("percentile" in r for r in results)
    assert all("status" in r for r in results)
    assert all(r["status"] in ["OK", "WARN", "REJECT"] for r in results)


def test_guardrail_classification():
    """Test that classification logic works correctly."""
    guardrail = GeometryGuardrail(method="gmm", warn_pct=90.0, reject_pct=95.0)

    assert guardrail._classify(50.0) == "OK"
    assert guardrail._classify(89.9) == "OK"
    assert guardrail._classify(90.0) == "WARN"
    assert guardrail._classify(94.9) == "WARN"
    assert guardrail._classify(95.0) == "REJECT"
    assert guardrail._classify(99.9) == "REJECT"


@pytest.mark.parametrize(
    "method,components",
    [
        ("gmm", {"gmm_components": 1}),
        ("pce", {"pce_components": 5, "poly_degree": 2, "interaction_only": False}),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
            id="cuda",
        ),
    ],
)
def test_guardrail_save_load(method, components, device):
    """Test saving and loading guardrail with both methods on CPU and GPU."""
    # Create and fit guardrail
    train_meshes = [from_pyvista(pv.Cube()) for _ in range(10)]

    guardrail = GeometryGuardrail(
        method=method,
        warn_pct=95.0,
        reject_pct=99.0,
        device=device,
        random_state=42,
        **components,
    )
    guardrail.fit(train_meshes)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / f"guardrail_{method}.npz"
        guardrail.save(save_path)

        # Load and verify (use same device as original)
        loaded = GeometryGuardrail.load(save_path, device=device)

        assert loaded.method == method
        assert loaded.warn_pct == guardrail.warn_pct
        assert loaded.reject_pct == guardrail.reject_pct
        assert loaded.feature_names == guardrail.feature_names
        assert loaded.feature_version == guardrail.feature_version

        # Verify method-specific attributes
        if method == "gmm":
            assert loaded.gmm_components == components["gmm_components"]
        else:
            assert loaded.pce_components == components["pce_components"]
            assert loaded.poly_degree == components["poly_degree"]
            assert loaded.interaction_only == components["interaction_only"]

        # Test that loaded model gives same results
        test_mesh = [from_pyvista(pv.Cube())]
        results_orig = guardrail.query(test_mesh)
        results_loaded = loaded.query(test_mesh)

        assert np.isclose(
            results_orig[0]["percentile"],
            results_loaded[0]["percentile"],
        )
        assert results_orig[0]["status"] == results_loaded[0]["status"]


def test_guardrail_save_before_fit():
    """Test that saving before fit raises error."""
    guardrail = GeometryGuardrail(method="gmm")

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
            size = 1 + i * 0.1
            box = pv.Box(
                bounds=(-size / 2, size / 2, -size / 2, size / 2, -size / 2, size / 2)
            )
            box.save(str(stl_dir / f"mesh_{i:03d}.stl"))

        # Fit guardrail
        guardrail = GeometryGuardrail(method="gmm", random_state=42)
        guardrail.fit_from_dir(stl_dir, n_workers=2)

        assert guardrail.density.ref_scores is not None
        assert len(guardrail.density.ref_scores) == 5


def test_guardrail_query_from_dir():
    """Test querying from STL directory."""
    # Fit guardrail on some meshes
    train_meshes = [from_pyvista(pv.Cube()) for _ in range(10)]
    guardrail = GeometryGuardrail(method="gmm", random_state=42)
    guardrail.fit(train_meshes)

    # Create temporary directory with test STL files
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_dir = Path(tmpdir)

        # Save test meshes
        for i in range(3):
            size = 1 + i * 0.5
            box = pv.Box(
                bounds=(-size / 2, size / 2, -size / 2, size / 2, -size / 2, size / 2)
            )
            box.save(str(stl_dir / f"test_{i:03d}.stl"))

        # Query directory
        results = guardrail.query_from_dir(stl_dir, n_workers=2)

        assert len(results) == 3
        assert all("name" in r for r in results)
        assert all("percentile" in r for r in results)
        assert all("status" in r for r in results)
        assert all(r["name"].endswith(".stl") for r in results)


@pytest.mark.parametrize(
    "method,components",
    [
        ("gmm", {"gmm_components": 1}),
        ("pce", {"pce_components": 5}),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
            id="cuda",
        ),
    ],
)
def test_guardrail_outlier_detection(method, components, device):
    """Test that guardrail correctly identifies outliers with both methods on CPU and GPU."""
    # Train on diverse shapes to ensure well-conditioned covariance matrix
    train_meshes = []
    for i in range(20):
        size = 1 + 0.1 * i
        # Mix boxes and spheres for diversity
        if i % 2 == 0:
            train_meshes.append(
                from_pyvista(
                    pv.Box(
                        bounds=(
                            -size / 2,
                            size / 2,
                            -size / 2,
                            size / 2,
                            -size / 2,
                            size / 2,
                        )
                    )
                )
            )
        else:
            train_meshes.append(from_pyvista(pv.Sphere(radius=size / 2)))

    guardrail = GeometryGuardrail(
        method=method,
        warn_pct=95.0,
        reject_pct=99.0,
        device=device,
        random_state=42,
        **components,
    )
    guardrail.fit(train_meshes)

    # Test with inlier (similar to training) and outlier (very different)
    # Use a box similar to training data as inlier
    inlier = from_pyvista(pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)))
    # Use a very extreme outlier to ensure it's clearly anomalous
    outlier = from_pyvista(pv.Sphere(radius=2.0))

    results = guardrail.query([inlier, outlier])

    # Basic sanity check: outlier should have higher or equal percentile than inlier
    assert results[1]["percentile"] >= results[0]["percentile"]

    # Outlier should be flagged if its percentile is above warning threshold
    # Use tolerance for numerical precision differences between CPU/CUDA
    if (
        results[1]["percentile"] >= guardrail.warn_pct - 0.1
    ):  # Small tolerance for numerical differences
        assert results[1]["status"] in ["WARN", "REJECT"]

    # Verify results have expected structure
    assert len(results) == 2
    assert all("percentile" in r for r in results)
    assert all("status" in r for r in results)
    assert all(r["status"] in ["OK", "WARN", "REJECT"] for r in results)
    assert all(0 <= r["percentile"] <= 100 for r in results)
