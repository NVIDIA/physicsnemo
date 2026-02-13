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

"""Tests for geometry guardrail density model."""

import numpy as np
import pytest
import torch

pytest.importorskip("pyvista")

from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel


@pytest.mark.parametrize(
    "method,components,expected_attr",
    [
        ("gmm", {"gmm_components": 2}, "gmm_components"),
        ("pce", {"pce_components": 5}, "pce_components"),
    ],
)
def test_density_model_constructor(method, components, expected_attr):
    """Test GeometryDensityModel constructor with both methods."""
    model = GeometryDensityModel(method=method, random_state=42, **components)

    assert getattr(model, expected_attr) == components[expected_attr]
    assert model.ref_scores is None


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
def test_density_model_fit(method, components, device):
    """Test fitting the density model with both methods on CPU and GPU."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 22)

    model = GeometryDensityModel(
        method=method, device=device, random_state=42, **components
    )
    model.fit(X)

    # Check that model is fitted
    assert model.model is not None
    assert model.ref_scores is not None
    assert model.ref_scores.shape == (100,)
    ref_scores_np = (
        model.ref_scores.cpu().numpy()
        if hasattr(model.ref_scores, "cpu")
        else model.ref_scores
    )
    assert np.isfinite(ref_scores_np).all()


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
def test_density_model_score(method, components, device):
    """Test anomaly scoring with both methods on CPU and GPU."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)
    X_test = rng.randn(10, 22)

    model = GeometryDensityModel(
        method=method, device=device, random_state=42, **components
    )
    model.fit(X_train)

    scores = model.score(X_test)

    assert scores.shape == (10,)
    # Convert to numpy for assertions
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else scores
    assert np.isfinite(scores_np).all()
    assert (scores_np >= 0).all()  # Scores should be non-negative


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
def test_density_model_percentiles(method, components, device):
    """Test percentile computation with both methods on CPU and GPU."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)

    model = GeometryDensityModel(
        method=method, device=device, random_state=42, **components
    )
    model.fit(X_train)

    # Score the training data itself
    scores = model.score(X_train)
    pcts = model.percentiles(scores)

    assert pcts.shape == (100,)
    assert np.all(pcts >= 0)
    assert np.all(pcts <= 100)

    # Percentiles should be uniformly distributed for training data
    # (approximately)
    mean_pct = np.mean(pcts)
    assert 40 < mean_pct < 60  # Should be around 50


@pytest.mark.parametrize(
    "method,components",
    [
        ("gmm", {"gmm_components": 1}),
        ("pce", {"pce_components": 5}),
    ],
)
def test_density_model_percentiles_before_fit(method, components):
    """Test that percentiles raises error before fitting for both methods."""
    model = GeometryDensityModel(method=method, **components)
    scores = np.array([1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError, match="Density model not fitted"):
        model.percentiles(scores)


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
def test_density_model_outlier_detection(method, components, device):
    """Test that outliers get high percentiles with both methods on CPU and GPU."""
    rng = np.random.RandomState(42)

    # Train on standard normal
    X_train = rng.randn(100, 22)

    model = GeometryDensityModel(
        method=method, device=device, random_state=42, **components
    )
    model.fit(X_train)

    # Test on inliers and outliers
    X_inlier = rng.randn(1, 22)
    X_outlier = rng.randn(1, 22) * 10  # 10x standard deviation

    score_inlier = model.score(X_inlier)
    score_outlier = model.score(X_outlier)

    pct_inlier = model.percentiles(score_inlier)
    pct_outlier = model.percentiles(score_outlier)

    # Outlier should have higher percentile
    assert pct_outlier[0] > pct_inlier[0]
    assert pct_outlier[0] > 90  # Should be well above 90th percentile


@pytest.mark.parametrize(
    "method,components",
    [
        ("gmm", {"gmm_components": 1}),
        ("gmm", {"gmm_components": 2}),
        ("pce", {"pce_components": 5}),
        ("pce", {"pce_components": 10}),
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
def test_density_model_various_configs(method, components, device):
    """Test density model with various methods and component configurations on CPU and GPU."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 22)

    model = GeometryDensityModel(
        method=method, device=device, random_state=42, **components
    )
    model.fit(X)

    scores = model.score(X)
    pcts = model.percentiles(scores)

    assert scores.shape == (100,)
    assert pcts.shape == (100,)
    # Convert to numpy for assertions
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else scores
    assert np.isfinite(scores_np).all()
    assert np.all((pcts >= 0) & (pcts <= 100))


def test_density_model_get_set_state():
    """Test state serialization for GeometryDensityModel."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)
    X_test = rng.randn(10, 22)

    # Test GMM
    model_gmm = GeometryDensityModel(method="gmm", gmm_components=1, random_state=42)
    model_gmm.fit(X_train)
    state_gmm = model_gmm.get_state()

    new_model_gmm = GeometryDensityModel(
        method="gmm", gmm_components=1, random_state=42
    )
    new_model_gmm.set_state(state_gmm, device="cpu")

    scores1 = model_gmm.score(X_test)
    scores2 = new_model_gmm.score(X_test)
    np.testing.assert_allclose(scores1.cpu().numpy(), scores2.cpu().numpy(), rtol=1e-5)

    # Test PCE
    model_pce = GeometryDensityModel(
        method="pce", pce_components=5, poly_degree=2, random_state=42
    )
    model_pce.fit(X_train)
    state_pce = model_pce.get_state()

    new_model_pce = GeometryDensityModel(
        method="pce", pce_components=5, poly_degree=2, random_state=42
    )
    new_model_pce.set_state(state_pce, device="cpu")

    scores1 = model_pce.score(X_test)
    scores2 = new_model_pce.score(X_test)
    np.testing.assert_allclose(scores1.cpu().numpy(), scores2.cpu().numpy(), rtol=1e-5)
