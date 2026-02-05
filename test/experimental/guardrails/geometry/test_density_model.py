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

"""Tests for geometry guardrail density model."""

import numpy as np
import pytest

from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel


def test_density_model_constructor():
    """Test GeometryDensityModel constructor."""
    model = GeometryDensityModel(
        n_components=2,
        random_state=42,
    )

    assert model.n_components == 2
    assert model.ref_scores is None


def test_density_model_fit():
    """Test fitting the density model."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 22)

    model = GeometryDensityModel(n_components=1, random_state=42)
    model.fit(X)

    # Check that model is fitted
    assert model.model is not None
    assert model.ref_scores is not None
    assert model.ref_scores.shape == (100,)
    assert np.isfinite(model.ref_scores).all()


def test_density_model_score():
    """Test anomaly scoring."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)
    X_test = rng.randn(10, 22)

    model = GeometryDensityModel(random_state=42)
    model.fit(X_train)

    scores = model.score(X_test)

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()  # Negative log-likelihood should be non-negative


def test_density_model_percentiles():
    """Test percentile computation."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)

    model = GeometryDensityModel(random_state=42)
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


def test_density_model_percentiles_before_fit():
    """Test that percentiles raises error before fitting."""
    model = GeometryDensityModel()
    scores = np.array([1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError, match="Density model not fitted"):
        model.percentiles(scores)


def test_density_model_outlier_detection():
    """Test that outliers get high percentiles."""
    rng = np.random.RandomState(42)

    # Train on standard normal
    X_train = rng.randn(100, 22)

    model = GeometryDensityModel(random_state=42)
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


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_density_model_various_components(n_components):
    """Test density model with various numbers of components."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 22)

    model = GeometryDensityModel(n_components=n_components, random_state=42)
    model.fit(X)

    scores = model.score(X)
    pcts = model.percentiles(scores)

    assert scores.shape == (100,)
    assert pcts.shape == (100,)
    assert np.isfinite(scores).all()
    assert np.all((pcts >= 0) & (pcts <= 100))
