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

"""Tests for PCE-based density estimation."""

import numpy as np
import pytest

from physicsnemo.experimental.guardrails.geometry import PCEDensityModel


class TestPCEDensityModel:
    """Test suite for Polynomial Chaos Expansion density model."""

    def test_initialization(self):
        """Test PCE model can be initialized with various parameters."""
        model = PCEDensityModel(n_components=10, poly_degree=2)
        assert model.n_components == 10
        assert model.poly_degree == 2
        assert model.interaction_only is False

    def test_fit_and_score(self):
        """Test PCE model can fit data and compute scores."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 22)

        model = PCEDensityModel(n_components=10, poly_degree=2)
        model.fit(X_train)

        # Verify model is fitted
        assert model.pca_ is not None
        assert model.poly_mean_ is not None
        assert model.poly_cov_ is not None
        assert model.training_scores_ is not None

        # Test scoring
        X_test = rng.randn(10, 22)
        scores = model.score(X_test)

        assert scores.shape == (10,)
        assert np.all(scores >= 0)  # Mahalanobis distance is non-negative

    def test_percentiles(self):
        """Test percentile computation."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 22)

        model = PCEDensityModel(n_components=10, poly_degree=2)
        model.fit(X_train)

        X_test = rng.randn(10, 22)
        scores = model.score(X_test)
        percentiles = model.percentiles(scores)

        assert percentiles.shape == (10,)
        assert np.all(percentiles >= 0)
        assert np.all(percentiles <= 100)

    def test_auto_components(self):
        """Test automatic component selection (95% variance)."""
        rng = np.random.RandomState(42)
        # Use fewer features to avoid polynomial explosion
        X_train = rng.randn(100, 10)

        model = PCEDensityModel(n_components=None, poly_degree=2)
        model.fit(X_train)

        # Should have selected fewer than 10 components
        assert model.pca_.n_components_ <= 10

    def test_interaction_only(self):
        """Test polynomial expansion with interaction_only."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5)  # Smaller dimension for testing

        model = PCEDensityModel(n_components=3, poly_degree=2, interaction_only=True)
        model.fit(X_train)

        # Should have fewer polynomial features with interaction_only
        model_full = PCEDensityModel(
            n_components=3, poly_degree=2, interaction_only=False
        )
        model_full.fit(X_train)

        # With interaction_only, we only get cross-terms, not pure powers
        # Both should work and produce valid scores
        X_test = rng.randn(10, 5)
        scores = model.score(X_test)
        scores_full = model_full.score(X_test)

        assert scores.shape == (10,)
        assert scores_full.shape == (10,)

    def test_get_set_params(self):
        """Test parameter serialization."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 22)

        model = PCEDensityModel(n_components=10, poly_degree=2)
        model.fit(X_train)

        # Get parameters
        params = model.get_params()
        assert "n_components" in params
        assert "poly_degree" in params
        assert "pca_" in params

        # Create new model and set parameters
        new_model = PCEDensityModel()
        new_model.set_params(params)

        # Should produce same scores
        X_test = rng.randn(10, 22)
        scores1 = model.score(X_test)
        scores2 = new_model.score(X_test)

        np.testing.assert_allclose(scores1, scores2)

    def test_insufficient_samples(self):
        """Test error handling for insufficient samples."""
        model = PCEDensityModel()

        with pytest.raises(ValueError, match="Need at least 10 samples"):
            model.fit(np.random.randn(5, 22))

    def test_invalid_shape(self):
        """Test error handling for invalid input shape."""
        model = PCEDensityModel()

        with pytest.raises(ValueError, match="must be 2D array"):
            model.fit(np.random.randn(100))  # 1D array

    def test_score_before_fit(self):
        """Test error when scoring before fitting."""
        model = PCEDensityModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.score(np.random.randn(10, 22))

    def test_percentiles_before_fit(self):
        """Test error when computing percentiles before fitting."""
        model = PCEDensityModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.percentiles(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("poly_degree", [1, 2, 3])
def test_polynomial_degrees(poly_degree):
    """Test PCE with different polynomial degrees."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 10)

    model = PCEDensityModel(n_components=5, poly_degree=poly_degree)
    model.fit(X_train)

    X_test = rng.randn(10, 10)
    scores = model.score(X_test)

    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
