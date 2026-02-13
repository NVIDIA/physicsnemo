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

"""Tests for PCE-based density estimation."""

import numpy as np
import pytest

pytest.importorskip("pyvista")

from physicsnemo.experimental.guardrails.geometry import TorchPCEDensityModel


def test_pce_auto_components():
    """Test automatic component selection (95% variance)."""
    rng = np.random.RandomState(42)
    # Use fewer features to avoid polynomial explosion
    X_train = rng.randn(100, 10)

    model = TorchPCEDensityModel(n_components=None, poly_degree=2)
    model.fit(X_train)

    # Should have selected fewer than 10 components
    assert model.n_pca_components_ <= 10


def test_pce_interaction_only():
    """Test polynomial expansion with interaction_only."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5)  # Smaller dimension for testing

    model = TorchPCEDensityModel(n_components=3, poly_degree=2, interaction_only=True)
    model.fit(X_train)

    # Should have fewer polynomial features with interaction_only
    model_full = TorchPCEDensityModel(
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


def test_pce_insufficient_samples():
    """Test error handling for insufficient samples."""
    model = TorchPCEDensityModel()

    with pytest.raises(ValueError, match="Need at least 10 samples"):
        model.fit(np.random.randn(5, 22))


def test_pce_invalid_shape():
    """Test error handling for invalid input shape."""
    model = TorchPCEDensityModel()

    with pytest.raises(ValueError, match="must be 2D array"):
        model.fit(np.random.randn(100))  # 1D array


@pytest.mark.parametrize("poly_degree", [1, 2, 3])
def test_polynomial_degrees(poly_degree):
    """Test PCE with different polynomial degrees."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 10)

    model = TorchPCEDensityModel(n_components=5, poly_degree=poly_degree)
    model.fit(X_train)

    X_test = rng.randn(10, 10)
    scores = model.score(X_test)

    assert scores.shape == (10,)
    # Convert to numpy for assertion
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else scores
    assert np.all(np.isfinite(scores_np))
