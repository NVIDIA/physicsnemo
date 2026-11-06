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

from physicsnemo.experimental.uq.ood import DensityScorer


@pytest.mark.parametrize(
    "method,components,expected_attr",
    [
        ("gmm", {"gmm_components": 2}, "gmm_components"),
        ("pce", {"pce_components": 5}, "pce_components"),
    ],
)
def test_density_model_constructor(method, components, expected_attr):
    """Test DensityScorer constructor with both methods."""
    model = DensityScorer(method=method, random_state=42, **components)

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

    model = DensityScorer(method=method, device=device, random_state=42, **components)
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

    model = DensityScorer(method=method, device=device, random_state=42, **components)
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

    model = DensityScorer(method=method, device=device, random_state=42, **components)
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
    model = DensityScorer(method=method, **components)
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

    model = DensityScorer(method=method, device=device, random_state=42, **components)
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

    model = DensityScorer(method=method, device=device, random_state=42, **components)
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
    """Test state serialization for DensityScorer."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)
    X_test = rng.randn(10, 22)

    # Test GMM
    model_gmm = DensityScorer(method="gmm", gmm_components=1, random_state=42)
    model_gmm.fit(X_train)
    state_gmm = model_gmm.get_state()

    new_model_gmm = DensityScorer(method="gmm", gmm_components=1, random_state=42)
    new_model_gmm.set_state(state_gmm, device="cpu")

    scores1 = model_gmm.score(X_test)
    scores2 = new_model_gmm.score(X_test)
    np.testing.assert_allclose(scores1.cpu().numpy(), scores2.cpu().numpy(), rtol=1e-5)

    # Test PCE
    model_pce = DensityScorer(
        method="pce", pce_components=5, poly_degree=2, random_state=42
    )
    model_pce.fit(X_train)
    state_pce = model_pce.get_state()

    new_model_pce = DensityScorer(
        method="pce", pce_components=5, poly_degree=2, random_state=42
    )
    new_model_pce.set_state(state_pce, device="cpu")

    scores1 = model_pce.score(X_test)
    scores2 = new_model_pce.score(X_test)
    np.testing.assert_allclose(scores1.cpu().numpy(), scores2.cpu().numpy(), rtol=1e-5)


@pytest.mark.parametrize("method", ["gmm", "pce"])
def test_density_roundtrip_decisions_and_detachment(method, device):
    import copy

    import torch

    torch.manual_seed(22)
    reference = torch.randn(80, 4, device=device, requires_grad=True)
    query = torch.randn(12, 4, device=device)
    scorer = DensityScorer(method=method, pce_components=3, device=device)
    scorer.fit(reference)
    scores = scorer.score(query)
    assert not scores.requires_grad
    assert not scorer.ref_scores.requires_grad
    saved = copy.deepcopy(scorer.get_state())
    restored = DensityScorer(device=device)
    restored.set_state(saved, device=device)
    torch.testing.assert_close(restored.score(query), scores)
    np.testing.assert_array_equal(
        restored.percentiles(scores) > 90, scorer.percentiles(scores) > 90
    )
    if method == "pce":
        torch.testing.assert_close(restored.model.pca_mean_, scorer.model.pca_mean_)
        torch.testing.assert_close(restored.model.pca_std_, scorer.model.pca_std_)
        torch.testing.assert_close(
            restored.model.pca_components_, scorer.model.pca_components_
        )


@pytest.mark.parametrize("method", ["gmm", "pce"])
def test_density_state_owns_arrays(method):
    """CPU restoration and snapshots must not alias caller-owned NumPy arrays."""
    rng = np.random.default_rng(4)
    reference, query = rng.normal(size=(50, 4)), rng.normal(size=(4, 4))
    scorer = DensityScorer(method=method, pce_components=3)
    scorer.fit(reference)
    expected = scorer.score(query)
    state = scorer.get_state()
    restored = DensityScorer()
    restored.set_state(state, device="cpu")
    for value in state.values():
        if isinstance(value, np.ndarray):
            value.fill(0)
    import torch

    torch.testing.assert_close(restored.score(query), expected)
    torch.testing.assert_close(scorer.score(query), expected)


@pytest.mark.parametrize("method", ["gmm", "pce"])
def test_density_does_not_reseed_prediction_sampling(method, device):
    """Scorer construction/fitting must not perturb an application's dropout RNG."""
    import torch

    reference = (
        torch.arange(200, dtype=torch.float32, device=device).reshape(50, 4).sin()
    )
    before = torch.get_rng_state().clone()
    cuda_before = (
        torch.cuda.get_rng_state(device).clone()
        if str(device).startswith("cuda")
        else None
    )
    scorer = DensityScorer(
        method=method, pce_components=3, device=device, random_state=42
    )
    scorer.fit(reference)
    assert torch.equal(before, torch.get_rng_state())
    if cuda_before is not None:
        assert torch.equal(cuda_before, torch.cuda.get_rng_state(device))


def test_pce_collinear_features_remain_finite(device):
    """A low-rank polynomial feature population has a stable regularized metric."""
    import torch

    values = torch.linspace(-2.0, 2.0, 20, device=device)
    reference = torch.stack(
        [values, values * 2, values.square(), values * 0, values * 0 + 1], dim=1
    )
    scorer = DensityScorer(method="pce", pce_components=5, device=device)
    scorer.fit(reference)
    query = reference[[0, 3, 7]] * 5
    scores = scorer.score(query)
    assert torch.isfinite(scorer.ref_scores).all()
    assert torch.isfinite(scores).all()
    model = scorer.model
    transformed = ((query - model.pca_mean_) / model.pca_std_) @ model.pca_components_
    features = model._generate_hermite_features(transformed).double() - model.poly_mean_
    expected = (
        (features * torch.linalg.solve(model.poly_cov_, features.T).T).sum(1).sqrt()
    )
    torch.testing.assert_close(scores.double(), expected, rtol=1e-5, atol=1e-5)
    restored = DensityScorer()
    restored.set_state(scorer.get_state(), device=device)
    assert restored.model.poly_cov_inv_.dtype == torch.float64
    torch.testing.assert_close(restored.score(query), scores)
    with pytest.raises(ValueError, match="finite"):
        scorer.percentiles(torch.tensor([float("nan")], device=device))


def test_pce_loads_legacy_float32_state():
    """Existing artifacts retain their original precision and score construction."""
    import torch

    rng = np.random.default_rng(18)
    scorer = DensityScorer(method="pce", pce_components=2)
    scorer.fit(rng.normal(size=(80, 4)))
    state = scorer.get_state()
    for key in ("pce_poly_mean_", "pce_poly_cov_", "pce_poly_cov_inv_"):
        state[key] = state[key].astype(np.float32)
    restored = DensityScorer()
    restored.set_state(state, device="cpu")
    model = restored.model
    query = torch.tensor(rng.normal(size=(5, 4)), dtype=torch.float32)
    features = model._generate_hermite_features(
        ((query - model.pca_mean_) / model.pca_std_) @ model.pca_components_
    )
    centered = features - model.poly_mean_
    expected = ((centered @ model.poly_cov_inv_) * centered).sum(1).sqrt()
    assert model.poly_cov_inv_.dtype == torch.float32
    torch.testing.assert_close(restored.score(query), expected)


@pytest.mark.parametrize("method", ["gmm", "pce"])
def test_generic_density_rejects_feature_broadcasting(method, device):
    """The generic API must enforce the width formerly checked by the geometry wrapper."""
    import torch

    reference = (
        torch.arange(120, device=device, dtype=torch.float32).reshape(30, 4).sin()
    )
    scorer = DensityScorer(method=method, pce_components=2, device=device)
    scorer.fit(reference)
    restored = DensityScorer()
    restored.set_state(scorer.get_state(), device=device)
    for candidate in (scorer, restored):
        for width in (0, 1, 3, 5):
            with pytest.raises(ValueError, match="feature dimension"):
                candidate.score(torch.zeros(2, width, device=device))
        assert candidate.score(reference[:0]).shape == (0,)
        scores = candidate.score(reference[:2])
        with pytest.raises(ValueError, match="one-dimensional"):
            candidate.percentiles(scores[:, None])
        with pytest.raises(ValueError, match="one-dimensional"):
            candidate.percentiles(scores[0])
        assert candidate.percentiles(scores).shape == (2,)


def test_failed_density_refit_clears_reference():
    """A failed backend fit cannot leave stale reference scores attached to its state."""
    import torch

    reference = torch.arange(120, dtype=torch.float32).reshape(30, 4).sin()
    scorer = DensityScorer(method="pce", pce_components=2)
    scorer.fit(reference)
    # PCE requires at least ten samples; this reaches its own fit validation.
    with pytest.raises(ValueError, match="at least 10"):
        scorer.fit(reference[:5])
    with pytest.raises(RuntimeError, match="fit"):
        scorer.score(reference[:2])
    with pytest.raises(RuntimeError, match="fit"):
        scorer.get_state()
    scorer.fit(reference)
    assert scorer.score(reference[:2]).shape == (2,)


@pytest.mark.parametrize("method", ["gmm", "pce"])
def test_restoration_passes_only_backend_state(method, device, monkeypatch):
    """Flat scorer configuration must not be mistaken for backend parameters."""
    import torch

    scorer = DensityScorer(
        method=method, gmm_components=2, pce_components=2, device=device
    )
    reference = torch.randn(40, 3, device=device)
    scorer.fit(reference)
    state = scorer.get_state()
    expected_keys = set(scorer.model.get_state())
    backend = type(scorer.model)
    restore = backend.set_state
    received = []

    def checked_restore(self, backend_state, device):
        received.append(set(backend_state))
        assert set(backend_state) == expected_keys
        restore(self, backend_state, device=device)

    monkeypatch.setattr(backend, "set_state", checked_restore)
    restored = DensityScorer(device=device)
    restored.set_state(state, device=device)
    assert len(received) == 1
    assert restored.gmm_components == 2
    assert restored.pce_components == 2
    torch.testing.assert_close(restored.score(reference), scorer.score(reference))


def test_seeded_gmm_refit_is_reproducible(device):
    """A configured seed applies to each fit without reseeding the application."""
    generator = torch.Generator().manual_seed(71)
    reference = torch.randn(80, 3, generator=generator).to(device)
    query = torch.randn(11, 3, generator=generator).to(device)
    scorer = DensityScorer(gmm_components=4, random_state=0, device=device)
    scorer.fit(reference)
    before = scorer.score(query)
    rng = torch.random.get_rng_state().clone()
    scorer.fit(reference)
    torch.testing.assert_close(scorer.score(query), before, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert scorer.model.n_features_in_ == 3


@pytest.mark.parametrize("method", ["gmm", "pce"])
@pytest.mark.parametrize(
    "damage", ["missing_backend", "missing_config", "bad_method", "bad_reference"]
)
def test_failed_density_restore_preserves_fitted_scorer(device, method, damage):
    """A failed load must leave configuration, predictions and percentiles intact."""
    reference = torch.randn(40, 3, device=device)
    query = torch.randn(5, 3, device=device)
    scorer = DensityScorer(method=method, device=device)
    scorer.fit(reference)
    before, backend = scorer.score(query), scorer.model
    percentiles = scorer.percentiles(before)
    state = scorer.get_state()
    if damage == "missing_backend":
        state["method"] = "pce" if method == "gmm" else "gmm"
    elif damage == "missing_config":
        del state["poly_degree"]
    elif damage == "bad_method":
        state["method"] = "unknown"
    else:
        state["ref_scores"][:] = np.nan
    with pytest.raises((ValueError, KeyError)):
        scorer.set_state(state, device)
    assert scorer.method == method and scorer.model is backend
    torch.testing.assert_close(scorer.score(query), before, rtol=0, atol=0)
    np.testing.assert_array_equal(scorer.percentiles(before), percentiles)


def test_legacy_gmm_state_derives_feature_width(device):
    """Checkpoints predating n_features_in_ remain readable."""
    reference = torch.randn(40, 3, device=device)
    scorer = DensityScorer(device=device)
    scorer.fit(reference)
    state = scorer.get_state()
    state.pop("gmm_n_features_in_")
    restored = DensityScorer(device=device)
    restored.set_state(state, device)
    assert restored.model.n_features_in_ == 3
    torch.testing.assert_close(restored.score(reference), scorer.score(reference))
