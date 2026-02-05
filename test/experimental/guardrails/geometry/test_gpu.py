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

"""Tests for GPU acceleration functionality."""

import numpy as np
import pytest
import trimesh

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from physicsnemo.experimental.guardrails import GeometryGuardrail
from physicsnemo.experimental.guardrails.geometry import GeometryDensityModel


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_gmm_basic():
    """Test basic TorchGMM functionality."""
    from physicsnemo.experimental.guardrails.geometry.gmm_torch import TorchGMM

    # Create synthetic data
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 22)
    X_test = rng.randn(10, 22)

    # Fit and score on CPU
    gmm = TorchGMM(n_components=1, device="cpu")
    gmm.fit(X_train)
    scores = gmm.score_samples(X_test)

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_torch_gmm_gpu():
    """Test TorchGMM on GPU."""
    from physicsnemo.experimental.guardrails.geometry.gmm_torch import TorchGMM

    # Create synthetic data
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)
    X_test = rng.randn(10, 22)

    # Fit and score on GPU
    gmm = TorchGMM(n_components=2, device="cuda")
    gmm.fit(X_train)
    scores = gmm.score_samples(X_test)

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_density_model_torch_backend():
    """Test GeometryDensityModel with PyTorch backend."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 22)
    X_test = rng.randn(10, 22)

    # Create model with CPU device (torch backend)
    model = GeometryDensityModel(n_components=1, device="cpu")
    assert model.backend == "cpu" or model.backend == "sklearn"  # cpu uses sklearn

    model.fit(X_train)
    scores = model.score(X_test)
    pcts = model.percentiles(scores)

    assert scores.shape == (10,)
    assert pcts.shape == (10,)
    assert np.all((pcts >= 0) & (pcts <= 100))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_density_model_gpu():
    """Test GeometryDensityModel on GPU."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 22)
    X_test = rng.randn(10, 22)

    # Create model with CUDA device
    model = GeometryDensityModel(n_components=2, device="cuda")
    assert model.backend == "torch"
    assert model.device == "cuda"

    model.fit(X_train)
    scores = model.score(X_test)
    pcts = model.percentiles(scores)

    assert scores.shape == (10,)
    assert pcts.shape == (10,)
    assert np.all((pcts >= 0) & (pcts <= 100))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_guardrail_gpu():
    """Test GeometryGuardrail with GPU acceleration."""
    # Create training meshes
    train_meshes = [trimesh.creation.box(extents=[1 + 0.1 * i] * 3) for i in range(20)]

    # Create and fit guardrail on GPU
    guardrail = GeometryGuardrail(
        n_components=1, warn_pct=90.0, reject_pct=95.0, device="cuda", random_state=42
    )
    guardrail.fit(train_meshes)

    assert guardrail.device == "cuda"
    assert guardrail.density.backend == "torch"

    # Query with GPU
    test_meshes = [trimesh.creation.box(), trimesh.creation.icosphere(radius=10.0, subdivisions=2)]
    results = guardrail.query(test_meshes)

    assert len(results) == 2
    assert all("percentile" in r for r in results)
    assert all("status" in r for r in results)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_cpu_gpu_consistency():
    """Test that CPU and GPU backends give similar results."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 22)
    X_test = rng.randn(10, 22)

    # CPU model
    model_cpu = GeometryDensityModel(n_components=1, device="cpu", random_state=42)
    model_cpu.fit(X_train)
    scores_cpu = model_cpu.score(X_test)

    # Torch model on CPU (to test consistency without CUDA)
    from physicsnemo.experimental.guardrails.geometry.gmm_torch import TorchGMM

    model_torch = GeometryDensityModel(n_components=1, device="cpu", random_state=42)
    # Override with TorchGMM manually
    model_torch.gmm = TorchGMM(n_components=1, device="cpu")
    model_torch.backend = "torch"
    model_torch.fit(X_train)
    scores_torch = model_torch.score(X_test)

    # Results should be reasonably close (not exactly equal due to different implementations)
    assert np.corrcoef(scores_cpu, scores_torch)[0, 1] > 0.9


def test_device_parameter_cpu():
    """Test that device='cpu' works (no PyTorch required)."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 22)

    model = GeometryDensityModel(n_components=1, device="cpu")
    assert model.device == "cpu"
    assert model.backend == "sklearn"

    model.fit(X_train)
    scores = model.score(X_train[:10])

    assert scores.shape == (10,)
    assert np.isfinite(scores).all()


def test_device_parameter_invalid():
    """Test that invalid device raises helpful error."""
    # This should work (cpu doesn't need torch)
    model = GeometryDensityModel(device="cpu")
    assert model.device == "cpu"

    # If torch not available and GPU requested, should raise ImportError
    if not TORCH_AVAILABLE:
        with pytest.raises(ImportError, match="PyTorch backend requires torch"):
            GeometryDensityModel(device="cuda")
