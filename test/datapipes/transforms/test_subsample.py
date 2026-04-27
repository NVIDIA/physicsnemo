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


"""Tests for subsampling transforms."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms import (
    SubsamplePoints,
    poisson_sample_indices_fixed,
)


@pytest.mark.parametrize("replacement", [False, True])
def test_poisson_sample_indices(replacement):
    """Test Poisson sampling indices generation in both modes."""
    N = 10000
    k = 1000

    indices = poisson_sample_indices_fixed(N, k, replacement=replacement)

    assert indices.shape == (k,)
    assert indices.min() >= 0
    assert indices.max() < N
    assert indices.dtype == torch.long

    if not replacement:
        # No duplicates and strictly increasing when sampling without replacement.
        assert indices.unique().numel() == k
        assert torch.all(indices[1:] - indices[:-1] >= 1)


@pytest.mark.parametrize("replacement", [False, True])
def test_poisson_sample_large_array(replacement):
    """Test Poisson sampling with very large arrays in both modes."""
    N = 100_000_000  # 100M points
    k = 10000

    indices = poisson_sample_indices_fixed(N, k, replacement=replacement)

    assert indices.shape == (k,)
    assert indices.min() >= 0
    assert indices.max() < N

    if not replacement:
        # Sort-and-diff check: unique and strictly increasing.
        sorted_indices, _ = torch.sort(indices)
        assert torch.equal(sorted_indices, indices)
        assert torch.all(sorted_indices[1:] - sorted_indices[:-1] >= 1)


def test_poisson_sample_no_duplicates_stress():
    """Stress the duplicate-prone regime (k close to N) without replacement.

    In this regime the mean gap is close to 1, so the original replacement-style
    algorithm produced duplicates frequently. The without-replacement mode must
    never return duplicates, across many seeds.
    """
    N = 2000
    k = 1500

    for seed in range(32):
        gen = torch.Generator().manual_seed(seed)
        indices = poisson_sample_indices_fixed(N, k, generator=gen, replacement=False)
        assert indices.shape == (k,)
        assert indices.unique().numel() == k, (
            f"Duplicates found with seed={seed} in replacement=False mode"
        )
        assert indices.min() >= 0
        assert indices.max() < N


def test_poisson_sample_replacement_may_duplicate():
    """Document that replacement=True can produce duplicates.

    This is the pre-existing behavior that we keep available under the new
    ``replacement=True`` flag. Across several seeds in a duplicate-prone
    regime we expect to see at least one seed with duplicates.
    """
    N = 2000
    k = 1500

    saw_duplicate = False
    for seed in range(32):
        gen = torch.Generator().manual_seed(seed)
        indices = poisson_sample_indices_fixed(N, k, generator=gen, replacement=True)
        assert indices.shape == (k,)
        if indices.unique().numel() < k:
            saw_duplicate = True
            break

    assert saw_duplicate, (
        "Expected replacement=True to produce duplicates in a "
        "duplicate-prone regime (N=2000, k=1500) for at least one seed."
    )


@pytest.mark.parametrize("k", [1000, 1001])
def test_poisson_sample_requires_k_lt_N(k):
    """``replacement=False`` must raise when k >= N; ``replacement=True`` works."""
    N = 1000

    with pytest.raises(ValueError, match="k < N"):
        poisson_sample_indices_fixed(N, k, replacement=False)

    # With replacement, k >= N is still allowed (duplicates are expected).
    indices = poisson_sample_indices_fixed(N, k, replacement=True)
    assert indices.shape == (k,)
    assert indices.min() >= 0
    assert indices.max() < N


def test_subsample_points_basic():
    """Test basic point subsampling."""
    transform = SubsamplePoints(
        input_keys=["coords", "fields"],
        n_points=100,
        algorithm="uniform",
    )

    sample = TensorDict(
        {
            "coords": torch.randn(1000, 3),
            "fields": torch.randn(1000, 4),
        }
    )

    result = transform(sample)

    assert result["coords"].shape == (100, 3)
    assert result["fields"].shape == (100, 4)


def test_subsample_points_coordinated():
    """Test that same indices are applied to all input_keys."""
    transform = SubsamplePoints(
        input_keys=["coords", "fields"],
        n_points=100,
        algorithm="uniform",
    )

    # Create data where indices can be verified
    coords = torch.arange(1000).unsqueeze(-1).expand(-1, 3).float()
    fields = torch.arange(1000).unsqueeze(-1).expand(-1, 4).float()

    sample = TensorDict(
        {
            "coords": coords,
            "fields": fields,
        }
    )

    result = transform(sample)

    # First column of coords and fields should match
    assert torch.allclose(result["coords"][:, 0], result["fields"][:, 0])


def test_subsample_points_skip_small():
    """Test that subsampling is skipped if already small enough."""
    transform = SubsamplePoints(
        input_keys=["coords"],
        n_points=1000,
    )

    coords = torch.randn(500, 3)
    sample = TensorDict({"coords": coords})

    result = transform(sample)

    # Should return original data unchanged
    assert torch.equal(result["coords"], coords)


def test_subsample_points_weighted():
    """Test weighted sampling with weights_key parameter."""
    transform = SubsamplePoints(
        input_keys=["surface_coords", "surface_fields"],
        n_points=100,
        algorithm="uniform",
        weights_key="surface_areas",
    )

    # Create sample with
    # areas (larger areas should be sampled more)
    sample = TensorDict(
        {
            "surface_coords": torch.randn(1000, 3),
            "surface_fields": torch.randn(1000, 2),
            "surface_areas": torch.rand(1000),
        }
    )

    result = transform(sample)

    assert result["surface_coords"].shape == (100, 3)
    assert result["surface_fields"].shape == (100, 2)


def test_subsample_missing_weights_key():
    """Test that error is raised if weights key is missing."""
    transform = SubsamplePoints(
        input_keys=["surface_coords"],
        n_points=100,
        algorithm="uniform",
        weights_key="surface_areas",
    )

    sample = TensorDict(
        {
            "surface_coords": torch.randn(1000, 3),
            # Missing surface_areas
        }
    )

    with pytest.raises(KeyError, match="Weights key"):
        transform(sample)


def test_subsample_device_preservation():
    """Test that subsampling preserves device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    transform = SubsamplePoints(
        input_keys=["coords"],
        n_points=100,
    )

    sample = TensorDict(
        {
            "coords": torch.randn(1000, 3, device="cuda"),
        }
    )

    result = transform(sample)

    assert result["coords"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
