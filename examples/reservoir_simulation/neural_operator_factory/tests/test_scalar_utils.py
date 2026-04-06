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

"""Unit tests for MIONet scalar channel detection utilities."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scalar_utils import (
    create_mionet_collate_fn,
    detect_scalar_channels,
    verify_scalar_consistency,
)


def _make_sample(H=8, W=16, T=4, n_spatial=9, n_scalar=3):
    """Create a sample with known scalar and spatial channels.

    Returns (sample, scalar_indices, spatial_indices) where:
    - Channels 0..n_spatial-1 vary spatially
    - Channels n_spatial..n_spatial+n_scalar-1 are constant
    """
    C = n_spatial + n_scalar
    sample = torch.randn(H, W, T, C)
    scalar_indices = []
    for i in range(n_spatial, C):
        val = float(i) * 10.0
        sample[..., i] = val
        scalar_indices.append(i)
    spatial_indices = list(range(n_spatial))
    return sample, scalar_indices, spatial_indices


class TestDetectScalarChannels:
    """Tests for detect_scalar_channels."""

    def test_identifies_scalar_channels(self):
        """Verify scalar and spatial channel indices are correctly identified."""
        sample, expected_scalar, expected_spatial = _make_sample()
        result = detect_scalar_channels(sample)
        assert set(result["scalar_indices"]) == set(expected_scalar)
        assert set(result["spatial_indices"]) == set(expected_spatial)

    def test_num_channels(self):
        """Verify reported scalar and spatial channel counts match construction."""
        sample, _, _ = _make_sample(n_spatial=9, n_scalar=3)
        result = detect_scalar_channels(sample)
        assert result["num_scalar_channels"] == 3
        assert result["num_spatial_channels"] == 9

    def test_all_spatial(self):
        """Verify detection returns zero scalar channels for a fully varying tensor."""
        sample = torch.randn(8, 16, 4, 5)
        result = detect_scalar_channels(sample)
        assert result["num_scalar_channels"] == 0
        assert result["num_spatial_channels"] == 5

    def test_all_scalar(self):
        """Verify detection identifies all channels as scalar when all are constant."""
        sample = torch.ones(8, 16, 4, 3)
        for i in range(3):
            sample[..., i] = float(i)
        result = detect_scalar_channels(sample)
        assert result["num_scalar_channels"] == 3

    def test_batched_input(self):
        """Verify scalar detection works correctly on batched (5D) inputs."""
        sample = torch.randn(2, 8, 16, 4, 12)
        sample[..., 10] = 5.0
        sample[..., 11] = 7.0
        result = detect_scalar_channels(sample)
        assert 10 in result["scalar_indices"]
        assert 11 in result["scalar_indices"]

    def test_scalar_values_tensor(self):
        """Verify scalar_values tensor has the correct shape for detected scalars."""
        sample, _, _ = _make_sample(n_spatial=2, n_scalar=2)
        result = detect_scalar_channels(sample)
        assert result["scalar_values"].shape == (2,)


class TestVerifyScalarConsistency:
    """Tests for verify_scalar_consistency."""

    def test_consistent_dataset(self):
        """Verify consistency check passes when scalar channels are uniform."""

        class FakeDataset:
            """Fake dataset with uniform scalar channels for testing."""

            def __len__(self):
                return 5

            def __getitem__(self, i):
                s, _, _ = _make_sample(n_scalar=2)
                return s, torch.zeros(8, 16, 4)

        is_ok, msg = verify_scalar_consistency(FakeDataset(), [9, 10], num_samples=3)
        assert is_ok
        assert msg is None

    def test_inconsistent_dataset(self):
        """Verify consistency check fails when scalar channels vary across samples."""

        class FakeDataset:
            """Fake dataset with inconsistent scalar channels for testing."""

            def __len__(self):
                return 5

            def __getitem__(self, i):
                s = torch.randn(8, 16, 4, 12)
                if i == 0:
                    s[..., 10] = 5.0
                return s, torch.zeros(8, 16, 4)

        is_ok, msg = verify_scalar_consistency(FakeDataset(), [10], num_samples=3)
        assert not is_ok


class TestMIONetCollateFn:
    """Tests for create_mionet_collate_fn."""

    def test_separates_channels(self):
        """Verify collate function splits spatial and scalar channels correctly."""
        scalar_idx = [3, 4]
        spatial_idx = [0, 1, 2]
        collate = create_mionet_collate_fn(scalar_idx, spatial_idx)

        batch = []
        for _ in range(2):
            inp = torch.randn(8, 16, 4, 5)
            inp[..., 3] = 10.0
            inp[..., 4] = 20.0
            tgt = torch.randn(8, 16, 4)
            batch.append((inp, tgt))

        spatial, scalar, targets = collate(batch)
        assert spatial.shape == (2, 8, 16, 4, 3)
        assert scalar.shape == (2, 2)
        assert targets.shape == (2, 8, 16, 4)

    def test_scalar_values_correct(self):
        """Verify collate function extracts the correct scalar constant values."""
        collate = create_mionet_collate_fn([2], [0, 1])
        inp = torch.randn(4, 6, 3, 3)
        inp[..., 2] = 42.0
        _, scalar, _ = collate([(inp, torch.zeros(4, 6, 3))])
        assert torch.isclose(scalar[0, 0], torch.tensor(42.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
