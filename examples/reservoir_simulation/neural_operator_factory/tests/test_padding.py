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

"""Unit tests for padding utilities (dimension-agnostic)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.padding import (
    compute_right_pad_to_multiple,
    compute_right_pad_to_multiple_per_dim,
    pad_right_nd,
    pad_spatial_right,
)


class TestComputePads:
    """Tests for ComputePads."""

    def test_compute_right_pad_to_multiple(self):
        """Verify right-pad computation aligns dimensions to a given multiple."""
        assert compute_right_pad_to_multiple((16, 24), multiple=8, min_right_pad=0) == (
            0,
            0,
        )
        assert compute_right_pad_to_multiple((17, 24), multiple=8, min_right_pad=0) == (
            7,
            0,
        )
        assert compute_right_pad_to_multiple((16, 24), multiple=8, min_right_pad=8) == (
            8,
            8,
        )
        # Ensure min_right_pad does not break alignment (d + pad must stay multiple-of-8)
        assert compute_right_pad_to_multiple((46,), multiple=8, min_right_pad=8) == (
            10,
        )

    def test_compute_right_pad_to_multiple_per_dim(self):
        """Verify per-dimension padding respects independent min-pad constraints."""
        assert compute_right_pad_to_multiple_per_dim(
            (16, 17), multiple=8, min_right_pad=(0, 0)
        ) == (0, 7)
        assert compute_right_pad_to_multiple_per_dim(
            (16, 17), multiple=8, min_right_pad=(8, 0)
        ) == (8, 7)


class TestPadRightNd:
    """Tests for PadRightNd."""

    def test_replicate_right_pad_6d(self):
        """Verify replicate padding extends a 6D tensor along the T dimension."""
        # Shape: (B, X, Y, Z, T, C)
        x = torch.zeros(1, 1, 1, 1, 2, 1)
        x[..., 0, 0] = 10.0
        x[..., 1, 0] = 20.0

        y = pad_right_nd(x, dims=(4,), right_pad=(3,), mode="replicate")
        assert y.shape == (1, 1, 1, 1, 5, 1)
        # Last value should replicate the original last along T (20)
        assert y[0, 0, 0, 0, -1, 0].item() == 20.0


class TestPadSpatialRight:
    """Tests for PadSpatialRight."""

    def test_2d_spatial_keeps_rest(self):
        """Verify 2D spatial padding grows H and W while preserving T and C."""
        x = torch.randn(2, 5, 7, 3, 4)  # (B,H,W,T,C)
        y = pad_spatial_right(x, spatial_ndim=2, right_pad=(1, 2), mode="replicate")
        assert y.shape == (2, 6, 9, 3, 4)

    def test_3d_spatial_includes_time_when_requested(self):
        """Verify 3D spatial padding pads H, W, and T dimensions."""
        x = torch.randn(2, 5, 7, 3, 4)  # (B,H,W,T,C)
        y = pad_spatial_right(x, spatial_ndim=3, right_pad=(1, 2, 3), mode="replicate")
        assert y.shape == (2, 6, 9, 6, 4)

    def test_4d_spatial_works_for_6d_inputs(self):
        """Verify 4D spatial padding works on 6D inputs with replicate mode."""
        # (B,X,Y,Z,T,C)
        x = torch.tensor([[[[[[10.0], [20.0]]]]]])  # (1,1,1,1,2,1)
        y = pad_spatial_right(
            x, spatial_ndim=4, right_pad=(1, 1, 1, 2), mode="replicate"
        )
        assert y.shape == (1, 2, 2, 2, 4, 1)
        assert y[0, -1, -1, -1, -1, 0].item() == 20.0


class TestPaddingAdditional:
    """Additional padding tests for edge cases."""

    def test_zero_padding_returns_unchanged(self):
        """Verify zero padding returns an identical tensor."""
        from utils.padding import pad_spatial_right

        x = torch.randn(2, 8, 16, 4, 3)
        out = pad_spatial_right(x, spatial_ndim=2, right_pad=(0, 0))
        assert torch.equal(x, out)

    def test_non_uniform_per_dim_padding(self):
        """Verify per-dim padding satisfies both alignment and minimum-pad constraints."""
        from utils.padding import compute_right_pad_to_multiple_per_dim

        pads = compute_right_pad_to_multiple_per_dim(
            (10, 13, 7), multiple=8, min_right_pad=[2, 4, 1]
        )
        for i, (orig, pad) in enumerate(zip([10, 13, 7], pads)):
            assert (orig + pad) % 8 == 0, f"Dim {i}: {orig}+{pad} not multiple of 8"
            assert pad >= [2, 4, 1][i], f"Dim {i}: pad {pad} < min {[2, 4, 1][i]}"

    def test_constant_mode(self):
        """Verify constant-mode padding fills new cells with the specified value."""
        from utils.padding import pad_right_nd

        x = torch.zeros(2, 4, 6)
        out = pad_right_nd(
            x, dims=[1, 2], right_pad=[2, 3], mode="constant", constant_value=99.0
        )
        assert out.shape == (2, 6, 9)
        assert out[0, 5, 0].item() == 99.0
        assert out[0, 0, 7].item() == 99.0

    def test_replicate_mode(self):
        """Verify replicate-mode padding copies the last value along padded dims."""
        from utils.padding import pad_right_nd

        x = torch.arange(4).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
        out = pad_right_nd(x, dims=[2], right_pad=[2], mode="replicate")
        assert out.shape == (1, 1, 6)
        assert out[0, 0, 4] == out[0, 0, 3]
        assert out[0, 0, 5] == out[0, 0, 3]

    def test_invalid_spatial_ndim_raises(self):
        """Verify ValueError is raised for an unsupported spatial_ndim."""
        from utils.padding import pad_spatial_right

        with pytest.raises(ValueError, match="spatial_ndim must be"):
            pad_spatial_right(torch.randn(2, 4, 4), spatial_ndim=1, right_pad=(2,))

    def test_wrong_right_pad_length_raises(self):
        """Verify ValueError is raised when right_pad length mismatches spatial_ndim."""
        from utils.padding import pad_spatial_right

        with pytest.raises(ValueError, match="right_pad must have length"):
            pad_spatial_right(
                torch.randn(2, 4, 4, 3), spatial_ndim=2, right_pad=(2, 3, 4)
            )

    def test_4d_spatial_padding(self):
        """Verify 4D spatial padding produces correct output shape on 6D input."""
        from utils.padding import pad_spatial_right

        x = torch.randn(2, 4, 6, 3, 5, 8)  # (B, X, Y, Z, T, C)
        out = pad_spatial_right(x, spatial_ndim=4, right_pad=(1, 2, 1, 3))
        assert out.shape == (2, 5, 8, 4, 8, 8)

    def test_multiple_of_8_already_aligned(self):
        """Verify zero padding when dimensions are already aligned to the multiple."""
        from utils.padding import compute_right_pad_to_multiple

        pads = compute_right_pad_to_multiple((16, 24), multiple=8, min_right_pad=0)
        assert pads == (0, 0)

    def test_multiple_with_min_pad(self):
        """Verify padding meets the minimum pad constraint while staying aligned."""
        from utils.padding import compute_right_pad_to_multiple

        pads = compute_right_pad_to_multiple((16,), multiple=8, min_right_pad=4)
        assert pads[0] >= 4
        assert (16 + pads[0]) % 8 == 0
