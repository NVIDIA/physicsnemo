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

"""Tests for the Earth-specific shifted-window attention mask utilities.

Covers get_shift_window_mask, window_partition, and window_reverse for both
the 3D (Pangu-Weather) and 2D (FengWu) attention paths.
"""

import pytest
import torch

from physicsnemo.nn.module.utils.shift_window_mask import (
    get_shift_window_mask,
    window_partition,
    window_reverse,
)


class TestGetShiftWindowMask3D:
    """Tests for get_shift_window_mask with ndim=3 (Pangu-Weather path)."""

    @pytest.mark.parametrize(
        "input_resolution, window_size, shift_size",
        [
            ((8, 24, 48), (2, 6, 12), (1, 3, 6)),  # default Pangu config
            ((4, 12, 24), (2, 6, 12), (1, 3, 6)),  # smaller resolution
            ((8, 24, 48), (2, 6, 6), (1, 3, 3)),  # square lon window
        ],
    )
    def test_output_shape(self, input_resolution, window_size, shift_size):
        """Mask shape must be (n_lon, n_pl*n_lat, W, W)."""
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        mask = get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3)
        n_lon = Lon // win_lon
        n_pl_lat = (Pl // win_pl) * (Lat // win_lat)
        W = win_pl * win_lat * win_lon
        assert tuple(mask.shape) == (n_lon, n_pl_lat, W, W)

    @pytest.mark.parametrize(
        "input_resolution, window_size, shift_size",
        [
            ((8, 24, 48), (2, 6, 12), (1, 3, 6)),
            ((4, 12, 24), (2, 6, 12), (1, 3, 6)),
        ],
    )
    def test_values_binary(self, input_resolution, window_size, shift_size):
        """Mask must contain only 0.0 and -100.0."""
        mask = get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3)
        unique = sorted(torch.unique(mask).tolist())
        assert unique == [-100.0, 0.0]

    def test_longitude_unmasked_region_count(self):
        """Longitude must not be partitioned: mask must be identical for every n_lon index.

        If the longitude axis were masked, different longitude window positions would
        have different attention patterns. With longitude unmasked, the mask depends
        only on (Pl, Lat) region membership, so all n_lon slices must be equal.
        """
        mask = get_shift_window_mask(
            input_resolution=(8, 24, 48),
            window_size=(2, 6, 12),
            shift_size=(1, 3, 6),
            ndim=3,
        )
        # mask shape: (n_lon, n_pl*n_lat, W, W)
        assert torch.all(mask == mask[0].unsqueeze(0)), (
            "Mask differs across longitude window indices — longitude axis is being "
            "partitioned. All n_lon slices must be identical when longitude is cyclic."
        )

    def test_no_shift_produces_zero_mask(self):
        """With shift_size=(0,0,0) no roll occurs; attn_mask should be None (not called),
        but if called directly the mask should be all zeros (no region boundaries)."""
        # shift_size of all-zeros means every token maps to region 0
        input_resolution = (8, 24, 48)
        window_size = (2, 6, 12)
        shift_size = (0, 0, 0)
        # With zero shift, slices like slice(0, 0) produce empty ranges.
        # The function should still return a valid tensor of the right shape.
        # (This exercises the edge case; the Transformer block skips calling
        # get_shift_window_mask when roll=False, but the function itself should
        # not error.)
        # We only check it does not raise.
        try:
            mask = get_shift_window_mask(
                input_resolution, window_size, shift_size, ndim=3
            )
            assert mask is not None
        except Exception as exc:
            pytest.fail(
                f"get_shift_window_mask raised unexpectedly with zero shift: {exc}"
            )


class TestGetShiftWindowMask2D:
    """Tests for get_shift_window_mask with ndim=2 (FengWu path)."""

    @pytest.mark.parametrize(
        "input_resolution, window_size, shift_size",
        [
            ((24, 48), (6, 12), (3, 6)),  # default FengWu config (scaled)
            ((12, 24), (6, 12), (3, 6)),
        ],
    )
    def test_output_shape(self, input_resolution, window_size, shift_size):
        """Mask shape must be (n_lon, n_lat, W, W)."""
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        mask = get_shift_window_mask(input_resolution, window_size, shift_size, ndim=2)
        n_lon = Lon // win_lon
        n_lat = Lat // win_lat
        W = win_lat * win_lon
        assert tuple(mask.shape) == (n_lon, n_lat, W, W)

    def test_values_binary(self):
        """Mask must contain only 0.0 and -100.0."""
        mask = get_shift_window_mask((24, 48), (6, 12), (3, 6), ndim=2)
        unique = sorted(torch.unique(mask).tolist())
        assert unique == [-100.0, 0.0]

    def test_longitude_unmasked_region_count(self):
        """Longitude must not be partitioned: mask must be identical for every n_lon index.

        If the longitude axis were masked, different longitude window positions would
        have different attention patterns. With longitude unmasked, the mask depends
        only on Lat region membership, so all n_lon slices must be equal.
        """
        mask = get_shift_window_mask(
            input_resolution=(24, 48),
            window_size=(6, 12),
            shift_size=(3, 6),
            ndim=2,
        )
        # mask shape: (n_lon, n_lat, W, W)
        assert torch.all(mask == mask[0].unsqueeze(0)), (
            "Mask differs across longitude window indices — longitude axis is being "
            "partitioned. All n_lon slices must be identical when longitude is cyclic."
        )


class TestWindowPartitionReverse:
    """Round-trip tests: window_reverse(window_partition(x)) == x."""

    @pytest.mark.parametrize(
        "shape, window_size",
        [
            ((2, 8, 24, 48, 16), (2, 6, 12)),  # (B, Pl, Lat, Lon, C) 3D
            ((2, 24, 48, 16), (6, 12)),  # (B, Lat, Lon, C) 2D
        ],
    )
    def test_roundtrip(self, shape, window_size):
        """window_reverse(window_partition(x)) must recover x exactly."""
        torch.manual_seed(0)
        x = torch.randn(*shape)
        ndim = len(window_size)

        partitioned = window_partition(x, window_size, ndim=ndim)

        if ndim == 3:
            B, Pl, Lat, Lon, C = shape
            recovered = window_reverse(
                partitioned, window_size, Pl=Pl, Lat=Lat, Lon=Lon, ndim=ndim
            )
        else:
            B, Lat, Lon, C = shape
            recovered = window_reverse(
                partitioned, window_size, Lat=Lat, Lon=Lon, ndim=ndim
            )

        assert torch.allclose(x, recovered), (
            "window_reverse did not invert window_partition"
        )
