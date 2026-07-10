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

import torch

from physicsnemo.nn.module.utils.shift_window_mask import get_shift_window_mask


def test_shift_window_mask_3d_cyclic_lon():
    """The longitude axis is cyclic: the mask must not partition it.

    Every longitude window must receive an identical mask, including the
    wrap-around window at the dateline, and masking must only separate
    regions along the non-cyclic pressure-level and latitude axes.
    """
    Pl, Lat, Lon = 8, 24, 48
    win_pl, win_lat, win_lon = 2, 6, 12
    mask = get_shift_window_mask(
        input_resolution=(Pl, Lat, Lon),
        window_size=(win_pl, win_lat, win_lon),
        shift_size=(1, 3, 6),
        ndim=3,
    )

    n_lon = Lon // win_lon
    n_pl, n_lat = Pl // win_pl, Lat // win_lat
    win_total = win_pl * win_lat * win_lon
    assert mask.shape == (n_lon, n_pl * n_lat, win_total, win_total)

    # Cyclic axis: the mask must be identical for every longitude window,
    # in particular for the wrap-around window.
    for i in range(1, n_lon):
        assert torch.equal(mask[i], mask[0])

    # Non-cyclic axes keep the standard shifted-window partition: exactly the
    # windows in the last pressure-level row or the last latitude column
    # contain a region boundary and carry masked pairs.
    masked = (mask[0] != 0).any(dim=(1, 2)).reshape(n_pl, n_lat)
    for pl in range(n_pl):
        for lat in range(n_lat):
            expected = pl == n_pl - 1 or lat == n_lat - 1
            assert masked[pl, lat].item() == expected


def test_shift_window_mask_2d_cyclic_lon():
    """Same contract for the 2D path: no partition along longitude."""
    Lat, Lon = 24, 48
    win_lat, win_lon = 6, 12
    mask = get_shift_window_mask(
        input_resolution=(Lat, Lon),
        window_size=(win_lat, win_lon),
        shift_size=(3, 6),
        ndim=2,
    )

    n_lon = Lon // win_lon
    n_lat = Lat // win_lat
    win_total = win_lat * win_lon
    assert mask.shape == (n_lon, n_lat, win_total, win_total)

    for i in range(1, n_lon):
        assert torch.equal(mask[i], mask[0])

    masked = (mask[0] != 0).any(dim=(1, 2))
    for lat in range(n_lat):
        assert masked[lat].item() == (lat == n_lat - 1)
