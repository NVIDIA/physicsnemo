# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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


def window_partition(x: torch.Tensor, window_size, ndim=3):
    """
    Args:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        ndim (int): dimension of window (3 or 2)

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
    """
    if ndim == 3:
        B, Pl, Lat, Lon, C = x.shape
        win_pl, win_lat, win_lon = window_size
        x = x.view(
            B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C
        )
        windows = (
            x.permute(0, 5, 1, 3, 2, 4, 6, 7)
            .contiguous()
            .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
        )
        return windows
    elif ndim == 2:
        B, Lat, Lon, C = x.shape
        win_lat, win_lon = window_size
        x = x.view(B, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
        windows = (
            x.permute(0, 3, 1, 2, 4, 5)
            .contiguous()
            .view(-1, (Lat // win_lat), win_lat, win_lon, C)
        )
        return windows


def window_reverse(windows, window_size, Pl=1, Lat=1, Lon=1, ndim=3):
    """
    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C) or (B*num_lon, num_lat, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon] or [win_lat, win_lon]
        Pl (int): pressure levels
        Lat (int): latitude
        Lon (int): longitude
        ndim (int): dimension of window (3 or 2)

    Returns:
        x: (B, Pl, Lat, Lon, C) or (B, Lat, Lon, C)
    """
    if ndim == 3:
        win_pl, win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(
            B,
            Lon // win_lon,
            Pl // win_pl,
            Lat // win_lat,
            win_pl,
            win_lat,
            win_lon,
            -1,
        )
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
        return x
    elif ndim == 2:
        win_lat, win_lon = window_size
        B = int(windows.shape[0] / (Lon / win_lon))
        x = windows.view(B, Lon // win_lon, Lat // win_lat, win_lat, win_lon, -1)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, Lat, Lon, -1)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size, ndim=3):
    """
    Along the longitude dimension, the leftmost and rightmost indices are actually close to each other.
    If half windows apper at both leftmost and rightmost positions, they are dircetly merged into one window.
    Args:
        input_resolution (tuple[int]): [pressure levels, latitude, longitude] or [latitude, longitude]
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude] or [latitude, longitude]
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude] or [latitude, longitude]
        ndim (int): dimension of window (3 or 2)

    Returns:
        attn_mask: (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon) or (n_lon, n_lat, win_lat*win_lon, win_lat*win_lon)
    """
    if ndim == 3:
        Pl, Lat, Lon = input_resolution
        win_pl, win_lat, win_lon = window_size
        shift_pl, shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))
    elif ndim == 2:
        Lat, Lon = input_resolution
        win_lat, win_lon = window_size
        shift_lat, shift_lon = shift_size

        img_mask = torch.zeros((1, Lat, Lon + shift_lon, 1))

    if ndim == 3:
        pl_slices = (
            slice(0, -win_pl),
            slice(-win_pl, -shift_pl),
            slice(-shift_pl, None),
        )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )

    cnt = 0
    if ndim == 3:
        for pl in pl_slices:
            for lat in lat_slices:
                for lon in lon_slices:
                    img_mask[:, pl, lat, lon, :] = cnt
                    cnt += 1
        img_mask = img_mask[:, :, :, :Lon, :]
    elif ndim == 2:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, lat, lon, :] = cnt
                cnt += 1
        img_mask = img_mask[:, :, :Lon, :]

    mask_windows = window_partition(
        img_mask, window_size, ndim=ndim
    )  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1 or n_lon, n_lat, win_lat, win_lon, 1
    if ndim == 3:
        win_total = win_pl * win_lat * win_lon
    elif ndim == 2:
        win_total = win_lat * win_lon
    mask_windows = mask_windows.view(
        mask_windows.shape[0], mask_windows.shape[1], win_total
    )
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask
