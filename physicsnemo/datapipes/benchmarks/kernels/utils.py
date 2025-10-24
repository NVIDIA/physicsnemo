# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

try:
    import warp as wp
except ImportError:
    print(
        """NVIDIA WARP is required for this datapipe. This package is under the 
NVIDIA Source Code License (NVSCL). To install use:

pip install warp-lang
"""
    )
    raise SystemExit(1)

from .indexing import index_zero_edges_batched_2d


@wp.kernel
def bilinear_upsample_batched_2d(
    array: wp.array3d(dtype=float), lx: int, ly: int, grid_reduction_factor: int
):  # pragma: no cover
    """Bilinear upsampling from batch 2d array

    Parameters
    ----------
    array : wp.array3d
        Array to perform upsampling on
    lx : int
        Grid size X
    ly : int
        Grid size Y
    grid_reduction_factor : int
        Grid reduction factor for multi-grid
    """
    # get index
    b, x, y = wp.tid()

    # get four neighbors coordinates
    x_0 = x - (x + 1) % grid_reduction_factor
    x_1 = x + (x + 1) % grid_reduction_factor
    y_0 = y - (y + 1) % grid_reduction_factor
    y_1 = y + (y + 1) % grid_reduction_factor

    # simple linear upsampling
    d_0_0 = index_zero_edges_batched_2d(array, b, x_0, y_0, lx, ly)
    d_1_0 = index_zero_edges_batched_2d(array, b, x_1, y_0, lx, ly)
    d_0_1 = index_zero_edges_batched_2d(array, b, x_0, y_1, lx, ly)
    d_1_1 = index_zero_edges_batched_2d(array, b, x_1, y_1, lx, ly)

    # get relative distance
    rel_x = wp.float32(x - x_0) / wp.float32(grid_reduction_factor)
    rel_y = wp.float32(y - y_0) / wp.float32(grid_reduction_factor)

    # interpolation in x direction
    d_x_0 = (1.0 - rel_x) * d_0_0 + rel_x * d_1_0
    d_x_1 = (1.0 - rel_x) * d_0_1 + rel_x * d_1_1

    # interpolation in y direction
    d = (1.0 - rel_y) * d_x_0 + rel_y * d_x_1

    # set interpolation
    array[b, x, y] = d


@wp.kernel
def threshold_3d(
    array: wp.array3d(dtype=float), threshold: float, min_value: float, max_value: float
):  # pragma: no cover
    """Threshold 3d array by value. Values bellow threshold will be `min_value` and those above will be `max_value`.

    Parameters
    ----------
    array : wp.array3d
        Array to apply threshold on
    threshold : float
        Threshold value
    min_value : float
        Value to set if bellow threshold
    max_value : float
        Value to set if above threshold
    """
    i, j, k = wp.tid()
    if array[i, j, k] < threshold:
        array[i, j, k] = min_value
    else:
        array[i, j, k] = max_value


@wp.kernel
def fourier_to_array_batched_2d(
    array: wp.array3d(dtype=float),
    fourier: wp.array4d(dtype=float),
    nr_freq: int,
    lx: int,
    ly: int,
):  # pragma: no cover
    """Array of Fourier amplitudes to batched 2d spatial array

    Parameters
    ----------
    array : wp.array3d
        Spatial array
    fourier : wp.array4d
        Array of Fourier amplitudes
    nr_freq : int
        Number of frequencies in Fourier array
    lx : int
        Grid size x
    ly : int
        Grid size y
    """
    b, x, y = wp.tid()
    dx = 6.28318 / wp.float32(lx)
    dy = 6.28318 / wp.float32(ly)
    rx = dx * wp.float32(x)
    ry = dy * wp.float32(y)
    for i in range(nr_freq):
        for j in range(nr_freq):
            ri = wp.float32(i)
            rj = wp.float32(j)
            ss = fourier[0, b, i, j] * wp.sin(ri * rx) * wp.sin(rj * ry)
            cs = fourier[1, b, i, j] * wp.cos(ri * rx) * wp.sin(rj * ry)
            sc = fourier[2, b, i, j] * wp.sin(ri * rx) * wp.cos(rj * ry)
            cc = fourier[3, b, i, j] * wp.cos(ri * rx) * wp.cos(rj * ry)
            wp.atomic_add(
                array, b, x, y, 1.0 / (wp.float32(nr_freq) ** 2.0) * (ss + cs + sc + cc)
            )
