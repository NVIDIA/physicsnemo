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


# TODO bug in warp mod function
@wp.func
def _mod_int(x: int, length: int):  # pragma: no cover
    """Mod int

    Parameters
    ----------
    x : int
        Int to mod
    length : int
        Mod by value

    Returns
    -------
    int
        Mod of x
    """
    if x < 0:
        return x + length
    elif x > length - 1:
        return x - length
    return x


@wp.func
def index_zero_edges_batched_2d(
    array: wp.array3d(dtype=float), b: int, x: int, y: int, lx: int, ly: int
):  # pragma: no cover
    """Index batched 2d array with zero on edges

    Parameters
    ----------
    array : wp.array3d
        Array to index
    b : int
        Batch index
    x : int
        X index
    y : int
        Y index
    lx : int
        Grid size x
    ly : int
        Grid size y

    Returns
    -------
    float
        Array value
    """
    if x == -1:
        return 0.0
    elif x == lx:
        return 0.0
    elif y == -1:
        return 0.0
    elif y == ly:
        return 0.0
    else:
        return array[b, x, y]


@wp.func
def index_clamped_edges_batched_2d(
    array: wp.array3d(dtype=float), b: int, x: int, y: int, lx: int, ly: int
):  # pragma: no cover
    """Index batched 2d array with edges clamped to same value

    Parameters
    ----------
    array : wp.array3d
        Array to index
    b : int
        Batch index
    x : int
        X index
    y : int
        Y index
    lx : int
        Grid size x
    ly : int
        Grid size y

    Returns
    -------
    float
        Array value
    """
    x = wp.clamp(x, 0, lx - 1)
    y = wp.clamp(y, 0, ly - 1)
    return array[b, x, y]


@wp.func
def index_periodic_edges_batched_2d(
    array: wp.array3d(dtype=float), b: int, x: int, y: int, lx: int, ly: int
):  # pragma: no cover
    """Index batched 2d array with periodic edges

    Parameters
    ----------
    array : wp.array3d
        Array to index
    b : int
        Batch index
    x : int
        X index
    y : int
        Y index
    lx : int
        Grid size x
    ly : int
        Grid size y

    Returns
    -------
    float
        Array value
    """
    x = _mod_int(x, lx)
    y = _mod_int(y, ly)
    return array[b, x, y]


@wp.func
def index_vec2_periodic_edges_batched_2d(
    vec: wp.array3d(dtype=wp.vec2), b: int, x: int, y: int, lx: int, ly: int
):  # pragma: no cover
    """Index batched 2d array of wp.vec2 with periodic edges

    Parameters
    ----------
    vec : wp.array3d
        Array to index
    b : int
        Batch index
    x : int
        X index
    y : int
        Y index
    lx : int
        Grid size x
    ly : int
        Grid size y

    Returns
    -------
    wp.vec2
        Vector value
    """
    x = _mod_int(x, lx)
    y = _mod_int(y, ly)
    return vec[b, x, y]
