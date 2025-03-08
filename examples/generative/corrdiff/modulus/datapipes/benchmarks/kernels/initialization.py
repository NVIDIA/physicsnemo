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


@wp.kernel
def init_uniform_random_2d(
    array: wp.array2d(dtype=float),
    min_value: float,
    max_value: float,
    external_seed: int,
):  # pragma: no cover
    """Initialize 2d array with uniform random values

    Parameters
    ----------
    array : wp.array2d
        Array to initialize
    min_value : float
        Min random value
    max_value : float
        Max random value
    external_seed : int
        External seed to use
    """
    i, j = wp.tid()
    state = wp.rand_init(external_seed, wp.tid())
    array[i, j] = wp.randf(state, -min_value, max_value)


@wp.kernel
def init_uniform_random_4d(
    array: wp.array4d(dtype=float),
    min_value: float,
    max_value: float,
    external_seed: int,
):  # pragma: no cover
    """Initialize 4d array with uniform random values

    Parameters
    ----------
    array : wp.array4d
        Array to initialize
    min_value : float
        Min random value
    max_value : float
        Max random value
    external_seed : int
        External seed to use
    """
    b, i, j, k = wp.tid()
    state = wp.rand_init(external_seed, wp.tid())
    array[b, i, j, k] = wp.randf(state, min_value, max_value)
