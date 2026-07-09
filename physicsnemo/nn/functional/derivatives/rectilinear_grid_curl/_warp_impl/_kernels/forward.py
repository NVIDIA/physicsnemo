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

from __future__ import annotations

import warp as wp

from ...._rectilinear_grid_warp_utils import _axis_coeff


@wp.kernel
def _curl_2d_kernel(
    vector_field: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    c0 = _axis_coeff(x0, period0, i)
    c1 = _axis_coeff(x1, period1, j)
    dvy_dx = (
        c0[0] * vector_field[1, im, j]
        + c0[1] * vector_field[1, i, j]
        + c0[2] * vector_field[1, ip, j]
    )
    dvx_dy = (
        c1[0] * vector_field[0, i, jm]
        + c1[1] * vector_field[0, i, j]
        + c1[2] * vector_field[0, i, jp]
    )
    output[i, j] = dvy_dx - dvx_dy


@wp.kernel
def _curl_3d_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    output: wp.array4d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = output.shape[1]
    n1 = output.shape[2]
    n2 = output.shape[3]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2
    c0 = _axis_coeff(x0, period0, i)
    c1 = _axis_coeff(x1, period1, j)
    c2 = _axis_coeff(x2, period2, k)
    dvz_dy = (
        c1[0] * vector_field[2, i, jm, k]
        + c1[1] * vector_field[2, i, j, k]
        + c1[2] * vector_field[2, i, jp, k]
    )
    dvy_dz = (
        c2[0] * vector_field[1, i, j, km]
        + c2[1] * vector_field[1, i, j, k]
        + c2[2] * vector_field[1, i, j, kp]
    )
    dvx_dz = (
        c2[0] * vector_field[0, i, j, km]
        + c2[1] * vector_field[0, i, j, k]
        + c2[2] * vector_field[0, i, j, kp]
    )
    dvz_dx = (
        c0[0] * vector_field[2, im, j, k]
        + c0[1] * vector_field[2, i, j, k]
        + c0[2] * vector_field[2, ip, j, k]
    )
    dvy_dx = (
        c0[0] * vector_field[1, im, j, k]
        + c0[1] * vector_field[1, i, j, k]
        + c0[2] * vector_field[1, ip, j, k]
    )
    dvx_dy = (
        c1[0] * vector_field[0, i, jm, k]
        + c1[1] * vector_field[0, i, j, k]
        + c1[2] * vector_field[0, i, jp, k]
    )
    output[0, i, j, k] = dvz_dy - dvy_dz
    output[1, i, j, k] = dvx_dz - dvz_dx
    output[2, i, j, k] = dvy_dx - dvx_dy
