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

from ....uniform_grid_gradient._warp_impl.utils import (
    _wrap_minus1,
    _wrap_minus2,
    _wrap_plus1,
    _wrap_plus2,
)


@wp.kernel
def _divergence_1d_order2_kernel(
    vector_field: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    output: wp.array(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = output.shape[0]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    output[i] = (vector_field[0, ip] - vector_field[0, im]) * (0.5 * inv_dx0)


@wp.kernel
def _divergence_1d_order4_kernel(
    vector_field: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    output: wp.array(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = output.shape[0]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    output[i] = (
        -vector_field[0, ip2]
        + 8.0 * vector_field[0, ip1]
        - 8.0 * vector_field[0, im1]
        + vector_field[0, im2]
    ) * (inv_dx0 / 12.0)


@wp.kernel
def _divergence_2d_order2_kernel(
    vector_field: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    div_x = (vector_field[0, ip, j] - vector_field[0, im, j]) * (0.5 * inv_dx0)
    div_y = (vector_field[1, i, jp] - vector_field[1, i, jm]) * (0.5 * inv_dx1)
    output[i, j] = div_x + div_y


@wp.kernel
def _divergence_2d_order4_kernel(
    vector_field: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    jm1 = _wrap_minus1(j, n1)
    jp1 = _wrap_plus1(j, n1)
    jm2 = _wrap_minus2(j, n1)
    jp2 = _wrap_plus2(j, n1)
    div_x = (
        -vector_field[0, ip2, j]
        + 8.0 * vector_field[0, ip1, j]
        - 8.0 * vector_field[0, im1, j]
        + vector_field[0, im2, j]
    ) * (inv_dx0 / 12.0)
    div_y = (
        -vector_field[1, i, jp2]
        + 8.0 * vector_field[1, i, jp1]
        - 8.0 * vector_field[1, i, jm1]
        + vector_field[1, i, jm2]
    ) * (inv_dx1 / 12.0)
    output[i, j] = div_x + div_y


@wp.kernel
def _divergence_3d_order2_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    output: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    n2 = output.shape[2]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)
    div_x = (vector_field[0, ip, j, k] - vector_field[0, im, j, k]) * (0.5 * inv_dx0)
    div_y = (vector_field[1, i, jp, k] - vector_field[1, i, jm, k]) * (0.5 * inv_dx1)
    div_z = (vector_field[2, i, j, kp] - vector_field[2, i, j, km]) * (0.5 * inv_dx2)
    output[i, j, k] = div_x + div_y + div_z


@wp.kernel
def _divergence_3d_order4_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    output: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    n2 = output.shape[2]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    jm1 = _wrap_minus1(j, n1)
    jp1 = _wrap_plus1(j, n1)
    jm2 = _wrap_minus2(j, n1)
    jp2 = _wrap_plus2(j, n1)
    km1 = _wrap_minus1(k, n2)
    kp1 = _wrap_plus1(k, n2)
    km2 = _wrap_minus2(k, n2)
    kp2 = _wrap_plus2(k, n2)
    div_x = (
        -vector_field[0, ip2, j, k]
        + 8.0 * vector_field[0, ip1, j, k]
        - 8.0 * vector_field[0, im1, j, k]
        + vector_field[0, im2, j, k]
    ) * (inv_dx0 / 12.0)
    div_y = (
        -vector_field[1, i, jp2, k]
        + 8.0 * vector_field[1, i, jp1, k]
        - 8.0 * vector_field[1, i, jm1, k]
        + vector_field[1, i, jm2, k]
    ) * (inv_dx1 / 12.0)
    div_z = (
        -vector_field[2, i, j, kp2]
        + 8.0 * vector_field[2, i, j, kp1]
        - 8.0 * vector_field[2, i, j, km1]
        + vector_field[2, i, j, km2]
    ) * (inv_dx2 / 12.0)
    output[i, j, k] = div_x + div_y + div_z
