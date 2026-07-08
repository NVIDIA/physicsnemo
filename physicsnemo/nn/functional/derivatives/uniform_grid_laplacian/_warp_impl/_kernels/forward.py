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
def _laplacian_1d_order2_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx0_sq: float,
    output: wp.array(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = field.shape[0]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    output[i] = (field[ip] - 2.0 * field[i] + field[im]) * inv_dx0_sq


@wp.kernel
def _laplacian_1d_order4_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx0_sq: float,
    output: wp.array(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = field.shape[0]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    output[i] = (
        -field[ip2]
        + 16.0 * field[ip1]
        - 30.0 * field[i]
        + 16.0 * field[im1]
        - field[im2]
    ) * (inv_dx0_sq / 12.0)


@wp.kernel
def _laplacian_2d_order2_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx0_sq: float,
    inv_dx1_sq: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    d2x = (field[ip, j] - 2.0 * field[i, j] + field[im, j]) * inv_dx0_sq
    d2y = (field[i, jp] - 2.0 * field[i, j] + field[i, jm]) * inv_dx1_sq
    output[i, j] = d2x + d2y


@wp.kernel
def _laplacian_2d_order4_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx0_sq: float,
    inv_dx1_sq: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    jm1 = _wrap_minus1(j, n1)
    jp1 = _wrap_plus1(j, n1)
    jm2 = _wrap_minus2(j, n1)
    jp2 = _wrap_plus2(j, n1)
    d2x = (
        -field[ip2, j]
        + 16.0 * field[ip1, j]
        - 30.0 * field[i, j]
        + 16.0 * field[im1, j]
        - field[im2, j]
    ) * (inv_dx0_sq / 12.0)
    d2y = (
        -field[i, jp2]
        + 16.0 * field[i, jp1]
        - 30.0 * field[i, j]
        + 16.0 * field[i, jm1]
        - field[i, jm2]
    ) * (inv_dx1_sq / 12.0)
    output[i, j] = d2x + d2y


@wp.kernel
def _laplacian_3d_order2_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx0_sq: float,
    inv_dx1_sq: float,
    inv_dx2_sq: float,
    output: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    n2 = field.shape[2]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)
    d2x = (field[ip, j, k] - 2.0 * field[i, j, k] + field[im, j, k]) * inv_dx0_sq
    d2y = (field[i, jp, k] - 2.0 * field[i, j, k] + field[i, jm, k]) * inv_dx1_sq
    d2z = (field[i, j, kp] - 2.0 * field[i, j, k] + field[i, j, km]) * inv_dx2_sq
    output[i, j, k] = d2x + d2y + d2z


@wp.kernel
def _laplacian_3d_order4_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx0_sq: float,
    inv_dx1_sq: float,
    inv_dx2_sq: float,
    output: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    n2 = field.shape[2]
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
    d2x = (
        -field[ip2, j, k]
        + 16.0 * field[ip1, j, k]
        - 30.0 * field[i, j, k]
        + 16.0 * field[im1, j, k]
        - field[im2, j, k]
    ) * (inv_dx0_sq / 12.0)
    d2y = (
        -field[i, jp2, k]
        + 16.0 * field[i, jp1, k]
        - 30.0 * field[i, j, k]
        + 16.0 * field[i, jm1, k]
        - field[i, jm2, k]
    ) * (inv_dx1_sq / 12.0)
    d2z = (
        -field[i, j, kp2]
        + 16.0 * field[i, j, kp1]
        - 30.0 * field[i, j, k]
        + 16.0 * field[i, j, km1]
        - field[i, j, km2]
    ) * (inv_dx2_sq / 12.0)
    output[i, j, k] = d2x + d2y + d2z
