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

### ============================================================
### Index wrapping helpers (periodic boundaries without modulo)
### ============================================================


@wp.func
def _wrap_plus1(i: int, n: int) -> int:
    return (i + 1) % n


@wp.func
def _wrap_minus1(i: int, n: int) -> int:
    return (i + n - 1) % n


@wp.func
def _wrap_plus2(i: int, n: int) -> int:
    return (i + 2) % n


@wp.func
def _wrap_minus2(i: int, n: int) -> int:
    return (i + n - 2) % n


### ============================================================
### Forward kernels (periodic central differences)
### ============================================================


@wp.kernel
def _uniform_grid_gradient_1d_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx: float,
    grad0: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)

    grad0[i] = (field[ip] - field[im]) * (0.5 * inv_dx)


@wp.kernel
def _uniform_grid_gradient_1d_order4_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx: float,
    grad0: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]

    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)

    grad0[i] = (-field[ip2] + 8.0 * field[ip1] - 8.0 * field[im1] + field[im2]) * (
        inv_dx / 12.0
    )


@wp.kernel
def _uniform_grid_gradient_2d_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    grad0[i, j] = (field[ip, j] - field[im, j]) * (0.5 * inv_dx0)
    grad1[i, j] = (field[i, jp] - field[i, jm]) * (0.5 * inv_dx1)


@wp.kernel
def _uniform_grid_gradient_2d_order4_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
):
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

    grad0[i, j] = (
        -field[ip2, j] + 8.0 * field[ip1, j] - 8.0 * field[im1, j] + field[im2, j]
    ) * (inv_dx0 / 12.0)
    grad1[i, j] = (
        -field[i, jp2] + 8.0 * field[i, jp1] - 8.0 * field[i, jm1] + field[i, jm2]
    ) * (inv_dx1 / 12.0)


@wp.kernel
def _uniform_grid_gradient_3d_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
):
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

    grad0[i, j, k] = (field[ip, j, k] - field[im, j, k]) * (0.5 * inv_dx0)
    grad1[i, j, k] = (field[i, jp, k] - field[i, jm, k]) * (0.5 * inv_dx1)
    grad2[i, j, k] = (field[i, j, kp] - field[i, j, km]) * (0.5 * inv_dx2)


@wp.kernel
def _uniform_grid_gradient_3d_order4_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
):
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

    grad0[i, j, k] = (
        -field[ip2, j, k]
        + 8.0 * field[ip1, j, k]
        - 8.0 * field[im1, j, k]
        + field[im2, j, k]
    ) * (inv_dx0 / 12.0)
    grad1[i, j, k] = (
        -field[i, jp2, k]
        + 8.0 * field[i, jp1, k]
        - 8.0 * field[i, jm1, k]
        + field[i, jm2, k]
    ) * (inv_dx1 / 12.0)
    grad2[i, j, k] = (
        -field[i, j, kp2]
        + 8.0 * field[i, j, kp1]
        - 8.0 * field[i, j, km1]
        + field[i, j, km2]
    ) * (inv_dx2 / 12.0)


@wp.kernel
def _uniform_grid_second_derivative_1d_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx2: float,
    grad0: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    grad0[i] = (field[ip] - 2.0 * field[i] + field[im]) * inv_dx2


@wp.kernel
def _uniform_grid_second_derivative_1d_order4_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx2: float,
    grad0: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]

    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    grad0[i] = (
        -field[ip2]
        + 16.0 * field[ip1]
        - 30.0 * field[i]
        + 16.0 * field[im1]
        - field[im2]
    ) * (inv_dx2 / 12.0)


@wp.kernel
def _uniform_grid_second_derivative_2d_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    grad0[i, j] = (field[ip, j] - 2.0 * field[i, j] + field[im, j]) * inv_dx20
    grad1[i, j] = (field[i, jp] - 2.0 * field[i, j] + field[i, jm]) * inv_dx21


@wp.kernel
def _uniform_grid_second_derivative_2d_order4_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
):
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

    grad0[i, j] = (
        -field[ip2, j]
        + 16.0 * field[ip1, j]
        - 30.0 * field[i, j]
        + 16.0 * field[im1, j]
        - field[im2, j]
    ) * (inv_dx20 / 12.0)
    grad1[i, j] = (
        -field[i, jp2]
        + 16.0 * field[i, jp1]
        - 30.0 * field[i, j]
        + 16.0 * field[i, jm1]
        - field[i, jm2]
    ) * (inv_dx21 / 12.0)


@wp.kernel
def _uniform_grid_second_derivative_3d_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
):
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

    grad0[i, j, k] = (
        field[ip, j, k] - 2.0 * field[i, j, k] + field[im, j, k]
    ) * inv_dx20
    grad1[i, j, k] = (
        field[i, jp, k] - 2.0 * field[i, j, k] + field[i, jm, k]
    ) * inv_dx21
    grad2[i, j, k] = (
        field[i, j, kp] - 2.0 * field[i, j, k] + field[i, j, km]
    ) * inv_dx22


@wp.kernel
def _uniform_grid_second_derivative_3d_order4_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
):
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

    grad0[i, j, k] = (
        -field[ip2, j, k]
        + 16.0 * field[ip1, j, k]
        - 30.0 * field[i, j, k]
        + 16.0 * field[im1, j, k]
        - field[im2, j, k]
    ) * (inv_dx20 / 12.0)
    grad1[i, j, k] = (
        -field[i, jp2, k]
        + 16.0 * field[i, jp1, k]
        - 30.0 * field[i, j, k]
        + 16.0 * field[i, jm1, k]
        - field[i, jm2, k]
    ) * (inv_dx21 / 12.0)
    grad2[i, j, k] = (
        -field[i, j, kp2]
        + 16.0 * field[i, j, kp1]
        - 30.0 * field[i, j, k]
        + 16.0 * field[i, j, km1]
        - field[i, j, km2]
    ) * (inv_dx22 / 12.0)


### ============================================================
### Fused forward kernels for order=2 (single launch for 1st+2nd+mixed)
### ============================================================


@wp.kernel
def _uniform_grid_derivatives_1d_order2_fused_kernel(
    field: wp.array(dtype=wp.float32),
    inv_dx: float,
    inv_dx2: float,
    grad0: wp.array(dtype=wp.float32),
    grad00: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)

    grad0[i] = (field[ip] - field[im]) * (0.5 * inv_dx)
    grad00[i] = (field[ip] - 2.0 * field[i] + field[im]) * inv_dx2


@wp.kernel
def _uniform_grid_derivatives_2d_order2_fused_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx20: float,
    inv_dx21: float,
    inv_dx01: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    grad00: wp.array2d(dtype=wp.float32),
    grad11: wp.array2d(dtype=wp.float32),
    grad01: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    grad0[i, j] = (field[ip, j] - field[im, j]) * (0.5 * inv_dx0)
    grad1[i, j] = (field[i, jp] - field[i, jm]) * (0.5 * inv_dx1)

    grad00[i, j] = (field[ip, j] - 2.0 * field[i, j] + field[im, j]) * inv_dx20
    grad11[i, j] = (field[i, jp] - 2.0 * field[i, j] + field[i, jm]) * inv_dx21

    grad01[i, j] = (field[ip, jp] - field[ip, jm] - field[im, jp] + field[im, jm]) * (
        0.25 * inv_dx01
    )


@wp.kernel
def _uniform_grid_derivatives_3d_order2_fused_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    inv_dx01: float,
    inv_dx02: float,
    inv_dx12: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    grad00: wp.array3d(dtype=wp.float32),
    grad11: wp.array3d(dtype=wp.float32),
    grad22: wp.array3d(dtype=wp.float32),
    grad01: wp.array3d(dtype=wp.float32),
    grad02: wp.array3d(dtype=wp.float32),
    grad12: wp.array3d(dtype=wp.float32),
):
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

    grad0[i, j, k] = (field[ip, j, k] - field[im, j, k]) * (0.5 * inv_dx0)
    grad1[i, j, k] = (field[i, jp, k] - field[i, jm, k]) * (0.5 * inv_dx1)
    grad2[i, j, k] = (field[i, j, kp] - field[i, j, km]) * (0.5 * inv_dx2)

    grad00[i, j, k] = (
        field[ip, j, k] - 2.0 * field[i, j, k] + field[im, j, k]
    ) * inv_dx20
    grad11[i, j, k] = (
        field[i, jp, k] - 2.0 * field[i, j, k] + field[i, jm, k]
    ) * inv_dx21
    grad22[i, j, k] = (
        field[i, j, kp] - 2.0 * field[i, j, k] + field[i, j, km]
    ) * inv_dx22

    grad01[i, j, k] = (
        field[ip, jp, k] - field[ip, jm, k] - field[im, jp, k] + field[im, jm, k]
    ) * (0.25 * inv_dx01)
    grad02[i, j, k] = (
        field[ip, j, kp] - field[ip, j, km] - field[im, j, kp] + field[im, j, km]
    ) * (0.25 * inv_dx02)
    grad12[i, j, k] = (
        field[i, jp, kp] - field[i, jp, km] - field[i, jm, kp] + field[i, jm, km]
    ) * (0.25 * inv_dx12)


@wp.kernel
def _uniform_grid_derivatives_2d_order2_fused_no_mixed_kernel(
    field: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx20: float,
    inv_dx21: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    grad00: wp.array2d(dtype=wp.float32),
    grad11: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    grad0[i, j] = (field[ip, j] - field[im, j]) * (0.5 * inv_dx0)
    grad1[i, j] = (field[i, jp] - field[i, jm]) * (0.5 * inv_dx1)

    grad00[i, j] = (field[ip, j] - 2.0 * field[i, j] + field[im, j]) * inv_dx20
    grad11[i, j] = (field[i, jp] - 2.0 * field[i, j] + field[i, jm]) * inv_dx21


@wp.kernel
def _uniform_grid_derivatives_3d_order2_fused_no_mixed_kernel(
    field: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    grad00: wp.array3d(dtype=wp.float32),
    grad11: wp.array3d(dtype=wp.float32),
    grad22: wp.array3d(dtype=wp.float32),
):
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

    grad0[i, j, k] = (field[ip, j, k] - field[im, j, k]) * (0.5 * inv_dx0)
    grad1[i, j, k] = (field[i, jp, k] - field[i, jm, k]) * (0.5 * inv_dx1)
    grad2[i, j, k] = (field[i, j, kp] - field[i, j, km]) * (0.5 * inv_dx2)

    grad00[i, j, k] = (
        field[ip, j, k] - 2.0 * field[i, j, k] + field[im, j, k]
    ) * inv_dx20
    grad11[i, j, k] = (
        field[i, jp, k] - 2.0 * field[i, j, k] + field[i, jm, k]
    ) * inv_dx21
    grad22[i, j, k] = (
        field[i, j, kp] - 2.0 * field[i, j, k] + field[i, j, km]
    ) * inv_dx22
