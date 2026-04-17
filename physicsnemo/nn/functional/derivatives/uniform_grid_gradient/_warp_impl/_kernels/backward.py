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
### Backward kernels (adjoint central differences)
### ============================================================


@wp.kernel
def _uniform_grid_gradient_1d_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    inv_dx: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    grad_field[i] = (grad0[im] - grad0[ip]) * (0.5 * inv_dx)


@wp.kernel
def _uniform_grid_gradient_1d_order4_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    inv_dx: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]

    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)

    grad_field[i] = (grad0[ip2] - 8.0 * grad0[ip1] + 8.0 * grad0[im1] - grad0[im2]) * (
        inv_dx / 12.0
    )


@wp.kernel
def _uniform_grid_gradient_2d_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    grad_field[i, j] = (grad0[im, j] - grad0[ip, j]) * (0.5 * inv_dx0) + (
        grad1[i, jm] - grad1[i, jp]
    ) * (0.5 * inv_dx1)


@wp.kernel
def _uniform_grid_gradient_2d_order4_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)

    jm1 = _wrap_minus1(j, n1)
    jp1 = _wrap_plus1(j, n1)
    jm2 = _wrap_minus2(j, n1)
    jp2 = _wrap_plus2(j, n1)

    gx = (grad0[ip2, j] - 8.0 * grad0[ip1, j] + 8.0 * grad0[im1, j] - grad0[im2, j]) * (
        inv_dx0 / 12.0
    )
    gy = (grad1[i, jp2] - 8.0 * grad1[i, jp1] + 8.0 * grad1[i, jm1] - grad1[i, jm2]) * (
        inv_dx1 / 12.0
    )
    grad_field[i, j] = gx + gy


@wp.kernel
def _uniform_grid_gradient_3d_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)

    grad_field[i, j, k] = (
        (grad0[im, j, k] - grad0[ip, j, k]) * (0.5 * inv_dx0)
        + (grad1[i, jm, k] - grad1[i, jp, k]) * (0.5 * inv_dx1)
        + (grad2[i, j, km] - grad2[i, j, kp]) * (0.5 * inv_dx2)
    )


@wp.kernel
def _uniform_grid_gradient_3d_order4_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

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

    gx = (
        grad0[ip2, j, k]
        - 8.0 * grad0[ip1, j, k]
        + 8.0 * grad0[im1, j, k]
        - grad0[im2, j, k]
    ) * (inv_dx0 / 12.0)
    gy = (
        grad1[i, jp2, k]
        - 8.0 * grad1[i, jp1, k]
        + 8.0 * grad1[i, jm1, k]
        - grad1[i, jm2, k]
    ) * (inv_dx1 / 12.0)
    gz = (
        grad2[i, j, kp2]
        - 8.0 * grad2[i, j, kp1]
        + 8.0 * grad2[i, j, km1]
        - grad2[i, j, km2]
    ) * (inv_dx2 / 12.0)
    grad_field[i, j, k] = gx + gy + gz


@wp.kernel
def _uniform_grid_second_derivative_1d_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    inv_dx20: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    grad_field[i] = (grad0[ip] - 2.0 * grad0[i] + grad0[im]) * inv_dx20


@wp.kernel
def _uniform_grid_second_derivative_1d_order4_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    inv_dx20: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]

    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)

    grad_field[i] = (
        -grad0[ip2]
        + 16.0 * grad0[ip1]
        - 30.0 * grad0[i]
        + 16.0 * grad0[im1]
        - grad0[im2]
    ) * (inv_dx20 / 12.0)


@wp.kernel
def _uniform_grid_second_derivative_2d_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    gxx = (grad0[ip, j] - 2.0 * grad0[i, j] + grad0[im, j]) * inv_dx20
    gyy = (grad1[i, jp] - 2.0 * grad1[i, j] + grad1[i, jm]) * inv_dx21
    grad_field[i, j] = gxx + gyy


@wp.kernel
def _uniform_grid_second_derivative_2d_order4_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)

    jm1 = _wrap_minus1(j, n1)
    jp1 = _wrap_plus1(j, n1)
    jm2 = _wrap_minus2(j, n1)
    jp2 = _wrap_plus2(j, n1)

    gxx = (
        -grad0[ip2, j]
        + 16.0 * grad0[ip1, j]
        - 30.0 * grad0[i, j]
        + 16.0 * grad0[im1, j]
        - grad0[im2, j]
    ) * (inv_dx20 / 12.0)
    gyy = (
        -grad1[i, jp2]
        + 16.0 * grad1[i, jp1]
        - 30.0 * grad1[i, j]
        + 16.0 * grad1[i, jm1]
        - grad1[i, jm2]
    ) * (inv_dx21 / 12.0)
    grad_field[i, j] = gxx + gyy


@wp.kernel
def _uniform_grid_second_derivative_3d_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)

    gxx = (grad0[ip, j, k] - 2.0 * grad0[i, j, k] + grad0[im, j, k]) * inv_dx20
    gyy = (grad1[i, jp, k] - 2.0 * grad1[i, j, k] + grad1[i, jm, k]) * inv_dx21
    gzz = (grad2[i, j, kp] - 2.0 * grad2[i, j, k] + grad2[i, j, km]) * inv_dx22
    grad_field[i, j, k] = gxx + gyy + gzz


@wp.kernel
def _uniform_grid_second_derivative_3d_order4_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

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

    gxx = (
        -grad0[ip2, j, k]
        + 16.0 * grad0[ip1, j, k]
        - 30.0 * grad0[i, j, k]
        + 16.0 * grad0[im1, j, k]
        - grad0[im2, j, k]
    ) * (inv_dx20 / 12.0)
    gyy = (
        -grad1[i, jp2, k]
        + 16.0 * grad1[i, jp1, k]
        - 30.0 * grad1[i, j, k]
        + 16.0 * grad1[i, jm1, k]
        - grad1[i, jm2, k]
    ) * (inv_dx21 / 12.0)
    gzz = (
        -grad2[i, j, kp2]
        + 16.0 * grad2[i, j, kp1]
        - 30.0 * grad2[i, j, k]
        + 16.0 * grad2[i, j, km1]
        - grad2[i, j, km2]
    ) * (inv_dx22 / 12.0)
    grad_field[i, j, k] = gxx + gyy + gzz


### ============================================================
### Fused backward kernels for order=2 (first+second, no mixed)
### ============================================================


@wp.kernel
def _uniform_grid_derivatives_1d_order2_fused_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    grad00: wp.array(dtype=wp.float32),
    inv_dx: float,
    inv_dx2: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)

    g1 = (grad0[im] - grad0[ip]) * (0.5 * inv_dx)
    g2 = (grad00[ip] - 2.0 * grad00[i] + grad00[im]) * inv_dx2
    grad_field[i] = g1 + g2


@wp.kernel
def _uniform_grid_derivatives_2d_order2_fused_no_mixed_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    grad00: wp.array2d(dtype=wp.float32),
    grad11: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx20: float,
    inv_dx21: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    g1x = (grad0[im, j] - grad0[ip, j]) * (0.5 * inv_dx0)
    g1y = (grad1[i, jm] - grad1[i, jp]) * (0.5 * inv_dx1)
    g2x = (grad00[ip, j] - 2.0 * grad00[i, j] + grad00[im, j]) * inv_dx20
    g2y = (grad11[i, jp] - 2.0 * grad11[i, j] + grad11[i, jm]) * inv_dx21
    grad_field[i, j] = g1x + g1y + g2x + g2y


@wp.kernel
def _uniform_grid_derivatives_3d_order2_fused_no_mixed_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    grad00: wp.array3d(dtype=wp.float32),
    grad11: wp.array3d(dtype=wp.float32),
    grad22: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)

    g1x = (grad0[im, j, k] - grad0[ip, j, k]) * (0.5 * inv_dx0)
    g1y = (grad1[i, jm, k] - grad1[i, jp, k]) * (0.5 * inv_dx1)
    g1z = (grad2[i, j, km] - grad2[i, j, kp]) * (0.5 * inv_dx2)
    g2x = (grad00[ip, j, k] - 2.0 * grad00[i, j, k] + grad00[im, j, k]) * inv_dx20
    g2y = (grad11[i, jp, k] - 2.0 * grad11[i, j, k] + grad11[i, jm, k]) * inv_dx21
    g2z = (grad22[i, j, kp] - 2.0 * grad22[i, j, k] + grad22[i, j, km]) * inv_dx22
    grad_field[i, j, k] = g1x + g1y + g1z + g2x + g2y + g2z


@wp.kernel
def _uniform_grid_derivatives_2d_order2_fused_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    grad00: wp.array2d(dtype=wp.float32),
    grad11: wp.array2d(dtype=wp.float32),
    grad01: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx20: float,
    inv_dx21: float,
    inv_dx01: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)

    g1x = (grad0[im, j] - grad0[ip, j]) * (0.5 * inv_dx0)
    g1y = (grad1[i, jm] - grad1[i, jp]) * (0.5 * inv_dx1)
    g2x = (grad00[ip, j] - 2.0 * grad00[i, j] + grad00[im, j]) * inv_dx20
    g2y = (grad11[i, jp] - 2.0 * grad11[i, j] + grad11[i, jm]) * inv_dx21
    gm = (grad01[im, jm] - grad01[im, jp] - grad01[ip, jm] + grad01[ip, jp]) * (
        0.25 * inv_dx01
    )
    grad_field[i, j] = g1x + g1y + g2x + g2y + gm


@wp.kernel
def _uniform_grid_derivatives_3d_order2_fused_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    grad00: wp.array3d(dtype=wp.float32),
    grad11: wp.array3d(dtype=wp.float32),
    grad22: wp.array3d(dtype=wp.float32),
    grad01: wp.array3d(dtype=wp.float32),
    grad02: wp.array3d(dtype=wp.float32),
    grad12: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    inv_dx20: float,
    inv_dx21: float,
    inv_dx22: float,
    inv_dx01: float,
    inv_dx02: float,
    inv_dx12: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)

    g1x = (grad0[im, j, k] - grad0[ip, j, k]) * (0.5 * inv_dx0)
    g1y = (grad1[i, jm, k] - grad1[i, jp, k]) * (0.5 * inv_dx1)
    g1z = (grad2[i, j, km] - grad2[i, j, kp]) * (0.5 * inv_dx2)

    g2x = (grad00[ip, j, k] - 2.0 * grad00[i, j, k] + grad00[im, j, k]) * inv_dx20
    g2y = (grad11[i, jp, k] - 2.0 * grad11[i, j, k] + grad11[i, jm, k]) * inv_dx21
    g2z = (grad22[i, j, kp] - 2.0 * grad22[i, j, k] + grad22[i, j, km]) * inv_dx22

    gm01 = (
        grad01[im, jm, k] - grad01[im, jp, k] - grad01[ip, jm, k] + grad01[ip, jp, k]
    ) * (0.25 * inv_dx01)
    gm02 = (
        grad02[im, j, km] - grad02[im, j, kp] - grad02[ip, j, km] + grad02[ip, j, kp]
    ) * (0.25 * inv_dx02)
    gm12 = (
        grad12[i, jm, km] - grad12[i, jm, kp] - grad12[i, jp, km] + grad12[i, jp, kp]
    ) * (0.25 * inv_dx12)
    grad_field[i, j, k] = g1x + g1y + g1z + g2x + g2y + g2z + gm01 + gm02 + gm12
