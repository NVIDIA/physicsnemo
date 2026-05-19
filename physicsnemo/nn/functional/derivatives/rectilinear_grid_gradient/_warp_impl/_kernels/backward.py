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

from .utils import _axis_coeff, _axis_second_coeff

### ============================================================
### Backward kernels (adjoint of rectilinear central differences)
### ============================================================


@wp.kernel
def _rectilinear_gradient_1d_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0

    ci = _axis_coeff(x0, period0, i)
    cip = _axis_coeff(x0, period0, ip)
    cim = _axis_coeff(x0, period0, im)
    grad_field[i] = ci[1] * grad0[i] + cip[0] * grad0[ip] + cim[2] * grad0[im]


@wp.kernel
def _rectilinear_gradient_2d_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1

    cxi = _axis_coeff(x0, period0, i)
    cxip = _axis_coeff(x0, period0, ip)
    cxim = _axis_coeff(x0, period0, im)

    cyi = _axis_coeff(x1, period1, j)
    cyip = _axis_coeff(x1, period1, jp)
    cyim = _axis_coeff(x1, period1, jm)

    gx = cxi[1] * grad0[i, j] + cxip[0] * grad0[ip, j] + cxim[2] * grad0[im, j]
    gy = cyi[1] * grad1[i, j] + cyip[0] * grad1[i, jp] + cyim[2] * grad1[i, jm]
    grad_field[i, j] = gx + gy


@wp.kernel
def _rectilinear_gradient_3d_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2

    cxi = _axis_coeff(x0, period0, i)
    cxip = _axis_coeff(x0, period0, ip)
    cxim = _axis_coeff(x0, period0, im)

    cyi = _axis_coeff(x1, period1, j)
    cyip = _axis_coeff(x1, period1, jp)
    cyim = _axis_coeff(x1, period1, jm)

    czi = _axis_coeff(x2, period2, k)
    czip = _axis_coeff(x2, period2, kp)
    czim = _axis_coeff(x2, period2, km)

    gx = cxi[1] * grad0[i, j, k] + cxip[0] * grad0[ip, j, k] + cxim[2] * grad0[im, j, k]
    gy = cyi[1] * grad1[i, j, k] + cyip[0] * grad1[i, jp, k] + cyim[2] * grad1[i, jm, k]
    gz = czi[1] * grad2[i, j, k] + czip[0] * grad2[i, j, kp] + czim[2] * grad2[i, j, km]
    grad_field[i, j, k] = gx + gy + gz


@wp.kernel
def _rectilinear_second_derivative_1d_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0

    ci = _axis_second_coeff(x0, period0, i)
    cip = _axis_second_coeff(x0, period0, ip)
    cim = _axis_second_coeff(x0, period0, im)
    grad_field[i] = ci[1] * grad0[i] + cip[0] * grad0[ip] + cim[2] * grad0[im]


@wp.kernel
def _rectilinear_second_derivative_2d_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1

    cxi = _axis_second_coeff(x0, period0, i)
    cxip = _axis_second_coeff(x0, period0, ip)
    cxim = _axis_second_coeff(x0, period0, im)

    cyi = _axis_second_coeff(x1, period1, j)
    cyip = _axis_second_coeff(x1, period1, jp)
    cyim = _axis_second_coeff(x1, period1, jm)

    gx = cxi[1] * grad0[i, j] + cxip[0] * grad0[ip, j] + cxim[2] * grad0[im, j]
    gy = cyi[1] * grad1[i, j] + cyip[0] * grad1[i, jp] + cyim[2] * grad1[i, jm]
    grad_field[i, j] = gx + gy


@wp.kernel
def _rectilinear_second_derivative_3d_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2

    cxi = _axis_second_coeff(x0, period0, i)
    cxip = _axis_second_coeff(x0, period0, ip)
    cxim = _axis_second_coeff(x0, period0, im)

    cyi = _axis_second_coeff(x1, period1, j)
    cyip = _axis_second_coeff(x1, period1, jp)
    cyim = _axis_second_coeff(x1, period1, jm)

    czi = _axis_second_coeff(x2, period2, k)
    czip = _axis_second_coeff(x2, period2, kp)
    czim = _axis_second_coeff(x2, period2, km)

    gx = cxi[1] * grad0[i, j, k] + cxip[0] * grad0[ip, j, k] + cxim[2] * grad0[im, j, k]
    gy = cyi[1] * grad1[i, j, k] + cyip[0] * grad1[i, jp, k] + cyim[2] * grad1[i, jm, k]
    gz = czi[1] * grad2[i, j, k] + czip[0] * grad2[i, j, kp] + czim[2] * grad2[i, j, km]
    grad_field[i, j, k] = gx + gy + gz


### ============================================================
### Fused backward kernels (adjoint of 1st+2nd, no mixed)
### ============================================================


@wp.kernel
def _rectilinear_derivatives_1d_fused_no_mixed_backward_kernel(
    grad0: wp.array(dtype=wp.float32),
    grad00: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad_field: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = grad0.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0

    c1_i = _axis_coeff(x0, period0, i)
    c1_ip = _axis_coeff(x0, period0, ip)
    c1_im = _axis_coeff(x0, period0, im)

    c2_i = _axis_second_coeff(x0, period0, i)
    c2_ip = _axis_second_coeff(x0, period0, ip)
    c2_im = _axis_second_coeff(x0, period0, im)

    g1 = c1_i[1] * grad0[i] + c1_ip[0] * grad0[ip] + c1_im[2] * grad0[im]
    g2 = c2_i[1] * grad00[i] + c2_ip[0] * grad00[ip] + c2_im[2] * grad00[im]
    grad_field[i] = g1 + g2


@wp.kernel
def _rectilinear_derivatives_2d_fused_no_mixed_backward_kernel(
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    grad00: wp.array2d(dtype=wp.float32),
    grad11: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad_field: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]

    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1

    cx1_i = _axis_coeff(x0, period0, i)
    cx1_ip = _axis_coeff(x0, period0, ip)
    cx1_im = _axis_coeff(x0, period0, im)
    cy1_i = _axis_coeff(x1, period1, j)
    cy1_ip = _axis_coeff(x1, period1, jp)
    cy1_im = _axis_coeff(x1, period1, jm)

    cx2_i = _axis_second_coeff(x0, period0, i)
    cx2_ip = _axis_second_coeff(x0, period0, ip)
    cx2_im = _axis_second_coeff(x0, period0, im)
    cy2_i = _axis_second_coeff(x1, period1, j)
    cy2_ip = _axis_second_coeff(x1, period1, jp)
    cy2_im = _axis_second_coeff(x1, period1, jm)

    g1x = cx1_i[1] * grad0[i, j] + cx1_ip[0] * grad0[ip, j] + cx1_im[2] * grad0[im, j]
    g1y = cy1_i[1] * grad1[i, j] + cy1_ip[0] * grad1[i, jp] + cy1_im[2] * grad1[i, jm]
    g2x = (
        cx2_i[1] * grad00[i, j] + cx2_ip[0] * grad00[ip, j] + cx2_im[2] * grad00[im, j]
    )
    g2y = (
        cy2_i[1] * grad11[i, j] + cy2_ip[0] * grad11[i, jp] + cy2_im[2] * grad11[i, jm]
    )
    grad_field[i, j] = g1x + g1y + g2x + g2y


@wp.kernel
def _rectilinear_derivatives_3d_fused_no_mixed_backward_kernel(
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
    grad00: wp.array3d(dtype=wp.float32),
    grad11: wp.array3d(dtype=wp.float32),
    grad22: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    grad_field: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = grad0.shape[0]
    n1 = grad0.shape[1]
    n2 = grad0.shape[2]

    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2

    cx1_i = _axis_coeff(x0, period0, i)
    cx1_ip = _axis_coeff(x0, period0, ip)
    cx1_im = _axis_coeff(x0, period0, im)
    cy1_i = _axis_coeff(x1, period1, j)
    cy1_ip = _axis_coeff(x1, period1, jp)
    cy1_im = _axis_coeff(x1, period1, jm)
    cz1_i = _axis_coeff(x2, period2, k)
    cz1_ip = _axis_coeff(x2, period2, kp)
    cz1_im = _axis_coeff(x2, period2, km)

    cx2_i = _axis_second_coeff(x0, period0, i)
    cx2_ip = _axis_second_coeff(x0, period0, ip)
    cx2_im = _axis_second_coeff(x0, period0, im)
    cy2_i = _axis_second_coeff(x1, period1, j)
    cy2_ip = _axis_second_coeff(x1, period1, jp)
    cy2_im = _axis_second_coeff(x1, period1, jm)
    cz2_i = _axis_second_coeff(x2, period2, k)
    cz2_ip = _axis_second_coeff(x2, period2, kp)
    cz2_im = _axis_second_coeff(x2, period2, km)

    g1x = (
        cx1_i[1] * grad0[i, j, k]
        + cx1_ip[0] * grad0[ip, j, k]
        + cx1_im[2] * grad0[im, j, k]
    )
    g1y = (
        cy1_i[1] * grad1[i, j, k]
        + cy1_ip[0] * grad1[i, jp, k]
        + cy1_im[2] * grad1[i, jm, k]
    )
    g1z = (
        cz1_i[1] * grad2[i, j, k]
        + cz1_ip[0] * grad2[i, j, kp]
        + cz1_im[2] * grad2[i, j, km]
    )

    g2x = (
        cx2_i[1] * grad00[i, j, k]
        + cx2_ip[0] * grad00[ip, j, k]
        + cx2_im[2] * grad00[im, j, k]
    )
    g2y = (
        cy2_i[1] * grad11[i, j, k]
        + cy2_ip[0] * grad11[i, jp, k]
        + cy2_im[2] * grad11[i, jm, k]
    )
    g2z = (
        cz2_i[1] * grad22[i, j, k]
        + cz2_ip[0] * grad22[i, j, kp]
        + cz2_im[2] * grad22[i, j, km]
    )
    grad_field[i, j, k] = g1x + g1y + g1z + g2x + g2y + g2z
