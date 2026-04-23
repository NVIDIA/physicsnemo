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
### Forward kernels (rectilinear periodic central differences)
### ============================================================


@wp.kernel
def _rectilinear_gradient_1d_kernel(
    field: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad0: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0

    coeff = _axis_coeff(x0, period0, i)
    grad0[i] = coeff[0] * field[im] + coeff[1] * field[i] + coeff[2] * field[ip]


@wp.kernel
def _rectilinear_gradient_2d_kernel(
    field: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1

    cx = _axis_coeff(x0, period0, i)
    cy = _axis_coeff(x1, period1, j)

    grad0[i, j] = cx[0] * field[im, j] + cx[1] * field[i, j] + cx[2] * field[ip, j]
    grad1[i, j] = cy[0] * field[i, jm] + cy[1] * field[i, j] + cy[2] * field[i, jp]


@wp.kernel
def _rectilinear_gradient_3d_kernel(
    field: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    n2 = field.shape[2]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2

    cx = _axis_coeff(x0, period0, i)
    cy = _axis_coeff(x1, period1, j)
    cz = _axis_coeff(x2, period2, k)

    grad0[i, j, k] = (
        cx[0] * field[im, j, k] + cx[1] * field[i, j, k] + cx[2] * field[ip, j, k]
    )
    grad1[i, j, k] = (
        cy[0] * field[i, jm, k] + cy[1] * field[i, j, k] + cy[2] * field[i, jp, k]
    )
    grad2[i, j, k] = (
        cz[0] * field[i, j, km] + cz[1] * field[i, j, k] + cz[2] * field[i, j, kp]
    )


@wp.kernel
def _rectilinear_second_derivative_1d_kernel(
    field: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad0: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0

    coeff = _axis_second_coeff(x0, period0, i)
    grad0[i] = coeff[0] * field[im] + coeff[1] * field[i] + coeff[2] * field[ip]


@wp.kernel
def _rectilinear_second_derivative_2d_kernel(
    field: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1

    cx = _axis_second_coeff(x0, period0, i)
    cy = _axis_second_coeff(x1, period1, j)

    grad0[i, j] = cx[0] * field[im, j] + cx[1] * field[i, j] + cx[2] * field[ip, j]
    grad1[i, j] = cy[0] * field[i, jm] + cy[1] * field[i, j] + cy[2] * field[i, jp]


@wp.kernel
def _rectilinear_second_derivative_3d_kernel(
    field: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    grad0: wp.array3d(dtype=wp.float32),
    grad1: wp.array3d(dtype=wp.float32),
    grad2: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    n2 = field.shape[2]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2

    cx = _axis_second_coeff(x0, period0, i)
    cy = _axis_second_coeff(x1, period1, j)
    cz = _axis_second_coeff(x2, period2, k)

    grad0[i, j, k] = (
        cx[0] * field[im, j, k] + cx[1] * field[i, j, k] + cx[2] * field[ip, j, k]
    )
    grad1[i, j, k] = (
        cy[0] * field[i, jm, k] + cy[1] * field[i, j, k] + cy[2] * field[i, jp, k]
    )
    grad2[i, j, k] = (
        cz[0] * field[i, j, km] + cz[1] * field[i, j, k] + cz[2] * field[i, j, kp]
    )


### ============================================================
### Fused forward kernels (single launch for 1st+2nd, no mixed)
### ============================================================


@wp.kernel
def _rectilinear_derivatives_1d_fused_no_mixed_kernel(
    field: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad0: wp.array(dtype=wp.float32),
    grad00: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n0 = field.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0

    coeff1 = _axis_coeff(x0, period0, i)
    coeff2 = _axis_second_coeff(x0, period0, i)
    grad0[i] = coeff1[0] * field[im] + coeff1[1] * field[i] + coeff1[2] * field[ip]
    grad00[i] = coeff2[0] * field[im] + coeff2[1] * field[i] + coeff2[2] * field[ip]


@wp.kernel
def _rectilinear_derivatives_2d_fused_no_mixed_kernel(
    field: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad0: wp.array2d(dtype=wp.float32),
    grad1: wp.array2d(dtype=wp.float32),
    grad00: wp.array2d(dtype=wp.float32),
    grad11: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1

    cx1 = _axis_coeff(x0, period0, i)
    cy1 = _axis_coeff(x1, period1, j)
    cx2 = _axis_second_coeff(x0, period0, i)
    cy2 = _axis_second_coeff(x1, period1, j)

    grad0[i, j] = cx1[0] * field[im, j] + cx1[1] * field[i, j] + cx1[2] * field[ip, j]
    grad1[i, j] = cy1[0] * field[i, jm] + cy1[1] * field[i, j] + cy1[2] * field[i, jp]
    grad00[i, j] = cx2[0] * field[im, j] + cx2[1] * field[i, j] + cx2[2] * field[ip, j]
    grad11[i, j] = cy2[0] * field[i, jm] + cy2[1] * field[i, j] + cy2[2] * field[i, jp]


@wp.kernel
def _rectilinear_derivatives_3d_fused_no_mixed_kernel(
    field: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
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
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2

    cx1 = _axis_coeff(x0, period0, i)
    cy1 = _axis_coeff(x1, period1, j)
    cz1 = _axis_coeff(x2, period2, k)
    cx2 = _axis_second_coeff(x0, period0, i)
    cy2 = _axis_second_coeff(x1, period1, j)
    cz2 = _axis_second_coeff(x2, period2, k)

    grad0[i, j, k] = (
        cx1[0] * field[im, j, k] + cx1[1] * field[i, j, k] + cx1[2] * field[ip, j, k]
    )
    grad1[i, j, k] = (
        cy1[0] * field[i, jm, k] + cy1[1] * field[i, j, k] + cy1[2] * field[i, jp, k]
    )
    grad2[i, j, k] = (
        cz1[0] * field[i, j, km] + cz1[1] * field[i, j, k] + cz1[2] * field[i, j, kp]
    )
    grad00[i, j, k] = (
        cx2[0] * field[im, j, k] + cx2[1] * field[i, j, k] + cx2[2] * field[ip, j, k]
    )
    grad11[i, j, k] = (
        cy2[0] * field[i, jm, k] + cy2[1] * field[i, j, k] + cy2[2] * field[i, jp, k]
    )
    grad22[i, j, k] = (
        cz2[0] * field[i, j, km] + cz2[1] * field[i, j, k] + cz2[2] * field[i, j, kp]
    )
