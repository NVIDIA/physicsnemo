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
def _divergence_backward_1d_kernel(
    grad_output: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    grad_vector: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = grad_output.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    ci = _axis_coeff(x0, period0, i)
    cip = _axis_coeff(x0, period0, ip)
    cim = _axis_coeff(x0, period0, im)
    grad_vector[0, i] = (
        ci[1] * grad_output[i] + cip[0] * grad_output[ip] + cim[2] * grad_output[im]
    )


@wp.kernel
def _divergence_backward_2d_kernel(
    grad_output: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    grad_vector: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = grad_output.shape[0]
    n1 = grad_output.shape[1]
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
    grad_vector[0, i, j] = (
        cxi[1] * grad_output[i, j]
        + cxip[0] * grad_output[ip, j]
        + cxim[2] * grad_output[im, j]
    )
    grad_vector[1, i, j] = (
        cyi[1] * grad_output[i, j]
        + cyip[0] * grad_output[i, jp]
        + cyim[2] * grad_output[i, jm]
    )


@wp.kernel
def _divergence_backward_3d_kernel(
    grad_output: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    grad_vector: wp.array4d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = grad_output.shape[0]
    n1 = grad_output.shape[1]
    n2 = grad_output.shape[2]
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
    grad_vector[0, i, j, k] = (
        cxi[1] * grad_output[i, j, k]
        + cxip[0] * grad_output[ip, j, k]
        + cxim[2] * grad_output[im, j, k]
    )
    grad_vector[1, i, j, k] = (
        cyi[1] * grad_output[i, j, k]
        + cyip[0] * grad_output[i, jp, k]
        + cyim[2] * grad_output[i, jm, k]
    )
    grad_vector[2, i, j, k] = (
        czi[1] * grad_output[i, j, k]
        + czip[0] * grad_output[i, j, kp]
        + czim[2] * grad_output[i, j, km]
    )
