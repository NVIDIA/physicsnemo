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
def _divergence_backward_1d_order2_kernel(
    grad_output: wp.array(dtype=wp.float32),
    inv_dx0: float,
    grad_vector: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = grad_output.shape[0]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    grad_vector[0, i] = (grad_output[im] - grad_output[ip]) * (0.5 * inv_dx0)


@wp.kernel
def _divergence_backward_1d_order4_kernel(
    grad_output: wp.array(dtype=wp.float32),
    inv_dx0: float,
    grad_vector: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = grad_output.shape[0]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    grad_vector[0, i] = (
        grad_output[ip2]
        - 8.0 * grad_output[ip1]
        + 8.0 * grad_output[im1]
        - grad_output[im2]
    ) * (inv_dx0 / 12.0)


@wp.kernel
def _divergence_backward_2d_order2_kernel(
    grad_output: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    grad_vector: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = grad_output.shape[0]
    n1 = grad_output.shape[1]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    grad_vector[0, i, j] = (grad_output[im, j] - grad_output[ip, j]) * (0.5 * inv_dx0)
    grad_vector[1, i, j] = (grad_output[i, jm] - grad_output[i, jp]) * (0.5 * inv_dx1)


@wp.kernel
def _divergence_backward_2d_order4_kernel(
    grad_output: wp.array2d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    grad_vector: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = grad_output.shape[0]
    n1 = grad_output.shape[1]
    im1 = _wrap_minus1(i, n0)
    ip1 = _wrap_plus1(i, n0)
    im2 = _wrap_minus2(i, n0)
    ip2 = _wrap_plus2(i, n0)
    jm1 = _wrap_minus1(j, n1)
    jp1 = _wrap_plus1(j, n1)
    jm2 = _wrap_minus2(j, n1)
    jp2 = _wrap_plus2(j, n1)
    grad_vector[0, i, j] = (
        grad_output[ip2, j]
        - 8.0 * grad_output[ip1, j]
        + 8.0 * grad_output[im1, j]
        - grad_output[im2, j]
    ) * (inv_dx0 / 12.0)
    grad_vector[1, i, j] = (
        grad_output[i, jp2]
        - 8.0 * grad_output[i, jp1]
        + 8.0 * grad_output[i, jm1]
        - grad_output[i, jm2]
    ) * (inv_dx1 / 12.0)


@wp.kernel
def _divergence_backward_3d_order2_kernel(
    grad_output: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    grad_vector: wp.array4d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = grad_output.shape[0]
    n1 = grad_output.shape[1]
    n2 = grad_output.shape[2]
    im = _wrap_minus1(i, n0)
    ip = _wrap_plus1(i, n0)
    jm = _wrap_minus1(j, n1)
    jp = _wrap_plus1(j, n1)
    km = _wrap_minus1(k, n2)
    kp = _wrap_plus1(k, n2)
    grad_vector[0, i, j, k] = (grad_output[im, j, k] - grad_output[ip, j, k]) * (
        0.5 * inv_dx0
    )
    grad_vector[1, i, j, k] = (grad_output[i, jm, k] - grad_output[i, jp, k]) * (
        0.5 * inv_dx1
    )
    grad_vector[2, i, j, k] = (grad_output[i, j, km] - grad_output[i, j, kp]) * (
        0.5 * inv_dx2
    )


@wp.kernel
def _divergence_backward_3d_order4_kernel(
    grad_output: wp.array3d(dtype=wp.float32),
    inv_dx0: float,
    inv_dx1: float,
    inv_dx2: float,
    grad_vector: wp.array4d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = grad_output.shape[0]
    n1 = grad_output.shape[1]
    n2 = grad_output.shape[2]
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
    grad_vector[0, i, j, k] = (
        grad_output[ip2, j, k]
        - 8.0 * grad_output[ip1, j, k]
        + 8.0 * grad_output[im1, j, k]
        - grad_output[im2, j, k]
    ) * (inv_dx0 / 12.0)
    grad_vector[1, i, j, k] = (
        grad_output[i, jp2, k]
        - 8.0 * grad_output[i, jp1, k]
        + 8.0 * grad_output[i, jm1, k]
        - grad_output[i, jm2, k]
    ) * (inv_dx1 / 12.0)
    grad_vector[2, i, j, k] = (
        grad_output[i, j, kp2]
        - 8.0 * grad_output[i, j, kp1]
        + 8.0 * grad_output[i, j, km1]
        - grad_output[i, j, km2]
    ) * (inv_dx2 / 12.0)
