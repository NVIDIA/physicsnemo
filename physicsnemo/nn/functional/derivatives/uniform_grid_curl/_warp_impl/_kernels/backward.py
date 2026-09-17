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
def _curl_backward_2d_order2_kernel(
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
    grad_vector[0, i, j] = (grad_output[i, jp] - grad_output[i, jm]) * (0.5 * inv_dx1)
    grad_vector[1, i, j] = (grad_output[im, j] - grad_output[ip, j]) * (0.5 * inv_dx0)


@wp.kernel
def _curl_backward_2d_order4_kernel(
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
        -grad_output[i, jp2]
        + 8.0 * grad_output[i, jp1]
        - 8.0 * grad_output[i, jm1]
        + grad_output[i, jm2]
    ) * (inv_dx1 / 12.0)
    grad_vector[1, i, j] = (
        grad_output[ip2, j]
        - 8.0 * grad_output[ip1, j]
        + 8.0 * grad_output[im1, j]
        - grad_output[im2, j]
    ) * (inv_dx0 / 12.0)
