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

from ...._rectilinear_grid_warp_utils import _axis_second_coeff


@wp.kernel
def _laplacian_1d_kernel(
    field: wp.array(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    output: wp.array(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = field.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    c0 = _axis_second_coeff(x0, period0, i)
    output[i] = c0[0] * field[im] + c0[1] * field[i] + c0[2] * field[ip]


@wp.kernel
def _laplacian_2d_kernel(
    field: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = field.shape[0]
    n1 = field.shape[1]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    c0 = _axis_second_coeff(x0, period0, i)
    c1 = _axis_second_coeff(x1, period1, j)
    d2x = c0[0] * field[im, j] + c0[1] * field[i, j] + c0[2] * field[ip, j]
    d2y = c1[0] * field[i, jm] + c1[1] * field[i, j] + c1[2] * field[i, jp]
    output[i, j] = d2x + d2y


@wp.kernel
def _laplacian_3d_kernel(
    field: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    output: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
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
    c0 = _axis_second_coeff(x0, period0, i)
    c1 = _axis_second_coeff(x1, period1, j)
    c2 = _axis_second_coeff(x2, period2, k)
    d2x = c0[0] * field[im, j, k] + c0[1] * field[i, j, k] + c0[2] * field[ip, j, k]
    d2y = c1[0] * field[i, jm, k] + c1[1] * field[i, j, k] + c1[2] * field[i, jp, k]
    d2z = c2[0] * field[i, j, km] + c2[1] * field[i, j, k] + c2[2] * field[i, j, kp]
    output[i, j, k] = d2x + d2y + d2z
