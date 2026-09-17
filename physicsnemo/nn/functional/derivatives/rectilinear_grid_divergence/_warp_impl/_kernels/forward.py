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
def _divergence_1d_kernel(
    vector_field: wp.array2d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    period0: float,
    output: wp.array(dtype=wp.float32),
):  # pragma: no cover
    i = wp.tid()
    n0 = output.shape[0]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    c0 = _axis_coeff(x0, period0, i)
    output[i] = (
        c0[0] * vector_field[0, im]
        + c0[1] * vector_field[0, i]
        + c0[2] * vector_field[0, ip]
    )


@wp.kernel
def _divergence_2d_kernel(
    vector_field: wp.array3d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    output: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    i, j = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    c0 = _axis_coeff(x0, period0, i)
    c1 = _axis_coeff(x1, period1, j)
    div_x = (
        c0[0] * vector_field[0, im, j]
        + c0[1] * vector_field[0, i, j]
        + c0[2] * vector_field[0, ip, j]
    )
    div_y = (
        c1[0] * vector_field[1, i, jm]
        + c1[1] * vector_field[1, i, j]
        + c1[2] * vector_field[1, i, jp]
    )
    output[i, j] = div_x + div_y


@wp.kernel
def _divergence_3d_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    x0: wp.array(dtype=wp.float32),
    x1: wp.array(dtype=wp.float32),
    x2: wp.array(dtype=wp.float32),
    period0: float,
    period1: float,
    period2: float,
    output: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    i, j, k = wp.tid()
    n0 = output.shape[0]
    n1 = output.shape[1]
    n2 = output.shape[2]
    im = (i + n0 - 1) % n0
    ip = (i + 1) % n0
    jm = (j + n1 - 1) % n1
    jp = (j + 1) % n1
    km = (k + n2 - 1) % n2
    kp = (k + 1) % n2
    c0 = _axis_coeff(x0, period0, i)
    c1 = _axis_coeff(x1, period1, j)
    c2 = _axis_coeff(x2, period2, k)
    div_x = (
        c0[0] * vector_field[0, im, j, k]
        + c0[1] * vector_field[0, i, j, k]
        + c0[2] * vector_field[0, ip, j, k]
    )
    div_y = (
        c1[0] * vector_field[1, i, jm, k]
        + c1[1] * vector_field[1, i, j, k]
        + c1[2] * vector_field[1, i, jp, k]
    )
    div_z = (
        c2[0] * vector_field[2, i, j, km]
        + c2[1] * vector_field[2, i, j, k]
        + c2[2] * vector_field[2, i, j, kp]
    )
    output[i, j, k] = div_x + div_y + div_z
