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


@wp.func
def _axis_coeff(
    coords: wp.array(dtype=wp.float32),
    period: float,
    idx: int,
) -> wp.vec3f:
    ### Compute nonuniform periodic central-difference weights at one index.
    n = coords.shape[0]
    im = (idx + n - 1) % n
    ip = (idx + 1) % n

    xi = coords[idx]
    xim = coords[im]
    xip = coords[ip]

    h_minus = xi - xim
    if idx == 0:
        h_minus = xi + period - xim

    h_plus = xip - xi
    if idx == (n - 1):
        h_plus = xip + period - xi

    denom = h_minus + h_plus
    w_minus = -h_plus / (h_minus * denom)
    w_center = (h_plus - h_minus) / (h_minus * h_plus)
    w_plus = h_minus / (h_plus * denom)
    return wp.vec3f(w_minus, w_center, w_plus)


@wp.func
def _axis_second_coeff(
    coords: wp.array(dtype=wp.float32),
    period: float,
    idx: int,
) -> wp.vec3f:
    ### Compute nonuniform periodic second-derivative weights at one index.
    n = coords.shape[0]
    im = (idx + n - 1) % n
    ip = (idx + 1) % n

    xi = coords[idx]
    xim = coords[im]
    xip = coords[ip]

    h_minus = xi - xim
    if idx == 0:
        h_minus = xi + period - xim

    h_plus = xip - xi
    if idx == (n - 1):
        h_plus = xip + period - xi

    denom = h_minus + h_plus
    w_minus = 2.0 / (h_minus * denom)
    w_center = -2.0 / (h_minus * h_plus)
    w_plus = 2.0 / (h_plus * denom)
    return wp.vec3f(w_minus, w_center, w_plus)
