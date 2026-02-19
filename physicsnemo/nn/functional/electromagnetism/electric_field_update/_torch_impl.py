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

from typing import Sequence

import torch

from .utils import (
    _as_spacing_tensor,
    _material_coefficients,
    _normalize_offset,
    _validate_common_inputs,
)


# Place impressed current on the electric grid using offset + clipping.
def _impressed_current_full(
    electric_field: torch.Tensor,
    impressed_current: torch.Tensor,
    impressed_current_offset: tuple[int, int, int],
) -> torch.Tensor:
    full = torch.zeros_like(electric_field)
    ex, ey, ez = electric_field.shape[1:]
    jx, jy, jz = impressed_current.shape[1:]
    ox, oy, oz = impressed_current_offset

    tx0 = max(ox, 0)
    ty0 = max(oy, 0)
    tz0 = max(oz, 0)
    tx1 = min(ox + jx, ex)
    ty1 = min(oy + jy, ey)
    tz1 = min(oz + jz, ez)

    if tx1 <= tx0 or ty1 <= ty0 or tz1 <= tz0:
        return full

    sx0 = max(-ox, 0)
    sy0 = max(-oy, 0)
    sz0 = max(-oz, 0)
    sx1 = sx0 + (tx1 - tx0)
    sy1 = sy0 + (ty1 - ty0)
    sz1 = sz0 + (tz1 - tz0)

    full[:, tx0:tx1, ty0:ty1, tz0:tz1] = impressed_current[:, sx0:sx1, sy0:sy1, sz0:sz1]
    return full


# Apply one FDTD electric-field update step with periodic boundaries.
def electric_field_update_torch(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps: float | torch.Tensor,
    sigma_e: float | torch.Tensor,
    spacing: torch.Tensor | Sequence[float],
    dt: float,
    impressed_current: torch.Tensor | None = None,
    impressed_current_offset: torch.Tensor | Sequence[int] = (0, 0, 0),
    inplace: bool = False,
) -> torch.Tensor:
    _validate_common_inputs(
        electric_field,
        magnetic_field,
        eps,
        sigma_e,
        spacing,
        impressed_current,
        inplace,
    )
    spacing_tensor = _as_spacing_tensor(
        spacing,
        device=electric_field.device,
        dtype=electric_field.dtype,
    )
    offset = _normalize_offset(impressed_current_offset)
    c_ee, c_eh, c_ej = _material_coefficients(
        eps,
        sigma_e,
        spacing_tensor,
        dt,
        spatial_shape=tuple(electric_field.shape[1:]),
        device=electric_field.device,
    )

    hx = magnetic_field[0]
    hy = magnetic_field[1]
    hz = magnetic_field[2]

    m_x_1_0_1 = torch.roll(hx, shifts=(0, 1, 0), dims=(0, 1, 2))
    m_x_1_1_0 = torch.roll(hx, shifts=(0, 0, 1), dims=(0, 1, 2))
    m_y_0_1_1 = torch.roll(hy, shifts=(1, 0, 0), dims=(0, 1, 2))
    m_y_1_1_0 = torch.roll(hy, shifts=(0, 0, 1), dims=(0, 1, 2))
    m_z_0_1_1 = torch.roll(hz, shifts=(1, 0, 0), dims=(0, 1, 2))
    m_z_1_0_1 = torch.roll(hz, shifts=(0, 1, 0), dims=(0, 1, 2))

    curl_h_x = (hz - m_z_1_0_1) - (hy - m_y_1_1_0)
    curl_h_y = (hx - m_x_1_1_0) - (hz - m_z_0_1_1)
    curl_h_z = (hy - m_y_0_1_1) - (hx - m_x_1_0_1)
    curl_h = torch.stack((curl_h_x, curl_h_y, curl_h_z), dim=0)

    if impressed_current is None:
        updated = c_ee * electric_field + c_eh * curl_h
    else:
        j_imp = _impressed_current_full(electric_field, impressed_current, offset)
        updated = c_ee * electric_field + c_eh * curl_h + c_ej * j_imp

    if inplace:
        electric_field.copy_(updated)
        return electric_field

    return updated
