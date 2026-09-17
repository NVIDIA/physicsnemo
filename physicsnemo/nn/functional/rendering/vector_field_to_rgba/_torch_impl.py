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

import math

import torch


def _validate_transfer_range(vmin: float, vmax: float) -> None:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")


def _validate_opacity(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in the closed interval [0, 1]")


def _jet_colormap(value: torch.Tensor) -> torch.Tensor:
    red = torch.minimum(4.0 * value - 1.5, -4.0 * value + 4.5).clamp(0.0, 1.0)
    green = torch.minimum(4.0 * value - 0.5, -4.0 * value + 3.5).clamp(0.0, 1.0)
    blue = torch.minimum(4.0 * value + 0.5, -4.0 * value + 2.5).clamp(0.0, 1.0)
    return torch.stack([red, green, blue], dim=-1)


def _rgba_to_uint8(color: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    rgba = torch.cat([color, alpha.unsqueeze(-1)], dim=-1)
    return (rgba * 255.0).clamp(0.0, 255.0).to(torch.uint8)


def vector_field_to_rgba_torch(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    """Map vector magnitude and LIC values to a uint8 RGBA volume."""
    if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
        raise ValueError(
            "vector_field must have shape (nx, ny, nz, 3), got "
            f"{tuple(vector_field.shape)}"
        )
    if lic_field.shape != vector_field.shape[:3]:
        raise ValueError(
            "lic_field must have shape matching vector_field spatial dimensions"
        )
    _validate_transfer_range(vmin, vmax)
    _validate_opacity(max_opacity, name="max_opacity")
    _validate_opacity(lic_threshold, name="lic_threshold")

    vector_fp32 = vector_field.to(torch.float32)
    normalized = ((vector_fp32.norm(dim=-1) - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    color = _jet_colormap(normalized)
    lic_value = lic_field.to(device=vector_field.device, dtype=torch.float32).clamp(
        0.0, 1.0
    )
    lic_value = torch.where(
        lic_value < lic_threshold, torch.zeros_like(lic_value), lic_value
    )
    alpha = (lic_value * normalized * max_opacity).clamp(0.0, 1.0)
    return _rgba_to_uint8(color, alpha)


__all__ = ["vector_field_to_rgba_torch"]
