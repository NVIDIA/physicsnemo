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


def scalar_field_to_rgba_torch(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    """Map a scalar volume to a uint8 RGBA transfer-function volume."""
    if field.ndim != 3:
        raise ValueError(
            f"field must have shape (nx, ny, nz), got {tuple(field.shape)}"
        )
    _validate_transfer_range(vmin, vmax)
    _validate_opacity(max_opacity, name="max_opacity")
    _validate_opacity(opacity_threshold, name="opacity_threshold")

    value = ((field.to(torch.float32) - vmin) / (vmax - vmin)).clamp(0.0, 1.0)
    color = _jet_colormap(value)
    alpha = torch.where(value < opacity_threshold, torch.zeros_like(value), value)
    alpha = (alpha * max_opacity).clamp(0.0, 1.0)
    return _rgba_to_uint8(color, alpha)


__all__ = ["scalar_field_to_rgba_torch"]
