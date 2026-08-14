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
import operator

import torch


def validate_inputs(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    min_neighbors: int,
) -> None:
    """Validate mesh LSQ Hessian tensor, device, dtype, and CSR contracts."""
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n_entities, dims), got {points.shape=}"
        )
    if points.shape[1] < 1 or points.shape[1] > 3:
        raise ValueError(f"points must be 1D/2D/3D, got dims={points.shape[1]}")
    if values.ndim < 1:
        raise ValueError(
            f"values must have shape (n_entities, ...), got {values.shape=}"
        )
    if values.shape[0] != points.shape[0]:
        raise ValueError(
            f"values leading dimension must match points: {values.shape[0]} != {points.shape[0]}"
        )
    if neighbor_offsets.ndim != 1:
        raise ValueError("neighbor_offsets must be rank-1")
    if neighbor_offsets.shape[0] != points.shape[0] + 1:
        raise ValueError(
            "neighbor_offsets must have shape (n_entities + 1,), "
            f"got {neighbor_offsets.shape} for n_entities={points.shape[0]}"
        )
    if neighbor_indices.ndim != 1:
        raise ValueError("neighbor_indices must be rank-1")
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be non-negative")

    if not (
        points.device == values.device
        and points.device == neighbor_offsets.device
        and points.device == neighbor_indices.device
    ):
        raise ValueError(
            "points, values, neighbor_offsets, and neighbor_indices must be on the same device"
        )

    if not torch.is_floating_point(points):
        raise TypeError("points must be floating-point")
    if not torch.is_floating_point(values):
        raise TypeError("values must be floating-point")
    if neighbor_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("neighbor_offsets must be int32 or int64")
    if neighbor_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("neighbor_indices must be int32 or int64")

    if int(neighbor_offsets[0].item()) != 0:
        raise ValueError("neighbor_offsets must start at 0")
    if int(neighbor_offsets[-1].item()) != neighbor_indices.shape[0]:
        raise ValueError("neighbor_offsets[-1] must equal len(neighbor_indices)")
    if torch.any(neighbor_offsets[1:] < neighbor_offsets[:-1]):
        raise ValueError("neighbor_offsets must be non-decreasing")

    if neighbor_indices.numel() > 0:
        idx_min = int(neighbor_indices.min().item())
        idx_max = int(neighbor_indices.max().item())
        if idx_min < 0 or idx_max >= points.shape[0]:
            raise ValueError(
                f"neighbor_indices must satisfy 0 <= index < n_entities ({points.shape[0]})"
            )


def resolve_safe_epsilon(*, safe_epsilon: float | None, dtype: torch.dtype) -> float:
    """Resolve the normalized squared-distance floor used by weighting."""
    if safe_epsilon is None:
        return float(torch.finfo(dtype).tiny ** 0.25)
    resolved = float(safe_epsilon)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("safe_epsilon must be a finite positive value")
    return resolved


def quadratic_fit_coefficient_count(n_dims: int) -> int:
    """Return the number of linear and symmetric-quadratic fit coefficients."""
    return n_dims + n_dims * (n_dims + 1) // 2


def resolve_min_neighbors(min_neighbors: int | None, *, n_dims: int) -> int:
    """Resolve the optional neighbor threshold for a quadratic Taylor fit."""
    if min_neighbors is None:
        return quadratic_fit_coefficient_count(n_dims)

    try:
        resolved = operator.index(min_neighbors)
    except TypeError as exc:
        raise TypeError("min_neighbors must be an integer or None") from exc

    if resolved < 0:
        raise ValueError("min_neighbors must be non-negative")
    return resolved


def validate_weight_power(weight_power: float) -> float:
    """Return a finite inverse-distance weighting exponent."""
    resolved = float(weight_power)
    if not math.isfinite(resolved):
        raise ValueError("weight_power must be finite")
    return resolved


def validate_rcond(rcond: float | None) -> float | None:
    """Return a finite, non-negative relative singular-value cutoff."""
    if rcond is None:
        return None
    resolved = float(rcond)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("rcond must be a finite non-negative value or None")
    return resolved


__all__ = [
    "quadratic_fit_coefficient_count",
    "resolve_safe_epsilon",
    "resolve_min_neighbors",
    "validate_inputs",
    "validate_rcond",
    "validate_weight_power",
]
