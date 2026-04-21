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


def validate_inputs(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    min_neighbors: int,
) -> None:
    """Validate shared mesh LSQ input contracts for torch and warp backends."""
    ### Validate core tensor shapes and dimensions.
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

    ### Validate all inputs are co-located on the same device.
    if not (
        points.device == values.device
        and points.device == neighbor_offsets.device
        and points.device == neighbor_indices.device
    ):
        raise ValueError(
            "points, values, neighbor_offsets, and neighbor_indices must be on the same device"
        )

    ### Validate floating-point and index dtypes.
    if not torch.is_floating_point(points):
        raise TypeError("points must be floating-point")
    if not torch.is_floating_point(values):
        raise TypeError("values must be floating-point")
    if neighbor_offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("neighbor_offsets must be int32 or int64")
    if neighbor_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("neighbor_indices must be int32 or int64")

    ### Validate CSR range invariants.
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
    """Resolve user-provided or dtype-derived distance floor epsilon."""
    if safe_epsilon is None:
        return float(torch.finfo(dtype).tiny ** 0.25)
    eps = float(safe_epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("safe_epsilon must be a finite positive value")
    return eps
