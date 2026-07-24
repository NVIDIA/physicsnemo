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

import torch


def _safe_eps(dtype: torch.dtype) -> float:
    """Return a dtype-aware floor for dual-volume normalization."""
    info = torch.finfo(dtype)
    return min(info.tiny**0.25, info.eps)


def _validate_inputs(
    *,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Validate cotangent Laplacian inputs for the torch implementation."""
    function_name = "mesh_cotan_laplacian"
    if values.ndim < 1:
        raise ValueError(
            f"{function_name}: values must have shape (n_points, ...), "
            f"got {values.shape=}"
        )
    if not torch.is_floating_point(values):
        raise TypeError(f"{function_name}: values must be floating-point")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"{function_name}: edges must have shape (n_edges, 2), got {edges.shape=}"
        )
    if edges.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{function_name}: edges must be int32 or int64")
    if cotan_weights.ndim != 1:
        raise ValueError(
            f"{function_name}: cotan_weights must have shape (n_edges,), "
            f"got {cotan_weights.shape=}"
        )
    if cotan_weights.shape[0] != edges.shape[0]:
        raise ValueError(
            f"{function_name}: cotan_weights length must match edges: "
            f"{cotan_weights.shape[0]} != {edges.shape[0]}"
        )
    if not torch.is_floating_point(cotan_weights):
        raise TypeError(f"{function_name}: cotan_weights must be floating-point")
    if dual_volumes.ndim != 1:
        raise ValueError(
            f"{function_name}: dual_volumes must have shape (n_points,), "
            f"got {dual_volumes.shape=}"
        )
    if dual_volumes.shape[0] != values.shape[0]:
        raise ValueError(
            f"{function_name}: dual_volumes length must match n_points: "
            f"{dual_volumes.shape[0]} != {values.shape[0]}"
        )
    if not torch.is_floating_point(dual_volumes):
        raise TypeError(f"{function_name}: dual_volumes must be floating-point")
    if (
        values.device != edges.device
        or edges.device != cotan_weights.device
        or edges.device != dual_volumes.device
    ):
        raise ValueError(f"{function_name}: values and geometry must be on same device")
    if edges.numel() > 0:
        # Transfer both bounds together so CUDA validation incurs one host sync.
        idx_min, idx_max = torch.stack(torch.aminmax(edges)).tolist()
        if idx_min < 0 or idx_max >= values.shape[0]:
            raise ValueError(
                f"{function_name}: edges must satisfy "
                f"0 <= index < n_points ({values.shape[0]})"
            )


def _normalize_accumulation(
    accumulation: torch.Tensor,
    dual_volumes: torch.Tensor,
) -> torch.Tensor:
    """Normalize in the output dtype, matching the backend output contract."""
    volumes = dual_volumes.to(
        dtype=accumulation.dtype, device=accumulation.device
    ).clamp(min=_safe_eps(accumulation.dtype))
    if accumulation.ndim == 1:
        return accumulation / volumes
    return accumulation / volumes.view(-1, *([1] * (accumulation.ndim - 1)))


def mesh_cotan_laplacian_torch(
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Apply the normalized cotangent Laplacian with eager PyTorch."""
    _validate_inputs(
        edges=edges,
        cotan_weights=cotan_weights,
        dual_volumes=dual_volumes,
        values=values,
    )

    if values.shape[0] == 0:
        return values * 0

    n_points = values.shape[0]
    value_shape = values.shape[1:]
    values_flat = values.reshape(n_points, -1)
    accumulation_flat = torch.zeros_like(values_flat)

    edge_indices = edges.to(dtype=torch.int64, device=values.device)
    v0 = edge_indices[:, 0]
    v1 = edge_indices[:, 1]
    weights = cotan_weights.to(dtype=values.dtype, device=values.device).view(-1, 1)

    delta = values_flat[v1] - values_flat[v0]
    contrib = weights * delta
    accumulation_flat.index_add_(0, v0, contrib)
    accumulation_flat.index_add_(0, v1, -contrib)

    accumulation = accumulation_flat.reshape(n_points, *value_shape)
    return _normalize_accumulation(accumulation, dual_volumes)
