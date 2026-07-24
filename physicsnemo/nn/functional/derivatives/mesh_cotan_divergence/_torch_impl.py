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
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
) -> None:
    """Validate cotangent divergence inputs for the torch implementation."""
    function_name = "mesh_cotan_divergence"
    if points.ndim != 2:
        raise ValueError(
            f"{function_name}: points must have shape (n_points, dims), "
            f"got {points.shape=}"
        )
    if not torch.is_floating_point(points):
        raise TypeError(f"{function_name}: points must be floating-point")
    if vector_field.shape != points.shape:
        raise ValueError(
            f"{function_name}: vector_field shape must match points shape, "
            f"got {vector_field.shape} and {points.shape}"
        )
    if not torch.is_floating_point(vector_field):
        raise TypeError(f"{function_name}: vector_field must be floating-point")
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
    if dual_volumes.shape[0] != points.shape[0]:
        raise ValueError(
            f"{function_name}: dual_volumes length must match n_points: "
            f"{dual_volumes.shape[0]} != {points.shape[0]}"
        )
    if not torch.is_floating_point(dual_volumes):
        raise TypeError(f"{function_name}: dual_volumes must be floating-point")
    if (
        vector_field.device != points.device
        or edges.device != points.device
        or cotan_weights.device != points.device
        or dual_volumes.device != points.device
    ):
        raise ValueError(
            f"{function_name}: points, vector_field, and geometry must be on same device"
        )
    if edges.numel() > 0:
        # Transfer both bounds together so CUDA validation incurs one host sync.
        idx_min, idx_max = torch.stack(torch.aminmax(edges)).tolist()
        if idx_min < 0 or idx_max >= points.shape[0]:
            raise ValueError(
                f"{function_name}: edges must satisfy "
                f"0 <= index < n_points ({points.shape[0]})"
            )


def _normalize_accumulation(
    accumulation: torch.Tensor,
    dual_volumes: torch.Tensor,
) -> torch.Tensor:
    """Normalize in the output dtype, matching the backend output contract."""
    volumes = dual_volumes.to(
        dtype=accumulation.dtype, device=accumulation.device
    ).clamp(min=_safe_eps(accumulation.dtype))
    return accumulation / volumes


def mesh_cotan_divergence_torch(
    points: torch.Tensor,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    dual_volumes: torch.Tensor,
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute cotangent/DEC mesh divergence with eager PyTorch."""
    _validate_inputs(
        points=points,
        edges=edges,
        cotan_weights=cotan_weights,
        dual_volumes=dual_volumes,
        vector_field=vector_field,
    )

    edge_indices = edges.to(dtype=torch.int64, device=points.device)
    v0 = edge_indices[:, 0]
    v1 = edge_indices[:, 1]

    edge_vectors = points[v1] - points[v0]
    edge_average = 0.5 * (vector_field[v0] + vector_field[v1])
    flat_edge_flux = (edge_average * edge_vectors).sum(dim=-1)
    weighted_flux = cotan_weights.to(dtype=points.dtype) * flat_edge_flux

    divergence = torch.zeros(
        (points.shape[0],),
        dtype=vector_field.dtype,
        device=points.device,
    )
    divergence.index_add_(0, v0, weighted_flux.to(dtype=vector_field.dtype))
    divergence.index_add_(0, v1, -weighted_flux.to(dtype=vector_field.dtype))
    return _normalize_accumulation(divergence, dual_volumes)
