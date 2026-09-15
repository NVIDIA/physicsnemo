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

from .utils import (
    resolve_min_neighbors,
    resolve_safe_epsilon,
    validate_inputs,
    validate_rcond,
    validate_weight_power,
)


def _resolve_compute_dtype(points: torch.Tensor, values: torch.Tensor) -> torch.dtype:
    """Choose a linear-algebra dtype supported by ``torch.linalg.svd``."""
    if points.dtype == torch.float64 or values.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _build_quadratic_design(
    relative_scaled: torch.Tensor,
    triangular: torch.Tensor,
    quadratic_column_scale: torch.Tensor,
) -> torch.Tensor:
    """Build normalized linear and symmetric-quadratic Taylor columns."""
    row, col = triangular
    quadratic = relative_scaled[..., row] * relative_scaled[..., col]
    return torch.cat(
        (relative_scaled, quadratic * quadratic_column_scale),
        dim=-1,
    )


def _build_hessian_basis(
    triangular: torch.Tensor,
    *,
    n_dims: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build basis matrices for packed symmetric Hessian coefficients."""
    n_quadratic = triangular.shape[1]
    basis = torch.zeros(
        (n_quadratic, n_dims, n_dims),
        dtype=dtype,
        device=triangular.device,
    )
    coefficient_index = torch.arange(n_quadratic, device=triangular.device)
    row, col = triangular
    basis[coefficient_index, row, col] = 1.0
    basis[coefficient_index, col, row] = 1.0
    return basis


def _resolve_relative_rcond(
    rcond: float | None,
    *,
    effective_rows: torch.Tensor,
    n_columns: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve per-entity relative singular-value rank thresholds."""
    if rcond is not None:
        return torch.full_like(effective_rows, rcond, dtype=dtype)
    matrix_extent = torch.maximum(
        effective_rows,
        torch.full_like(effective_rows, n_columns),
    )
    return matrix_extent.to(dtype) * torch.finfo(dtype).eps


def _full_column_rank(
    design: torch.Tensor,
    relative_rcond: torch.Tensor,
) -> torch.Tensor:
    """Classify identifiable fits without differentiating through the rank test."""
    with torch.no_grad():
        singular_values = torch.linalg.svdvals(design.detach())
    n_columns = design.shape[-1]
    tolerance = relative_rcond[..., None] * singular_values[..., :1]
    retained = singular_values > tolerance
    return retained.sum(dim=-1) == n_columns


def _solve_full_rank(
    design: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve systems already accepted by the detached SVD rank gate."""
    return torch.linalg.lstsq(
        design,
        right_hand_side,
        driver="gels",
    ).solution


def _coefficients_to_hessian(
    quadratic_coefficients: torch.Tensor,
    hessian_basis: torch.Tensor,
    scale_squared: torch.Tensor,
) -> torch.Tensor:
    """Expand packed symmetric coefficients into full Hessian matrices."""
    hessian_scaled = torch.einsum(
        "bqc,qij->bijc",
        quadratic_coefficients,
        hessian_basis,
    )
    return hessian_scaled / scale_squared[:, None, None, None]


def mesh_lsq_hessian_torch(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int | None = None,
    safe_epsilon: float | None = None,
    rcond: float | None = None,
) -> torch.Tensor:
    """Compute direct quadratic-LSQ Hessians with PyTorch tensor operations."""
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n_entities, dims), got {points.shape=}"
        )
    n_dims = points.shape[1]
    resolved_min_neighbors = resolve_min_neighbors(
        min_neighbors,
        n_dims=n_dims,
    )
    resolved_weight_power = validate_weight_power(weight_power)
    resolved_rcond = validate_rcond(rcond)
    validate_inputs(
        points=points,
        values=values,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        min_neighbors=resolved_min_neighbors,
    )

    points = points.contiguous()
    values = values.contiguous()
    neighbor_offsets = neighbor_offsets.to(
        dtype=torch.int64,
        device=points.device,
    ).contiguous()
    neighbor_indices = neighbor_indices.to(
        dtype=torch.int64,
        device=points.device,
    ).contiguous()

    n_entities = points.shape[0]
    value_shape = values.shape[1:]
    compute_dtype = _resolve_compute_dtype(points, values)
    points_compute = points.to(dtype=compute_dtype)
    n_components = math.prod(value_shape) if value_shape else 1
    values_flat = values.to(dtype=compute_dtype).reshape(n_entities, n_components)
    hessians_flat = torch.zeros(
        (n_entities, n_dims, n_dims, n_components),
        dtype=compute_dtype,
        device=values.device,
    )

    triangular = torch.triu_indices(
        n_dims,
        n_dims,
        device=points.device,
    )
    n_quadratic = triangular.shape[1]
    n_coefficients = n_dims + n_quadratic
    quadratic_column_scale = torch.ones(
        n_quadratic,
        dtype=compute_dtype,
        device=points.device,
    )
    quadratic_column_scale[triangular[0] == triangular[1]] = 0.5
    hessian_basis = _build_hessian_basis(
        triangular,
        n_dims=n_dims,
        dtype=compute_dtype,
    )

    counts = neighbor_offsets[1:] - neighbor_offsets[:-1]
    distance_epsilon = resolve_safe_epsilon(
        safe_epsilon=safe_epsilon,
        dtype=compute_dtype,
    )

    for count_tensor in torch.unique(counts):
        n_neighbors = int(count_tensor.item())
        if n_neighbors < resolved_min_neighbors or n_neighbors == 0:
            continue

        entity_indices = torch.where(counts == count_tensor)[0]
        if entity_indices.numel() == 0:
            continue

        column_offsets = torch.arange(
            n_neighbors,
            device=points.device,
            dtype=torch.int64,
        )
        flat_indices = neighbor_offsets[entity_indices, None] + column_offsets[None]
        neighbors = neighbor_indices[flat_indices]

        relative = points_compute[neighbors] - points_compute[entity_indices].unsqueeze(
            1
        )
        delta_values = values_flat[neighbors] - values_flat[entity_indices].unsqueeze(1)

        distance_squared = relative.square().sum(dim=-1)
        valid_distance = distance_squared > 0.0
        effective_rows = valid_distance.sum(dim=1)
        scale_squared = torch.where(
            valid_distance,
            distance_squared,
            torch.zeros_like(distance_squared),
        ).sum(dim=1) / effective_rows.clamp_min(1).to(compute_dtype)
        scale_squared = torch.where(
            effective_rows > 0,
            scale_squared,
            torch.ones_like(scale_squared),
        )
        relative_scaled = relative * scale_squared.rsqrt()[:, None, None]
        design = _build_quadratic_design(
            relative_scaled,
            triangular,
            quadratic_column_scale,
        )

        normalized_distance_squared = distance_squared / scale_squared[:, None]
        weight_distance_squared = normalized_distance_squared.clamp_min(
            distance_epsilon
        )
        raw_log_sqrt_weight = (
            -0.25 * resolved_weight_power * torch.log(weight_distance_squared)
        )
        masked_log_sqrt_weight = torch.where(
            valid_distance,
            raw_log_sqrt_weight,
            torch.finfo(compute_dtype).min,
        )
        max_log_sqrt_weight = masked_log_sqrt_weight.amax(
            dim=1,
            keepdim=True,
        )
        sqrt_weight = torch.where(
            valid_distance,
            torch.exp(masked_log_sqrt_weight - max_log_sqrt_weight),
            torch.zeros_like(masked_log_sqrt_weight),
        ).unsqueeze(-1)

        weighted_design = sqrt_weight * design
        weighted_delta_values = sqrt_weight * delta_values
        relative_rcond = _resolve_relative_rcond(
            resolved_rcond,
            effective_rows=effective_rows,
            n_columns=n_coefficients,
            dtype=compute_dtype,
        )
        full_rank = _full_column_rank(weighted_design, relative_rcond)
        group_hessians = torch.zeros(
            (entity_indices.shape[0], n_dims, n_dims, n_components),
            dtype=compute_dtype,
            device=values.device,
        )
        solution = _solve_full_rank(
            weighted_design[full_rank],
            weighted_delta_values[full_rank],
        )
        quadratic_coefficients = solution[:, n_dims:]
        group_hessians[full_rank] = _coefficients_to_hessian(
            quadratic_coefficients,
            hessian_basis,
            scale_squared[full_rank],
        )
        hessians_flat[entity_indices] = group_hessians

    # Preserve a zero-gradient autograd path when every local fit is skipped.
    hessians_flat = hessians_flat + (points_compute.sum() + values_flat.sum()) * 0.0
    hessians = hessians_flat.reshape(
        n_entities,
        n_dims,
        n_dims,
        *value_shape,
    )
    if hessians.dtype != values.dtype:
        hessians = hessians.to(dtype=values.dtype)
    return hessians


__all__ = ["mesh_lsq_hessian_torch"]
