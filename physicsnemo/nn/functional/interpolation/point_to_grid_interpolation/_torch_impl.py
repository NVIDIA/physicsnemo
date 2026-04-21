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

import itertools
from typing import List, Tuple

import torch
from torch import Tensor

_INTERP_NEAREST = "nearest_neighbor"
_INTERP_LINEAR = "linear"
_INTERP_SMOOTH_1 = "smooth_step_1"
_INTERP_SMOOTH_2 = "smooth_step_2"
_INTERP_GAUSSIAN = "gaussian"
_VALID_INTERP = {
    _INTERP_NEAREST,
    _INTERP_LINEAR,
    _INTERP_SMOOTH_1,
    _INTERP_SMOOTH_2,
    _INTERP_GAUSSIAN,
}


# Evaluate interpolation basis values for linear/smooth interpolation.
def _basis(interpolation_type: str, x: Tensor) -> Tensor:
    if interpolation_type == _INTERP_LINEAR:
        return x
    if interpolation_type == _INTERP_SMOOTH_1:
        return torch.clamp(3.0 * x**2 - 2.0 * x**3, 0.0, 1.0)
    if interpolation_type == _INTERP_SMOOTH_2:
        return torch.clamp(x**3 * (6.0 * x**2 - 15.0 * x + 10.0), 0.0, 1.0)
    raise RuntimeError(f"Unsupported interpolation_type {interpolation_type}")


# Flatten integer multi-dimensional indices into 1D flat-grid indices.
def _flatten_indices(indices: Tensor, sizes: list[int]) -> Tensor:
    flat = indices[:, 0]
    for dim in range(1, len(sizes)):
        flat = flat * sizes[dim] + indices[:, dim]
    return flat


# Scatter weighted point values into a flattened output grid.
def _scatter_weighted(
    output_flat: Tensor,
    flat_index: Tensor,
    point_values: Tensor,
    weight: Tensor,
) -> None:
    src = (point_values * weight.unsqueeze(-1)).transpose(0, 1).contiguous()
    output_flat.scatter_add_(
        1,
        flat_index.unsqueeze(0).expand(output_flat.shape[0], -1),
        src,
    )


# Validate and normalize point-to-grid interpolation inputs.
def _normalize_inputs(
    query_points: Tensor,
    point_values: Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str,
) -> tuple[Tensor, Tensor, int, list[int], Tensor, Tensor]:
    if interpolation_type not in _VALID_INTERP:
        raise RuntimeError(
            "interpolation_type is not supported; expected one of "
            f"{sorted(_VALID_INTERP)}, got {interpolation_type}"
        )

    dims = len(grid)
    if dims < 1 or dims > 3:
        raise ValueError("point_to_grid_interpolation supports 1-3D grids")

    if query_points.ndim == 1 and dims == 1:
        query_points = query_points.unsqueeze(-1)
    if query_points.ndim != 2 or query_points.shape[-1] != dims:
        raise ValueError(
            f"query_points must have shape (num_points, {dims}), got {tuple(query_points.shape)}"
        )

    if point_values.ndim == 1:
        point_values = point_values.unsqueeze(-1)
    if point_values.ndim != 2:
        raise ValueError(
            "point_values must have shape (num_points, channels) or (num_points,)"
        )

    if query_points.shape[0] != point_values.shape[0]:
        raise ValueError(
            "query_points and point_values must have the same leading dimension"
        )
    if query_points.device != point_values.device:
        raise ValueError("query_points and point_values must be on the same device")
    if query_points.dtype != torch.float32:
        raise TypeError("query_points must be float32")
    if point_values.dtype != torch.float32:
        raise TypeError("point_values must be float32")

    sizes = [int(entry[2]) for entry in grid]
    starts = torch.tensor(
        [float(entry[0]) for entry in grid],
        device=query_points.device,
        dtype=query_points.dtype,
    )
    dx = torch.tensor(
        [(float(entry[1]) - float(entry[0])) / (int(entry[2]) - 1) for entry in grid],
        device=query_points.device,
        dtype=query_points.dtype,
    )
    return query_points, point_values, dims, sizes, starts, dx


# Torch backend implementation for point-to-grid interpolation.
def point_to_grid_interpolation_torch(
    query_points: Tensor,
    point_values: Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str = "smooth_step_2",
    mem_speed_trade: bool = True,
) -> Tensor:
    # Keep API parity with Warp backend.
    _ = mem_speed_trade

    (
        query_points,
        point_values,
        dims,
        sizes,
        starts,
        dx,
    ) = _normalize_inputs(
        query_points=query_points,
        point_values=point_values,
        grid=grid,
        interpolation_type=interpolation_type,
    )

    num_points = query_points.shape[0]
    channels = point_values.shape[1]
    output = torch.zeros(
        (channels, *sizes),
        device=point_values.device,
        dtype=point_values.dtype,
    )
    if num_points == 0:
        return output
    output_flat = output.view(channels, -1)

    pos = (query_points - starts) / dx

    # Nearest-neighbor scatter.
    if interpolation_type == _INTERP_NEAREST:
        center = torch.floor(pos + 0.5).to(torch.int64)
        for dim, size in enumerate(sizes):
            center[:, dim].clamp_(0, size - 1)
        flat = _flatten_indices(center, sizes)
        _scatter_weighted(
            output_flat=output_flat,
            flat_index=flat,
            point_values=point_values,
            weight=torch.ones(
                num_points,
                device=point_values.device,
                dtype=point_values.dtype,
            ),
        )
        return output

    # Linear and smooth-step scatter.
    if interpolation_type in {_INTERP_LINEAR, _INTERP_SMOOTH_1, _INTERP_SMOOTH_2}:
        center = torch.floor(pos).to(torch.int64)
        frac = pos - center.to(pos.dtype)
        lower = _basis(interpolation_type, frac)
        upper = _basis(interpolation_type, 1.0 - frac)

        for bits in itertools.product((0, 1), repeat=dims):
            idx = center.clone()
            weight = torch.ones(
                num_points,
                device=point_values.device,
                dtype=point_values.dtype,
            )
            for dim, bit in enumerate(bits):
                if bit == 0:
                    weight = weight * upper[:, dim]
                else:
                    idx[:, dim] = idx[:, dim] + 1
                    weight = weight * lower[:, dim]
            for dim, size in enumerate(sizes):
                idx[:, dim].clamp_(0, size - 1)
            flat = _flatten_indices(idx, sizes)
            _scatter_weighted(
                output_flat=output_flat,
                flat_index=flat,
                point_values=point_values,
                weight=weight,
            )
        return output

    # Gaussian scatter over a 5-point stencil in each dimension.
    center = torch.floor(pos + 0.5).to(torch.int64)
    sigma = dx / 2.0
    offsets = list(itertools.product(range(-2, 3), repeat=dims))

    gaussian_entries: list[tuple[Tensor, Tensor]] = []
    sum_w = torch.zeros(
        num_points,
        device=point_values.device,
        dtype=point_values.dtype,
    )
    for offset in offsets:
        idx = center.clone()
        for dim, delta in enumerate(offset):
            idx[:, dim] = idx[:, dim] + int(delta)
            idx[:, dim].clamp_(0, sizes[dim] - 1)
        coord = starts + idx.to(pos.dtype) * dx
        dist = (query_points - coord) / sigma
        weight = torch.exp(-0.5 * (dist * dist).sum(dim=1))
        gaussian_entries.append((idx, weight))
        sum_w = sum_w + weight

    inv_sum_w = 1.0 / sum_w
    for idx, weight in gaussian_entries:
        flat = _flatten_indices(idx, sizes)
        _scatter_weighted(
            output_flat=output_flat,
            flat_index=flat,
            point_values=point_values,
            weight=weight * inv_sum_w,
        )
    return output


__all__ = ["point_to_grid_interpolation_torch"]
