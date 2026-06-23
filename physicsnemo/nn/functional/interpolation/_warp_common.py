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

"""Shared Warp interpolation helpers used by both gather and scatter backends."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import warp as wp

_INTERP_NEAREST = 0
_INTERP_LINEAR = 1
_INTERP_SMOOTH_1 = 2
_INTERP_SMOOTH_2 = 3
_INTERP_GAUSSIAN = 4

_INTERP_NAME_TO_ID = {
    "nearest_neighbor": _INTERP_NEAREST,
    "linear": _INTERP_LINEAR,
    "smooth_step_1": _INTERP_SMOOTH_1,
    "smooth_step_2": _INTERP_SMOOTH_2,
    "gaussian": _INTERP_GAUSSIAN,
}

_INTERP_ID_TO_STRIDE = {
    _INTERP_NEAREST: 1,
    _INTERP_LINEAR: 2,
    _INTERP_SMOOTH_1: 2,
    _INTERP_SMOOTH_2: 2,
    _INTERP_GAUSSIAN: 5,
}

wp.config.quiet = True
wp.init()


@wp.func
def smooth_step_1(x: wp.float32) -> wp.float32:
    """Return cubic smooth-step basis value."""
    return wp.clamp(3.0 * x * x - 2.0 * x * x * x, 0.0, 1.0)


@wp.func
def smooth_step_2(x: wp.float32) -> wp.float32:
    """Return quintic smooth-step basis value."""
    return wp.clamp(x * x * x * (6.0 * x * x - 15.0 * x + 10.0), 0.0, 1.0)


@wp.func
def basis_value(interp_id: int, x: wp.float32) -> wp.float32:
    """Evaluate basis function value for supported interpolation modes."""
    if interp_id == _INTERP_SMOOTH_1:
        return smooth_step_1(x)
    if interp_id == _INTERP_SMOOTH_2:
        return smooth_step_2(x)
    return x


@wp.func
def basis_derivative(interp_id: int, x: wp.float32) -> wp.float32:
    """Evaluate basis derivative for supported interpolation modes."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if interp_id == _INTERP_LINEAR:
        return 1.0
    if interp_id == _INTERP_SMOOTH_1:
        return 6.0 * x - 6.0 * x * x
    if interp_id == _INTERP_SMOOTH_2:
        return 30.0 * x * x * (x - 1.0) * (x - 1.0)
    return 0.0


@wp.func
def clamp_index(idx: int, size: int) -> int:
    """Clamp an index into ``[0, size - 1]``."""
    if idx < 0:
        return 0
    if idx >= size:
        return size - 1
    return idx


@wp.func
def clamp_stencil_pair(center: int, size: int) -> wp.vec2i:
    """Return clamped two-point stencil indices ``(center, center + 1)``."""
    return wp.vec2i(clamp_index(center, size), clamp_index(center + 1, size))


def parse_grid_metadata(
    grid_meta: torch.Tensor, *, op_name: str
) -> list[tuple[float, float, int]]:
    """Convert serialized grid metadata into validated Python tuples."""

    if grid_meta.ndim != 2 or grid_meta.shape[1] != 3:
        raise ValueError(
            "grid metadata must have shape (dims, 3) with (min, max, size)"
        )

    grid = [(float(g[0]), float(g[1]), int(g[2])) for g in grid_meta.to("cpu").tolist()]
    dims = len(grid)
    if dims < 1 or dims > 3:
        raise ValueError(f"{op_name} supports 1-3D grids")
    return grid


def interpolation_geometry(
    grid: list[tuple[float, float, int]],
    stride: int,
    *,
    pad_grid: bool,
) -> tuple[list[float], list[float], list[int], float]:
    """Build launch geometry shared by both interpolation operators."""

    k = stride // 2 if pad_grid else 0
    dx_vals = [(g[1] - g[0]) / (g[2] - 1) for g in grid]
    start_vals = [g[0] - k * dx for g, dx in zip(grid, dx_vals)]
    sizes = [g[2] + 2 * k for g in grid]
    center_offset = 0.5 if stride % 2 == 1 else 0.0
    return start_vals, dx_vals, sizes, center_offset


def pad_grid_for_stride(
    context_grid: torch.Tensor, dims: int, stride: int
) -> tuple[torch.Tensor, int]:
    """Pad grid tensors for non-nearest gather kernels."""

    k = stride // 2
    if k == 0:
        return context_grid, 0
    return F.pad(context_grid, dims * (k, k)), k


def crop_padded_grid_gradient(
    grad_padded: torch.Tensor | None,
    k: int,
    grid: list[tuple[float, float, int]],
    dims: int,
) -> torch.Tensor | None:
    """Crop padded gradients back to the original grid contract."""

    if grad_padded is None or k == 0:
        return grad_padded
    if dims == 1:
        return grad_padded[:, k : k + grid[0][2]]
    if dims == 2:
        return grad_padded[:, k : k + grid[0][2], k : k + grid[1][2]]
    return grad_padded[
        :,
        k : k + grid[0][2],
        k : k + grid[1][2],
        k : k + grid[2][2],
    ]


__all__ = [
    "_INTERP_GAUSSIAN",
    "_INTERP_ID_TO_STRIDE",
    "_INTERP_LINEAR",
    "_INTERP_NAME_TO_ID",
    "_INTERP_NEAREST",
    "_INTERP_SMOOTH_1",
    "_INTERP_SMOOTH_2",
    "basis_derivative",
    "basis_value",
    "clamp_index",
    "clamp_stencil_pair",
    "crop_padded_grid_gradient",
    "interpolation_geometry",
    "pad_grid_for_stride",
    "parse_grid_metadata",
]
