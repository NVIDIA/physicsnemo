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

"""Shared validation and postprocessing for regular-lattice deformation."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import Literal, get_args

import torch

LatticeInterpolation = Literal["linear", "smooth_step_1", "smooth_step_2"]
_SUPPORTED_INTERPOLATION_TYPES = get_args(LatticeInterpolation)


def _validate_inputs(
    points: torch.Tensor,
    control_displacements: torch.Tensor,
    lattice_bounds: Sequence[tuple[float, float]],
    interpolation_type: str,
    point_weights: torch.Tensor | None,
) -> tuple[
    list[tuple[float, float, int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Validate inputs and return normalized interpolation metadata."""
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points)!r}")
    if not isinstance(control_displacements, torch.Tensor):
        raise TypeError(
            "control_displacements must be a torch.Tensor, "
            f"got {type(control_displacements)!r}"
        )
    if not isinstance(interpolation_type, str):
        raise TypeError(
            f"interpolation_type must be a string, got {type(interpolation_type)!r}"
        )
    if interpolation_type not in _SUPPORTED_INTERPOLATION_TYPES:
        raise ValueError(
            "interpolation_type must be one of "
            f"{list(_SUPPORTED_INTERPOLATION_TYPES)}, got {interpolation_type!r}"
        )
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (num_points, dim), got {points.shape}"
        )
    if not torch.is_floating_point(points):
        raise TypeError(f"points must have a floating-point dtype, got {points.dtype}")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "lattice deformation supports float32 and float64 points, "
            f"got {points.dtype}"
        )

    num_points, dim = points.shape
    if dim < 1 or dim > 3:
        raise ValueError(
            f"lattice deformation supports 1D, 2D, or 3D points, got dim={dim}"
        )

    expected_rank = dim + 1
    if control_displacements.ndim != expected_rank:
        raise ValueError(
            "control_displacements must have shape "
            f"(dim, *lattice_shape), expected rank {expected_rank} for dim={dim}, "
            f"got {control_displacements.shape}"
        )
    if control_displacements.shape[0] != dim:
        raise ValueError(
            "control_displacements must have one channel per coordinate dimension, "
            f"expected {dim}, got {control_displacements.shape[0]}"
        )
    if not torch.is_floating_point(control_displacements):
        raise TypeError(
            "control_displacements must have a floating-point dtype, "
            f"got {control_displacements.dtype}"
        )
    if control_displacements.device != points.device:
        raise ValueError(
            "points and control_displacements must be on the same device, got "
            f"{points.device} and {control_displacements.device}"
        )
    if control_displacements.dtype != points.dtype:
        raise TypeError(
            "points and control_displacements must have the same dtype, got "
            f"{points.dtype} and {control_displacements.dtype}"
        )

    if isinstance(lattice_bounds, (str, bytes)) or not isinstance(
        lattice_bounds, Sequence
    ):
        raise TypeError(
            "lattice_bounds must be a sequence of (lower, upper) pairs, "
            f"got {type(lattice_bounds)!r}"
        )
    if len(lattice_bounds) != dim:
        raise ValueError(
            "lattice_bounds must contain one (lower, upper) pair per dimension; "
            f"expected {dim}, got {len(lattice_bounds)}"
        )

    grid: list[tuple[float, float, int]] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    for axis, bounds in enumerate(lattice_bounds):
        if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
            raise TypeError(
                f"lattice_bounds[{axis}] must be a (lower, upper) pair, "
                f"got {type(bounds)!r}"
            )
        if len(bounds) != 2:
            raise ValueError(
                f"lattice_bounds[{axis}] must be a (lower, upper) pair, got {bounds!r}"
            )
        try:
            lower, upper = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"lattice_bounds[{axis}] must contain numeric lower and upper "
                f"values, got {bounds!r}"
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(
                f"lattice_bounds[{axis}] must be finite, got ({lower}, {upper})"
            )
        if lower >= upper:
            raise ValueError(
                f"lattice_bounds[{axis}] must satisfy lower < upper, "
                f"got ({lower}, {upper})"
            )

        # The delegated interpolation implementations store their lattice
        # metadata in float32. Queries are normalized to [0, 1] below, but the
        # world-space bounds must first remain distinct in the point dtype.
        if points.dtype == torch.float32:
            try:
                lower_in_dtype = struct.unpack("f", struct.pack("f", lower))[0]
                upper_in_dtype = struct.unpack("f", struct.pack("f", upper))[0]
            except OverflowError as exc:
                raise ValueError(
                    f"lattice_bounds[{axis}] cannot be represented in {points.dtype}"
                ) from exc
            if (
                not math.isfinite(lower_in_dtype)
                or not math.isfinite(upper_in_dtype)
                or lower_in_dtype >= upper_in_dtype
            ):
                raise ValueError(
                    f"lattice_bounds[{axis}] are not distinguishable in "
                    f"{points.dtype}: ({lower}, {upper})"
                )
            try:
                span_in_dtype = struct.unpack(
                    "f", struct.pack("f", upper_in_dtype - lower_in_dtype)
                )[0]
            except OverflowError:
                span_in_dtype = math.inf
        else:
            span_in_dtype = upper - lower

        if not math.isfinite(span_in_dtype) or span_in_dtype <= 0:
            raise ValueError(
                f"lattice_bounds[{axis}] must define a finite positive span in "
                f"{points.dtype}, got ({lower}, {upper})"
            )

        resolution = control_displacements.shape[axis + 1]
        if resolution < 2:
            raise ValueError(
                "each lattice dimension must contain at least two control points, "
                f"but axis {axis} has resolution {resolution}"
            )
        lower_values.append(lower)
        upper_values.append(upper)
        grid.append((0.0, 1.0, resolution))

    if point_weights is not None:
        if not isinstance(point_weights, torch.Tensor):
            raise TypeError(
                f"point_weights must be a torch.Tensor or None, got {type(point_weights)!r}"
            )
        if point_weights.shape != (num_points,):
            raise ValueError(
                f"point_weights must have shape ({num_points},), got {point_weights.shape}"
            )
        if point_weights.device != points.device:
            raise ValueError(
                "points and point_weights must be on the same device, got "
                f"{points.device} and {point_weights.device}"
            )
        if point_weights.dtype != torch.bool and point_weights.dtype != points.dtype:
            raise TypeError(
                "point_weights must be bool or have the same dtype as points, got "
                f"{point_weights.dtype} and {points.dtype}"
            )

    lower = points.new_tensor(lower_values)
    upper = points.new_tensor(upper_values)
    return grid, lower, upper, point_weights


def prepare_lattice_deform_inputs(
    points: torch.Tensor,
    control_displacements: torch.Tensor,
    lattice_bounds: Sequence[tuple[float, float]],
    interpolation_type: LatticeInterpolation,
    point_weights: torch.Tensor | None,
) -> tuple[list[tuple[float, float, int]], torch.Tensor, torch.Tensor | None]:
    """Validate inputs and normalize world coordinates to a unit lattice."""
    grid, lower, upper, point_weights = _validate_inputs(
        points,
        control_displacements,
        lattice_bounds,
        interpolation_type,
        point_weights,
    )
    normalized_points = (points - lower) / (upper - lower)
    return grid, normalized_points, point_weights


def empty_lattice_deformation(
    points: torch.Tensor,
    control_displacements: torch.Tensor,
    point_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Return empty points while retaining all differentiable autograd edges."""
    # Reducing an empty view retains an autograd edge without reading unused
    # controls, so non-finite control values cannot contaminate an empty result.
    zero = control_displacements[:0].sum()
    if point_weights is not None and point_weights.is_floating_point():
        zero = zero + point_weights[:0].sum()
    return points.clone() + zero


def apply_lattice_displacement(
    points: torch.Tensor,
    displacement: torch.Tensor,
    point_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Apply optional point weights and add the interpolated displacement."""
    if point_weights is not None:
        if point_weights.dtype == torch.bool:
            displacement = torch.where(
                point_weights.unsqueeze(-1),
                displacement,
                0.0,
            )
        else:
            displacement = displacement * point_weights.unsqueeze(-1)
    return points + displacement


__all__ = [
    "LatticeInterpolation",
    "apply_lattice_displacement",
    "empty_lattice_deformation",
    "prepare_lattice_deform_inputs",
]
