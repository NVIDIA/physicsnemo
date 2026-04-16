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

from collections.abc import Sequence

import torch

_SUPPORTED_DERIVATIVE_ORDERS = (1, 2)


def validate_field(field: torch.Tensor) -> None:
    """Validate shared rectilinear field input constraints."""
    if field.ndim < 1 or field.ndim > 3:
        raise ValueError(
            f"rectilinear_grid_gradient supports 1D-3D fields, got {field.shape=}"
        )
    if not torch.is_floating_point(field):
        raise TypeError("field must be a floating-point tensor")


def normalize_periods(
    periods: float | Sequence[float] | None,
    coordinates: tuple[torch.Tensor, ...],
) -> tuple[float, ...]:
    """Normalize explicit/inferred periodic lengths to one value per axis."""
    ndim = len(coordinates)
    if periods is None:
        inferred: list[float] = []
        for coords in coordinates:
            if coords.numel() < 2:
                raise ValueError(
                    "rectilinear_grid_gradient requires at least two coordinates per axis"
                )
            inferred.append(
                float((coords[-1] - coords[0] + (coords[1] - coords[0])).item())
            )
        return tuple(inferred)
    if isinstance(periods, (float, int)):
        return tuple(float(periods) for _ in range(ndim))
    periods_tuple = tuple(float(v) for v in periods)
    if len(periods_tuple) != ndim:
        raise ValueError(
            f"periods must have {ndim} entries for a {ndim}D field, got {len(periods_tuple)}"
        )
    return periods_tuple


def validate_and_normalize_coordinates(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None,
    *,
    coordinates_dtype: torch.dtype,
    requires_grad_error: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[float, ...]]:
    """Validate rectilinear coordinates and return normalized coordinates/periods."""
    if len(coordinates) != field.ndim:
        raise ValueError(
            f"coordinates must contain one axis tensor per field dimension ({field.ndim}), "
            f"got {len(coordinates)}"
        )

    normalized_coords: list[torch.Tensor] = []
    for axis, coords in enumerate(coordinates):
        if not isinstance(coords, torch.Tensor):
            raise TypeError(f"coordinates[{axis}] must be a tensor")
        if coords.ndim != 1:
            raise ValueError(
                f"coordinates[{axis}] must be rank-1, got shape={tuple(coords.shape)}"
            )
        if coords.shape[0] != field.shape[axis]:
            raise ValueError(
                f"coordinates[{axis}] length must equal field.shape[{axis}] "
                f"({field.shape[axis]}), got {coords.shape[0]}"
            )
        if coords.requires_grad:
            raise ValueError(requires_grad_error)
        if not torch.is_floating_point(coords):
            raise TypeError(f"coordinates[{axis}] must be floating-point")
        if coords.device != field.device:
            raise ValueError("field and coordinates must be on the same device")
        if coords.numel() < 3:
            raise ValueError(
                "each coordinate axis must contain at least 3 points for central differencing"
            )

        coords_norm = coords.to(dtype=coordinates_dtype).contiguous()
        diffs = coords_norm[1:] - coords_norm[:-1]
        if torch.any(diffs <= 0):
            raise ValueError(f"coordinates[{axis}] must be strictly increasing")
        normalized_coords.append(coords_norm)

    period_tuple = normalize_periods(
        periods=periods, coordinates=tuple(normalized_coords)
    )
    for axis, period in enumerate(period_tuple):
        if period <= 0.0:
            raise ValueError("all periodic lengths must be strictly positive")
        min_period = float(
            (normalized_coords[axis][-1] - normalized_coords[axis][0]).item()
        )
        if period <= min_period:
            raise ValueError(
                f"periods[{axis}] must be larger than coordinate span ({min_period}), got {period}"
            )

    return tuple(normalized_coords), period_tuple


def validate_derivative_request(
    *,
    derivative_order: int,
    include_mixed: bool,
) -> int:
    """Validate derivative-order/mixed-term request for phase-1 behavior."""
    if not isinstance(derivative_order, int):
        raise TypeError(
            f"derivative_order must be an integer, got {type(derivative_order)}"
        )
    if derivative_order not in _SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            "rectilinear_grid_gradient supports derivative_order in [1, 2], "
            f"got derivative_order={derivative_order}"
        )
    if not isinstance(include_mixed, bool):
        raise TypeError(f"include_mixed must be a bool, got {type(include_mixed)}")
    if include_mixed and derivative_order != 2:
        raise ValueError("include_mixed is only valid when derivative_order=2")
    if include_mixed:
        raise NotImplementedError(
            "include_mixed=True is not yet supported; phase-1 supports pure axis-wise "
            "second derivatives only"
        )
    return derivative_order


def axis_central_weights(
    coords: torch.Tensor,
    period: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build periodic nonuniform second-order central-difference weights."""
    diffs = coords[1:] - coords[:-1]

    h_plus = torch.empty_like(coords)
    h_plus[:-1] = diffs
    h_plus[-1] = period - (coords[-1] - coords[0])

    h_minus = torch.empty_like(coords)
    h_minus[1:] = diffs
    h_minus[0] = h_plus[-1]

    if torch.any(h_minus <= 0.0) or torch.any(h_plus <= 0.0):
        raise ValueError(
            "rectilinear coordinates/period produce non-positive periodic spacing"
        )

    denom = h_minus + h_plus
    w_minus = -h_plus / (h_minus * denom)
    w_center = (h_plus - h_minus) / (h_minus * h_plus)
    w_plus = h_minus / (h_plus * denom)
    return w_minus, w_center, w_plus


def axis_second_derivative_weights(
    coords: torch.Tensor,
    period: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build periodic nonuniform second-derivative weights."""
    diffs = coords[1:] - coords[:-1]

    h_plus = torch.empty_like(coords)
    h_plus[:-1] = diffs
    h_plus[-1] = period - (coords[-1] - coords[0])

    h_minus = torch.empty_like(coords)
    h_minus[1:] = diffs
    h_minus[0] = h_plus[-1]

    if torch.any(h_minus <= 0.0) or torch.any(h_plus <= 0.0):
        raise ValueError(
            "rectilinear coordinates/period produce non-positive periodic spacing"
        )

    denom = h_minus + h_plus
    w_minus = 2.0 / (h_minus * denom)
    w_center = -2.0 / (h_minus * h_plus)
    w_plus = 2.0 / (h_plus * denom)
    return w_minus, w_center, w_plus
