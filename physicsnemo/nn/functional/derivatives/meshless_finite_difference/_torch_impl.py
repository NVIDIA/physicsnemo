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
from itertools import combinations, product

import torch


def _normalize_spacing(spacing: float | Sequence[float], dim: int) -> tuple[float, ...]:
    """Normalize meshless finite-difference spacing by spatial dimensionality."""
    if isinstance(spacing, (float, int)):
        spacing_tuple = tuple(float(spacing) for _ in range(dim))
    else:
        spacing_tuple = tuple(float(v) for v in spacing)
        if len(spacing_tuple) != dim:
            raise ValueError(
                f"spacing must have {dim} entries for {dim}D inputs, got {len(spacing_tuple)}"
            )

    for axis, step in enumerate(spacing_tuple):
        if step <= 0.0:
            raise ValueError(f"spacing[{axis}] must be strictly positive")
    return spacing_tuple


def _offset_lattice(dim: int) -> tuple[tuple[int, ...], ...]:
    """Return canonical ``{-1, 0, 1}`` stencil offsets in lexicographic order."""
    return tuple(product((-1, 0, 1), repeat=dim))


def _infer_dim_from_stencil_size(stencil_size: int) -> int:
    """Infer dimensionality from a central stencil size."""
    if stencil_size == 3:
        return 1
    if stencil_size == 9:
        return 2
    if stencil_size == 27:
        return 3
    raise ValueError(
        "stencil_values second dimension must be 3, 9, or 27 "
        f"(for 1D/2D/3D stencils), got {stencil_size}"
    )


def meshless_fd_stencil_points_torch(
    points: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    include_center: bool = True,
) -> torch.Tensor:
    """Build local Cartesian stencil points for meshless finite differences.

    Parameters
    ----------
    points : torch.Tensor
        Query points with shape ``(num_points, dim)`` where ``dim`` is 1, 2, or 3.
    spacing : float | Sequence[float], optional
        Stencil spacing per axis.
    include_center : bool, optional
        Include the center point ``(0, ..., 0)`` in the stencil if ``True``.

    Returns
    -------
    torch.Tensor
        Stencil points with shape ``(num_points, stencil_size, dim)``.
        ``stencil_size`` is ``3**dim`` when ``include_center=True`` and
        ``3**dim - 1`` otherwise.
    """
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (num_points, dim), got {tuple(points.shape)}"
        )
    if not torch.is_floating_point(points):
        raise TypeError("points must be a floating-point tensor")
    dim = points.shape[1]
    if dim < 1 or dim > 3:
        raise ValueError(f"only 1D/2D/3D points are supported, got dim={dim}")

    spacing_tuple = _normalize_spacing(spacing=spacing, dim=dim)
    offsets = torch.tensor(
        _offset_lattice(dim),
        device=points.device,
        dtype=points.dtype,
    )
    if not include_center:
        offsets = offsets[offsets.abs().sum(dim=-1) > 0]

    spacing_tensor = torch.tensor(
        spacing_tuple,
        device=points.device,
        dtype=points.dtype,
    )
    return points.unsqueeze(1) + offsets.unsqueeze(0) * spacing_tensor


def meshless_fd_derivatives_torch(
    stencil_values: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    order: int = 1,
    return_mixed_derivs: bool = False,
) -> torch.Tensor:
    """Compute central finite-difference derivatives from meshless stencil values.

    Parameters
    ----------
    stencil_values : torch.Tensor
        Values evaluated on canonical stencil points with shape
        ``(num_points, stencil_size)`` or ``(num_points, stencil_size, channels)``.
        Stencil ordering must match
        :func:`meshless_fd_stencil_points_torch` with ``include_center=True``.
    spacing : float | Sequence[float], optional
        Spacing per spatial axis.
    order : int, optional
        Derivative order, either ``1`` or ``2``.
    return_mixed_derivs : bool, optional
        Include mixed second derivatives. Valid only with ``order=2`` and
        dimensionality >= 2.

    Returns
    -------
    torch.Tensor
        Stacked derivatives with shape ``(num_derivatives, num_points)`` for
        scalar input or ``(num_derivatives, num_points, channels)`` for vector input.
    """
    if stencil_values.ndim not in (2, 3):
        raise ValueError(
            "stencil_values must have shape (num_points, stencil_size) or "
            "(num_points, stencil_size, channels)"
        )
    if not torch.is_floating_point(stencil_values):
        raise TypeError("stencil_values must be a floating-point tensor")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    dim = _infer_dim_from_stencil_size(stencil_values.shape[1])
    if return_mixed_derivs and order != 2:
        raise ValueError("return_mixed_derivs=True requires order=2")
    if return_mixed_derivs and dim == 1:
        raise ValueError("mixed derivatives require at least 2D stencils")

    spacing_tuple = _normalize_spacing(spacing=spacing, dim=dim)

    squeeze_channels = stencil_values.ndim == 2
    values = stencil_values.unsqueeze(-1) if squeeze_channels else stencil_values
    values_eval = (
        values.to(torch.float32)
        if values.dtype in (torch.float16, torch.bfloat16)
        else values
    )

    offsets = _offset_lattice(dim)
    offset_to_index = {offset: idx for idx, offset in enumerate(offsets)}
    center_index = offset_to_index[(0,) * dim]

    derivatives: list[torch.Tensor] = []
    for axis in range(dim):
        plus = [0] * dim
        minus = [0] * dim
        plus[axis] = 1
        minus[axis] = -1

        plus_index = offset_to_index[tuple(plus)]
        minus_index = offset_to_index[tuple(minus)]
        step = spacing_tuple[axis]

        if order == 1:
            derivatives.append(
                (values_eval[:, plus_index] - values_eval[:, minus_index])
                / (2.0 * step)
            )
        else:
            derivatives.append(
                (
                    values_eval[:, plus_index]
                    - 2.0 * values_eval[:, center_index]
                    + values_eval[:, minus_index]
                )
                / (step * step)
            )

    if order == 2 and return_mixed_derivs:
        for axis_i, axis_j in combinations(range(dim), 2):
            pp = [0] * dim
            pm = [0] * dim
            mp = [0] * dim
            mm = [0] * dim
            pp[axis_i], pp[axis_j] = 1, 1
            pm[axis_i], pm[axis_j] = 1, -1
            mp[axis_i], mp[axis_j] = -1, 1
            mm[axis_i], mm[axis_j] = -1, -1

            pp_index = offset_to_index[tuple(pp)]
            pm_index = offset_to_index[tuple(pm)]
            mp_index = offset_to_index[tuple(mp)]
            mm_index = offset_to_index[tuple(mm)]

            denominator = 4.0 * spacing_tuple[axis_i] * spacing_tuple[axis_j]
            derivatives.append(
                (
                    values_eval[:, pp_index]
                    - values_eval[:, pm_index]
                    - values_eval[:, mp_index]
                    + values_eval[:, mm_index]
                )
                / denominator
            )

    output = torch.stack(derivatives, dim=0)
    if values_eval.dtype != values.dtype:
        output = output.to(values.dtype)
    if squeeze_channels:
        return output.squeeze(-1)
    return output
