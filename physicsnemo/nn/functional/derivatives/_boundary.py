# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import torch

BoundaryMode = Literal["periodic", "one_sided"]


def normalize_boundary(boundary: str, *, function_name: str) -> BoundaryMode:
    """Validate and normalize a grid-gradient boundary mode."""
    if not isinstance(boundary, str):
        raise TypeError(f"{function_name} boundary must be a string")
    if boundary not in ("periodic", "one_sided"):
        raise ValueError(
            f'{function_name} boundary must be "periodic" or "one_sided", '
            f"got {boundary!r}"
        )
    return boundary


def _finite_difference_weights(
    nodes: torch.Tensor,
    evaluation_point: torch.Tensor,
    derivative_order: int,
) -> torch.Tensor:
    """Return finite-difference weights for arbitrary one-dimensional nodes."""
    offsets = nodes - evaluation_point
    powers = torch.arange(nodes.numel(), device=nodes.device, dtype=nodes.dtype)
    factorials = torch.exp(torch.lgamma(powers + 1.0))
    system = offsets.unsqueeze(0).pow(powers.unsqueeze(1)) / factorials.unsqueeze(1)
    rhs = torch.zeros_like(nodes)
    rhs[derivative_order] = 1.0
    return torch.linalg.solve(system, rhs)


def _weighted_boundary_value(
    field: torch.Tensor,
    *,
    axis: int,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a one-sided stencil and retain the differentiated axis."""
    values = torch.index_select(field, dim=axis, index=indices)
    view_shape = [1] * field.ndim
    view_shape[axis] = weights.numel()
    return (values * weights.view(view_shape)).sum(dim=axis, keepdim=True)


def apply_one_sided_boundaries(
    output: torch.Tensor,
    *,
    field: torch.Tensor,
    coordinates: tuple[torch.Tensor, ...],
    derivative_order: int,
    stencil_size: int,
    boundary_width: int = 1,
    function_name: str,
) -> torch.Tensor:
    """Replace periodic boundary rows with arbitrary-grid one-sided stencils."""
    corrected: list[torch.Tensor] = []
    for axis, coordinates_axis in enumerate(coordinates):
        axis_size = field.shape[axis]
        if axis_size < stencil_size:
            raise ValueError(
                f"{function_name} boundary='one_sided' requires at least "
                f"{stencil_size} points along axis {axis}, got {axis_size}"
            )

        weight_dtype = torch.float64 if field.dtype == torch.float64 else torch.float32
        nodes = coordinates_axis.to(device=field.device, dtype=weight_dtype)
        left_indices = torch.arange(stencil_size, device=field.device)
        right_indices = torch.arange(
            axis_size - stencil_size, axis_size, device=field.device
        )
        left_nodes = torch.index_select(nodes, 0, left_indices)
        right_nodes = torch.index_select(nodes, 0, right_indices)
        left_values = []
        right_values = []
        for offset in range(boundary_width):
            left_weights = _finite_difference_weights(
                left_nodes, nodes[offset], derivative_order
            ).to(dtype=field.dtype)
            right_weights = _finite_difference_weights(
                right_nodes,
                nodes[axis_size - boundary_width + offset],
                derivative_order,
            ).to(dtype=field.dtype)
            left_values.append(
                _weighted_boundary_value(
                    field,
                    axis=axis,
                    indices=left_indices,
                    weights=left_weights,
                )
            )
            right_values.append(
                _weighted_boundary_value(
                    field,
                    axis=axis,
                    indices=right_indices,
                    weights=right_weights,
                )
            )

        left = torch.cat(left_values, dim=axis)
        right = torch.cat(right_values, dim=axis)
        interior_slices = [slice(None)] * field.ndim
        interior_slices[axis] = slice(boundary_width, -boundary_width)
        interior = output[axis][tuple(interior_slices)]
        corrected.append(torch.cat((left, interior, right), dim=axis))

    return torch.stack(corrected, dim=0)


def uniform_coordinates(
    field: torch.Tensor, spacing: tuple[float, ...]
) -> tuple[torch.Tensor, ...]:
    """Build per-axis coordinates for uniform-grid boundary stencils."""
    dtype = torch.float64 if field.dtype == torch.float64 else torch.float32
    return tuple(
        torch.arange(size, device=field.device, dtype=dtype) * dx
        for size, dx in zip(field.shape, spacing, strict=True)
    )


__all__ = [
    "BoundaryMode",
    "apply_one_sided_boundaries",
    "normalize_boundary",
    "uniform_coordinates",
]
