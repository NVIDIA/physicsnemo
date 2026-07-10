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
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ..._rectilinear_grid_utils import validate_and_normalize_coordinates
from ..utils import validate_scalar_field
from .launch_backward import _launch_backward
from .launch_forward import _launch_forward

wp.init()
wp.config.quiet = True


def _to_fp32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.float32).contiguous()


def _restore_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def _normalize_geometry(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None,
) -> tuple[tuple[torch.Tensor, ...], tuple[float, ...]]:
    return validate_and_normalize_coordinates(
        field=field,
        coordinates=coordinates,
        periods=periods,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )


def _forward_common(
    field: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
) -> torch.Tensor:
    validate_scalar_field(field)
    orig_dtype = field.dtype
    field_fp32 = _to_fp32_contiguous(field)
    output_fp32 = torch.empty_like(field_fp32, dtype=torch.float32)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    _launch_forward(
        field_fp32=field_fp32,
        coords_tuple=coords_tuple[: field_fp32.ndim],
        period_tuple=period_tuple[: field_fp32.ndim],
        output_fp32=output_fp32,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )
    return _restore_dtype(output_fp32, orig_dtype)


def _setup_context_common(
    ctx: torch.autograd.function.FunctionCtx,
    field: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
) -> None:
    ctx.save_for_backward(*coords_tuple)
    ctx.period_tuple = period_tuple
    ctx.orig_dtype = field.dtype


def _backward_common(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> torch.Tensor | None:
    if grad_output is None or not ctx.needs_input_grad[0]:
        return None
    coords_tuple = tuple(ctx.saved_tensors)
    period_tuple = tuple(float(v) for v in ctx.period_tuple)
    grad_output_fp32 = _to_fp32_contiguous(grad_output)
    grad_field_fp32 = torch.empty_like(grad_output_fp32, dtype=torch.float32)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    _launch_backward(
        grad_output_fp32=grad_output_fp32,
        coords_tuple=coords_tuple,
        period_tuple=period_tuple,
        grad_field_fp32=grad_field_fp32,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )
    return _restore_dtype(grad_field_fp32, ctx.orig_dtype)


@torch.library.custom_op(
    "physicsnemo::rectilinear_grid_laplacian_1d_warp_impl", mutates_args=()
)
def rectilinear_grid_laplacian_1d_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    period0: float,
) -> torch.Tensor:
    """Compute 1D rectilinear-grid Laplacian with a fused Warp custom op."""
    return _forward_common(field, (coord0,), (float(period0),))


@rectilinear_grid_laplacian_1d_impl.register_fake
def _rectilinear_grid_laplacian_1d_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    period0: float,
) -> torch.Tensor:
    _ = (coord0, period0)
    return torch.empty_like(field)


def setup_rectilinear_grid_laplacian_1d_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    """Save 1D Laplacian tensors required by the custom backward pass."""
    field, coord0, period0 = inputs
    _ = output
    _setup_context_common(ctx, field, (coord0,), (float(period0),))


def backward_rectilinear_grid_laplacian_1d(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None]:
    return _backward_common(ctx, grad_output), None, None


rectilinear_grid_laplacian_1d_impl.register_autograd(
    backward_rectilinear_grid_laplacian_1d,
    setup_context=setup_rectilinear_grid_laplacian_1d_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_grid_laplacian_2d_warp_impl", mutates_args=()
)
def rectilinear_grid_laplacian_2d_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    period0: float,
    period1: float,
) -> torch.Tensor:
    """Compute 2D rectilinear-grid Laplacian with a fused Warp custom op."""
    return _forward_common(
        field,
        (coord0, coord1),
        (float(period0), float(period1)),
    )


@rectilinear_grid_laplacian_2d_impl.register_fake
def _rectilinear_grid_laplacian_2d_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    period0: float,
    period1: float,
) -> torch.Tensor:
    _ = (coord0, coord1, period0, period1)
    return torch.empty_like(field)


def setup_rectilinear_grid_laplacian_2d_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    """Save 2D Laplacian tensors required by the custom backward pass."""
    field, coord0, coord1, period0, period1 = inputs
    _ = output
    _setup_context_common(
        ctx,
        field,
        (coord0, coord1),
        (float(period0), float(period1)),
    )


def backward_rectilinear_grid_laplacian_2d(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None]:
    return _backward_common(ctx, grad_output), None, None, None, None


rectilinear_grid_laplacian_2d_impl.register_autograd(
    backward_rectilinear_grid_laplacian_2d,
    setup_context=setup_rectilinear_grid_laplacian_2d_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_grid_laplacian_3d_warp_impl", mutates_args=()
)
def rectilinear_grid_laplacian_3d_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    period0: float,
    period1: float,
    period2: float,
) -> torch.Tensor:
    """Compute 3D rectilinear-grid Laplacian with a fused Warp custom op."""
    return _forward_common(
        field,
        (coord0, coord1, coord2),
        (float(period0), float(period1), float(period2)),
    )


@rectilinear_grid_laplacian_3d_impl.register_fake
def _rectilinear_grid_laplacian_3d_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    period0: float,
    period1: float,
    period2: float,
) -> torch.Tensor:
    _ = (coord0, coord1, coord2, period0, period1, period2)
    return torch.empty_like(field)


def setup_rectilinear_grid_laplacian_3d_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    """Save 3D Laplacian tensors required by the custom backward pass."""
    field, coord0, coord1, coord2, period0, period1, period2 = inputs
    _ = output
    _setup_context_common(
        ctx,
        field,
        (coord0, coord1, coord2),
        (float(period0), float(period1), float(period2)),
    )


def backward_rectilinear_grid_laplacian_3d(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
    return _backward_common(ctx, grad_output), None, None, None, None, None, None


rectilinear_grid_laplacian_3d_impl.register_autograd(
    backward_rectilinear_grid_laplacian_3d,
    setup_context=setup_rectilinear_grid_laplacian_3d_context,
)


def rectilinear_grid_laplacian_warp(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None = None,
) -> torch.Tensor:
    """Compute periodic rectilinear-grid Laplacian with a fused Warp custom op."""
    validate_scalar_field(field)
    coords_tuple, period_tuple = _normalize_geometry(field, coordinates, periods)
    if field.ndim == 1:
        return rectilinear_grid_laplacian_1d_impl(
            field,
            coords_tuple[0],
            period_tuple[0],
        )
    if field.ndim == 2:
        return rectilinear_grid_laplacian_2d_impl(
            field,
            coords_tuple[0],
            coords_tuple[1],
            period_tuple[0],
            period_tuple[1],
        )
    return rectilinear_grid_laplacian_3d_impl(
        field,
        coords_tuple[0],
        coords_tuple[1],
        coords_tuple[2],
        period_tuple[0],
        period_tuple[1],
        period_tuple[2],
    )
