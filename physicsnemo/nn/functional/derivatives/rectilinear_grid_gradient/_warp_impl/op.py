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

from ..._request_utils import (
    compose_derivative_outputs,
    normalize_derivative_orders,
    normalize_include_mixed,
    validate_mixed_request,
)
from .._torch_impl import rectilinear_grid_gradient_torch
from ..utils import (
    validate_and_normalize_coordinates,
    validate_derivative_request,
    validate_field,
)
from .launch_backward import _launch_backward, _launch_backward_fused_no_mixed
from .launch_forward import _launch_forward, _launch_forward_fused_no_mixed

### Warp runtime initialization for custom kernels.
wp.init()
wp.config.quiet = True


def _rectilinear_forward_common(
    field: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Run rectilinear forward kernels and restore the caller dtype."""
    validate_field(field)
    derivative_order = validate_derivative_request(
        derivative_order=derivative_order,
        include_mixed=include_mixed,
    )
    coords_tuple, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coords_tuple,
        periods=period_tuple,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )

    orig_dtype = field.dtype
    field_fp32 = field.to(dtype=torch.float32).contiguous()
    grad_components = [torch.empty_like(field_fp32) for _ in range(field_fp32.ndim)]

    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    _launch_forward(
        field_fp32=field_fp32,
        coords_tuple=coords_tuple,
        period_tuple=period_tuple,
        derivative_order=derivative_order,
        grad_components=grad_components,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    output = torch.stack(grad_components, dim=0)
    if output.dtype != orig_dtype:
        output = output.to(dtype=orig_dtype)
    return output


def _rectilinear_setup_common(
    ctx: torch.autograd.function.FunctionCtx,
    field: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    derivative_order: int,
    include_mixed: bool,
) -> None:
    """Store normalized geometry metadata for rectilinear custom-op backward."""
    derivative_order = validate_derivative_request(
        derivative_order=derivative_order,
        include_mixed=include_mixed,
    )
    _, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coords_tuple,
        periods=period_tuple,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )
    ctx.save_for_backward(
        *[coord.to(dtype=torch.float32).contiguous() for coord in coords_tuple]
    )
    ctx.period_tuple = period_tuple
    ctx.derivative_order = derivative_order
    ctx.orig_dtype = field.dtype


def _rectilinear_backward_common(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> torch.Tensor | None:
    """Evaluate the rectilinear backward kernels for one custom-op invocation."""
    if grad_output is None or not ctx.needs_input_grad[0]:
        return None

    coords_tuple = tuple(ctx.saved_tensors)
    period_tuple = tuple(float(v) for v in ctx.period_tuple)
    grad_output_fp32 = grad_output.to(dtype=torch.float32).contiguous()
    derivative_order = int(ctx.derivative_order)

    ### CUDA 1D second-derivative VJP is routed through torch autograd for numerical stability.
    if (
        derivative_order == 2
        and grad_output_fp32.device.type == "cuda"
        and grad_output_fp32.shape[0] == 1
    ):
        with torch.enable_grad():
            probe = torch.zeros_like(grad_output_fp32[0], requires_grad=True)
            probe_out = rectilinear_grid_gradient_torch(
                field=probe,
                coordinates=coords_tuple,
                periods=period_tuple,
                derivative_order=2,
                include_mixed=False,
            )
            grad_field = torch.autograd.grad(
                outputs=probe_out,
                inputs=probe,
                grad_outputs=grad_output_fp32,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
        if grad_field.dtype != ctx.orig_dtype:
            grad_field = grad_field.to(dtype=ctx.orig_dtype)
        return grad_field

    grad_field = torch.empty_like(grad_output_fp32[0])
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    _launch_backward(
        grad_output_fp32=grad_output_fp32,
        coords_tuple=coords_tuple,
        period_tuple=period_tuple,
        derivative_order=derivative_order,
        grad_field=grad_field,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )
    if grad_field.dtype != ctx.orig_dtype:
        grad_field = grad_field.to(dtype=ctx.orig_dtype)
    return grad_field


def _rectilinear_forward_fused_no_mixed_common(
    field: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
) -> torch.Tensor:
    """Run fused first+second forward kernels and return stacked outputs."""
    validate_field(field)
    coords_tuple, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coords_tuple,
        periods=period_tuple,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )

    orig_dtype = field.dtype
    field_fp32 = field.to(dtype=torch.float32).contiguous()
    first_components = [torch.empty_like(field_fp32) for _ in range(field_fp32.ndim)]
    second_components = [torch.empty_like(field_fp32) for _ in range(field_fp32.ndim)]

    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    _launch_forward_fused_no_mixed(
        field_fp32=field_fp32,
        coords_tuple=coords_tuple,
        period_tuple=period_tuple,
        first_components=first_components,
        second_components=second_components,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    output = torch.stack([*first_components, *second_components], dim=0)
    if output.dtype != orig_dtype:
        output = output.to(dtype=orig_dtype)
    return output


def _rectilinear_setup_fused_no_mixed_common(
    ctx: torch.autograd.function.FunctionCtx,
    field: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
) -> None:
    """Store metadata for fused first+second rectilinear custom-op backward."""
    _, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coords_tuple,
        periods=period_tuple,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )
    ctx.save_for_backward(
        *[coord.to(dtype=torch.float32).contiguous() for coord in coords_tuple]
    )
    ctx.period_tuple = period_tuple
    ctx.orig_dtype = field.dtype
    ctx.n_dims = field.ndim


def _rectilinear_backward_fused_no_mixed_common(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> torch.Tensor | None:
    """Backward for fused first+second custom ops (field gradients only)."""
    if grad_output is None or not ctx.needs_input_grad[0]:
        return None

    coords_tuple = tuple(ctx.saved_tensors)
    period_tuple = tuple(float(v) for v in ctx.period_tuple)
    n_dims = int(ctx.n_dims)
    grad_output_fp32 = grad_output.to(dtype=torch.float32).contiguous()

    ### CUDA 1D fused VJP reuses torch autograd for the 2nd-derivative contribution.
    if n_dims == 1 and grad_output_fp32.device.type == "cuda":
        with torch.enable_grad():
            probe = torch.zeros_like(grad_output_fp32[0], requires_grad=True)
            probe_first = rectilinear_grid_gradient_torch(
                field=probe,
                coordinates=coords_tuple,
                periods=period_tuple,
                derivative_order=1,
                include_mixed=False,
            )
            probe_second = rectilinear_grid_gradient_torch(
                field=probe,
                coordinates=coords_tuple,
                periods=period_tuple,
                derivative_order=2,
                include_mixed=False,
            )
            probe_out = torch.cat((probe_first, probe_second), dim=0)
            grad_field = torch.autograd.grad(
                outputs=probe_out,
                inputs=probe,
                grad_outputs=grad_output_fp32,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
        if grad_field.dtype != ctx.orig_dtype:
            grad_field = grad_field.to(dtype=ctx.orig_dtype)
        return grad_field

    grad_first_components = [grad_output_fp32[i] for i in range(n_dims)]
    grad_second_components = [grad_output_fp32[n_dims + i] for i in range(n_dims)]

    grad_field = torch.empty_like(grad_output_fp32[0])
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    _launch_backward_fused_no_mixed(
        grad_first_components=grad_first_components,
        grad_second_components=grad_second_components,
        coords_tuple=coords_tuple,
        period_tuple=period_tuple,
        grad_field=grad_field,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    if grad_field.dtype != ctx.orig_dtype:
        grad_field = grad_field.to(dtype=ctx.orig_dtype)
    return grad_field


@torch.library.custom_op(
    "physicsnemo::rectilinear_grid_gradient_1d_warp_impl", mutates_args=()
)
def rectilinear_grid_gradient_1d_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    period0: float,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Compute periodic 1D first or pure second derivatives with Warp kernels."""
    return _rectilinear_forward_common(
        field=field,
        coords_tuple=(coord0,),
        period_tuple=(float(period0),),
        derivative_order=int(derivative_order),
        include_mixed=bool(include_mixed),
    )


@rectilinear_grid_gradient_1d_impl.register_fake
def _rectilinear_grid_gradient_1d_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    period0: float,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Fake tensor propagation for 1D rectilinear custom op."""
    _ = (coord0, period0, derivative_order, include_mixed)
    return torch.empty((1, *field.shape), device=field.device, dtype=field.dtype)


def setup_rectilinear_grid_gradient_1d_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for 1D rectilinear custom op."""
    field, coord0, period0, derivative_order, include_mixed = inputs
    _ = output
    _rectilinear_setup_common(
        ctx=ctx,
        field=field,
        coords_tuple=(coord0,),
        period_tuple=(float(period0),),
        derivative_order=int(derivative_order),
        include_mixed=bool(include_mixed),
    )


def backward_rectilinear_grid_gradient_1d(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None]:
    """Backward pass for 1D rectilinear custom op (field gradients only)."""
    grad_field = _rectilinear_backward_common(ctx, grad_output)
    return grad_field, None, None, None, None


rectilinear_grid_gradient_1d_impl.register_autograd(
    backward_rectilinear_grid_gradient_1d,
    setup_context=setup_rectilinear_grid_gradient_1d_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_grid_gradient_2d_warp_impl", mutates_args=()
)
def rectilinear_grid_gradient_2d_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    period0: float,
    period1: float,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Compute periodic 2D first or pure second derivatives with Warp kernels."""
    return _rectilinear_forward_common(
        field=field,
        coords_tuple=(coord0, coord1),
        period_tuple=(float(period0), float(period1)),
        derivative_order=int(derivative_order),
        include_mixed=bool(include_mixed),
    )


@rectilinear_grid_gradient_2d_impl.register_fake
def _rectilinear_grid_gradient_2d_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    period0: float,
    period1: float,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Fake tensor propagation for 2D rectilinear custom op."""
    _ = (coord0, coord1, period0, period1, derivative_order, include_mixed)
    return torch.empty((2, *field.shape), device=field.device, dtype=field.dtype)


def setup_rectilinear_grid_gradient_2d_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for 2D rectilinear custom op."""
    field, coord0, coord1, period0, period1, derivative_order, include_mixed = inputs
    _ = output
    _rectilinear_setup_common(
        ctx=ctx,
        field=field,
        coords_tuple=(coord0, coord1),
        period_tuple=(float(period0), float(period1)),
        derivative_order=int(derivative_order),
        include_mixed=bool(include_mixed),
    )


def backward_rectilinear_grid_gradient_2d(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
    """Backward pass for 2D rectilinear custom op (field gradients only)."""
    grad_field = _rectilinear_backward_common(ctx, grad_output)
    return grad_field, None, None, None, None, None, None


rectilinear_grid_gradient_2d_impl.register_autograd(
    backward_rectilinear_grid_gradient_2d,
    setup_context=setup_rectilinear_grid_gradient_2d_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_grid_gradient_3d_warp_impl", mutates_args=()
)
def rectilinear_grid_gradient_3d_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    period0: float,
    period1: float,
    period2: float,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Compute periodic 3D first or pure second derivatives with Warp kernels."""
    return _rectilinear_forward_common(
        field=field,
        coords_tuple=(coord0, coord1, coord2),
        period_tuple=(float(period0), float(period1), float(period2)),
        derivative_order=int(derivative_order),
        include_mixed=bool(include_mixed),
    )


@rectilinear_grid_gradient_3d_impl.register_fake
def _rectilinear_grid_gradient_3d_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    period0: float,
    period1: float,
    period2: float,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Fake tensor propagation for 3D rectilinear custom op."""
    _ = (
        coord0,
        coord1,
        coord2,
        period0,
        period1,
        period2,
        derivative_order,
        include_mixed,
    )
    return torch.empty((3, *field.shape), device=field.device, dtype=field.dtype)


def setup_rectilinear_grid_gradient_3d_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for 3D rectilinear custom op."""
    (
        field,
        coord0,
        coord1,
        coord2,
        period0,
        period1,
        period2,
        derivative_order,
        include_mixed,
    ) = inputs
    _ = output
    _rectilinear_setup_common(
        ctx=ctx,
        field=field,
        coords_tuple=(coord0, coord1, coord2),
        period_tuple=(float(period0), float(period1), float(period2)),
        derivative_order=int(derivative_order),
        include_mixed=bool(include_mixed),
    )


def backward_rectilinear_grid_gradient_3d(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None, None]:
    """Backward pass for 3D rectilinear custom op (field gradients only)."""
    grad_field = _rectilinear_backward_common(ctx, grad_output)
    return grad_field, None, None, None, None, None, None, None, None


rectilinear_grid_gradient_3d_impl.register_autograd(
    backward_rectilinear_grid_gradient_3d,
    setup_context=setup_rectilinear_grid_gradient_3d_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_derivatives_1d_fused_no_mixed_warp_impl",
    mutates_args=(),
)
def rectilinear_derivatives_1d_fused_no_mixed_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    period0: float,
) -> torch.Tensor:
    """Compute fused 1D first+second derivatives with one Warp launch."""
    return _rectilinear_forward_fused_no_mixed_common(
        field=field,
        coords_tuple=(coord0,),
        period_tuple=(float(period0),),
    )


@rectilinear_derivatives_1d_fused_no_mixed_impl.register_fake
def _rectilinear_derivatives_1d_fused_no_mixed_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    period0: float,
) -> torch.Tensor:
    """Fake tensor propagation for fused 1D rectilinear custom op."""
    _ = (coord0, period0)
    return torch.empty((2, *field.shape), device=field.device, dtype=field.dtype)


def setup_rectilinear_derivatives_1d_fused_no_mixed_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for fused 1D rectilinear custom op."""
    field, coord0, period0 = inputs
    _ = output
    _rectilinear_setup_fused_no_mixed_common(
        ctx=ctx,
        field=field,
        coords_tuple=(coord0,),
        period_tuple=(float(period0),),
    )


def backward_rectilinear_derivatives_1d_fused_no_mixed(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None]:
    """Backward pass for fused 1D rectilinear custom op."""
    grad_field = _rectilinear_backward_fused_no_mixed_common(ctx, grad_output)
    return grad_field, None, None


rectilinear_derivatives_1d_fused_no_mixed_impl.register_autograd(
    backward_rectilinear_derivatives_1d_fused_no_mixed,
    setup_context=setup_rectilinear_derivatives_1d_fused_no_mixed_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_derivatives_2d_fused_no_mixed_warp_impl",
    mutates_args=(),
)
def rectilinear_derivatives_2d_fused_no_mixed_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    period0: float,
    period1: float,
) -> torch.Tensor:
    """Compute fused 2D first+second derivatives with one Warp launch."""
    return _rectilinear_forward_fused_no_mixed_common(
        field=field,
        coords_tuple=(coord0, coord1),
        period_tuple=(float(period0), float(period1)),
    )


@rectilinear_derivatives_2d_fused_no_mixed_impl.register_fake
def _rectilinear_derivatives_2d_fused_no_mixed_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    period0: float,
    period1: float,
) -> torch.Tensor:
    """Fake tensor propagation for fused 2D rectilinear custom op."""
    _ = (coord0, coord1, period0, period1)
    return torch.empty((4, *field.shape), device=field.device, dtype=field.dtype)


def setup_rectilinear_derivatives_2d_fused_no_mixed_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for fused 2D rectilinear custom op."""
    field, coord0, coord1, period0, period1 = inputs
    _ = output
    _rectilinear_setup_fused_no_mixed_common(
        ctx=ctx,
        field=field,
        coords_tuple=(coord0, coord1),
        period_tuple=(float(period0), float(period1)),
    )


def backward_rectilinear_derivatives_2d_fused_no_mixed(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None]:
    """Backward pass for fused 2D rectilinear custom op."""
    grad_field = _rectilinear_backward_fused_no_mixed_common(ctx, grad_output)
    return grad_field, None, None, None, None


rectilinear_derivatives_2d_fused_no_mixed_impl.register_autograd(
    backward_rectilinear_derivatives_2d_fused_no_mixed,
    setup_context=setup_rectilinear_derivatives_2d_fused_no_mixed_context,
)


@torch.library.custom_op(
    "physicsnemo::rectilinear_derivatives_3d_fused_no_mixed_warp_impl",
    mutates_args=(),
)
def rectilinear_derivatives_3d_fused_no_mixed_impl(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    period0: float,
    period1: float,
    period2: float,
) -> torch.Tensor:
    """Compute fused 3D first+second derivatives with one Warp launch."""
    return _rectilinear_forward_fused_no_mixed_common(
        field=field,
        coords_tuple=(coord0, coord1, coord2),
        period_tuple=(float(period0), float(period1), float(period2)),
    )


@rectilinear_derivatives_3d_fused_no_mixed_impl.register_fake
def _rectilinear_derivatives_3d_fused_no_mixed_impl_fake(
    field: torch.Tensor,
    coord0: torch.Tensor,
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    period0: float,
    period1: float,
    period2: float,
) -> torch.Tensor:
    """Fake tensor propagation for fused 3D rectilinear custom op."""
    _ = (coord0, coord1, coord2, period0, period1, period2)
    return torch.empty((6, *field.shape), device=field.device, dtype=field.dtype)


def setup_rectilinear_derivatives_3d_fused_no_mixed_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for fused 3D rectilinear custom op."""
    field, coord0, coord1, coord2, period0, period1, period2 = inputs
    _ = output
    _rectilinear_setup_fused_no_mixed_common(
        ctx=ctx,
        field=field,
        coords_tuple=(coord0, coord1, coord2),
        period_tuple=(float(period0), float(period1), float(period2)),
    )


def backward_rectilinear_derivatives_3d_fused_no_mixed(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
    """Backward pass for fused 3D rectilinear custom op."""
    grad_field = _rectilinear_backward_fused_no_mixed_common(ctx, grad_output)
    return grad_field, None, None, None, None, None, None


rectilinear_derivatives_3d_fused_no_mixed_impl.register_autograd(
    backward_rectilinear_derivatives_3d_fused_no_mixed,
    setup_context=setup_rectilinear_derivatives_3d_fused_no_mixed_context,
)


def rectilinear_grid_gradient_warp(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None = None,
    derivative_order: int = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute periodic first or pure second derivatives on rectilinear grids.

    Notes
    -----
    Warp backends internally compute in ``float32``. Float64 inputs are
    accepted, but derivative accuracy is limited to ``float32`` precision.
    """
    ### Validate field shape/dtype and normalize coordinates.
    validate_field(field)
    derivative_order = validate_derivative_request(
        derivative_order=derivative_order,
        include_mixed=include_mixed,
    )

    coords_tuple, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coordinates,
        periods=periods,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )

    if field.ndim == 1:
        return rectilinear_grid_gradient_1d_impl(
            field,
            coords_tuple[0],
            float(period_tuple[0]),
            int(derivative_order),
            bool(include_mixed),
        )
    if field.ndim == 2:
        return rectilinear_grid_gradient_2d_impl(
            field,
            coords_tuple[0],
            coords_tuple[1],
            float(period_tuple[0]),
            float(period_tuple[1]),
            int(derivative_order),
            bool(include_mixed),
        )
    return rectilinear_grid_gradient_3d_impl(
        field,
        coords_tuple[0],
        coords_tuple[1],
        coords_tuple[2],
        float(period_tuple[0]),
        float(period_tuple[1]),
        float(period_tuple[2]),
        int(derivative_order),
        bool(include_mixed),
    )


def rectilinear_grid_gradient_warp_multi(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None = None,
    derivative_orders: int | Sequence[int] = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute multiple derivative families, fusing first+second when possible.

    For ``derivative_orders=(1, 2)`` with ``include_mixed=False``, this uses a
    fused custom-op path (single fused forward launch + fused backward kernels).
    Mixed requests are composed from single-order custom ops to preserve output
    ordering and autograd behavior.
    """
    validate_field(field)
    coords_tuple, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coordinates,
        periods=periods,
        coordinates_dtype=torch.float32,
        requires_grad_error="coordinate gradients are not supported in warp backend",
    )
    requested_orders = normalize_derivative_orders(
        derivative_orders=derivative_orders,
        function_name="rectilinear_grid_gradient",
    )
    mixed_terms = normalize_include_mixed(
        include_mixed=include_mixed,
        function_name="rectilinear_grid_gradient",
    )
    validate_mixed_request(
        derivative_orders=requested_orders,
        include_mixed=mixed_terms,
        ndim=field.ndim,
        function_name="rectilinear_grid_gradient",
    )

    ### Fused no-mixed path with custom-op backward for combined first+second.
    if not mixed_terms and requested_orders == (1, 2):
        if field.ndim == 1:
            fused = rectilinear_derivatives_1d_fused_no_mixed_impl(
                field,
                coords_tuple[0],
                float(period_tuple[0]),
            )
        elif field.ndim == 2:
            fused = rectilinear_derivatives_2d_fused_no_mixed_impl(
                field,
                coords_tuple[0],
                coords_tuple[1],
                float(period_tuple[0]),
                float(period_tuple[1]),
            )
        else:
            fused = rectilinear_derivatives_3d_fused_no_mixed_impl(
                field,
                coords_tuple[0],
                coords_tuple[1],
                coords_tuple[2],
                float(period_tuple[0]),
                float(period_tuple[1]),
                float(period_tuple[2]),
            )

        outputs: list[torch.Tensor] = []
        n_dims = field.ndim
        if 1 in requested_orders:
            outputs.extend(fused[:n_dims].unbind(0))
        if 2 in requested_orders:
            outputs.extend(fused[n_dims:].unbind(0))
        return torch.stack(outputs, dim=0)

    ### Compose through the single-order custom op path for remaining requests.
    return compose_derivative_outputs(
        field=field,
        requested_orders=requested_orders,
        include_mixed=mixed_terms,
        single_order_fn=lambda input_field, derivative_order: (
            rectilinear_grid_gradient_warp(
                field=input_field,
                coordinates=coords_tuple,
                periods=period_tuple,
                derivative_order=derivative_order,
                include_mixed=False,
            )
        ),
    )
