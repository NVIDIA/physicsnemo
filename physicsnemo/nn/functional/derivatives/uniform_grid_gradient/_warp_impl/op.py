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

from ..._request_utils import (
    compose_derivative_outputs,
    normalize_derivative_orders,
    normalize_include_mixed,
    validate_mixed_request,
)
from .launch_backward import _launch_backward, _launch_backward_fused_order2_no_mixed
from .launch_forward import _launch_forward, _launch_forward_fused_order2
from .utils import (
    _normalize_spacing,
    _validate_derivative_order,
    _validate_field,
    _validate_include_mixed,
    _validate_order,
    _warp_launch_context,
)


def _validate_positive_spacing(spacing_tuple: tuple[float, ...]) -> None:
    """Validate that all spacing entries are strictly positive."""
    for dx in spacing_tuple:
        if dx <= 0.0:
            raise ValueError("all spacing entries must be strictly positive")


def _to_fp32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to contiguous float32 when needed."""
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.float32).contiguous()


def _restore_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor back to requested dtype when needed."""
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def _mixed_term_count(ndim: int) -> int:
    """Return mixed derivative term count for 1D/2D/3D."""
    return ndim * (ndim - 1) // 2 if ndim > 1 else 0


@torch.library.custom_op(
    "physicsnemo::uniform_grid_gradient_warp_impl", mutates_args=()
)
def uniform_grid_gradient_impl(
    field: torch.Tensor,
    spacing_meta: torch.Tensor,
    order: int,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Compute periodic first or pure second derivatives on a uniform grid.

    Notes
    -----
    Warp kernels compute in ``float32`` internally. Non-``float32`` inputs are
    cast to ``float32`` for kernel execution and cast back to the original dtype
    on return. Float64 inputs are accepted, but derivative accuracy is limited
    to ``float32`` precision.
    """
    _validate_field(field)
    spacing_tuple = tuple(float(v) for v in spacing_meta.tolist())
    _validate_positive_spacing(spacing_tuple)
    order = _validate_order(int(order))
    derivative_order = _validate_derivative_order(int(derivative_order))
    _validate_include_mixed(
        derivative_order=derivative_order,
        include_mixed=bool(include_mixed),
    )

    orig_dtype = field.dtype
    field_fp32 = _to_fp32_contiguous(field)

    ### Write gradients directly into preallocated output slices to avoid stack copy.
    output_fp32 = torch.empty(
        (field_fp32.ndim, *field_fp32.shape),
        device=field_fp32.device,
        dtype=torch.float32,
    )
    grad_components = [output_fp32[axis] for axis in range(field_fp32.ndim)]

    wp_device, wp_stream = _warp_launch_context(field_fp32)
    _launch_forward(
        field_fp32=field_fp32,
        spacing_tuple=spacing_tuple,
        order=order,
        derivative_order=derivative_order,
        grad_components=grad_components,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    return _restore_dtype(output_fp32, orig_dtype)


@uniform_grid_gradient_impl.register_fake
def _uniform_grid_gradient_impl_fake(
    field: torch.Tensor,
    spacing_meta: torch.Tensor,
    order: int,
    derivative_order: int,
    include_mixed: bool,
) -> torch.Tensor:
    """Fake tensor propagation for uniform-grid custom op."""
    _ = (spacing_meta, order, derivative_order, include_mixed)
    return torch.empty(
        (field.ndim, *field.shape),
        device=field.device,
        dtype=field.dtype,
    )


def setup_uniform_grid_gradient_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for uniform-grid custom-op autograd."""
    field, spacing_meta, order, derivative_order, include_mixed = inputs
    _ = output
    ctx.spacing_tuple = tuple(float(v) for v in spacing_meta.tolist())
    ctx.order = int(order)
    ctx.derivative_order = int(derivative_order)
    ctx.include_mixed = bool(include_mixed)
    ctx.orig_dtype = field.dtype


def backward_uniform_grid_gradient(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None, None, None]:
    """Backward pass for uniform-grid custom op (gradients wrt field only)."""
    if grad_output is None or not ctx.needs_input_grad[0]:
        return None, None, None, None, None

    grad_output_fp32 = _to_fp32_contiguous(grad_output)
    grad_field = torch.empty_like(grad_output_fp32[0])

    wp_device, wp_stream = _warp_launch_context(grad_output_fp32)
    _launch_backward(
        grad_output_fp32=grad_output_fp32,
        spacing_tuple=ctx.spacing_tuple,
        order=ctx.order,
        derivative_order=ctx.derivative_order,
        grad_field=grad_field,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    grad_field = _restore_dtype(grad_field, ctx.orig_dtype)
    return grad_field, None, None, None, None


uniform_grid_gradient_impl.register_autograd(
    backward_uniform_grid_gradient,
    setup_context=setup_uniform_grid_gradient_context,
)


@torch.library.custom_op(
    "physicsnemo::uniform_grid_derivatives_order2_fused_warp_impl",
    mutates_args=(),
)
def uniform_grid_derivatives_order2_fused_impl(
    field: torch.Tensor,
    spacing_meta: torch.Tensor,
    include_mixed: bool,
) -> torch.Tensor:
    """Compute fused order-2 derivatives (first + second + optional mixed).

    Notes
    -----
    Warp kernels compute in ``float32`` internally. Non-``float32`` inputs are
    cast to ``float32`` for kernel execution and cast back to the original dtype
    on return. Float64 inputs are accepted, but derivative accuracy is limited
    to ``float32`` precision.
    """
    _validate_field(field)
    spacing_tuple = tuple(float(v) for v in spacing_meta.tolist())
    _validate_positive_spacing(spacing_tuple)
    if not isinstance(include_mixed, bool):
        raise TypeError(f"include_mixed must be a bool, got {type(include_mixed)}")
    if include_mixed and field.ndim < 2:
        raise ValueError("mixed derivatives require at least 2D inputs")

    orig_dtype = field.dtype
    field_fp32 = _to_fp32_contiguous(field)
    n_dims = field_fp32.ndim
    n_mixed = _mixed_term_count(n_dims) if include_mixed else 0
    output_fp32 = torch.empty(
        (2 * n_dims + n_mixed, *field_fp32.shape),
        device=field_fp32.device,
        dtype=torch.float32,
    )
    first_terms = [output_fp32[axis] for axis in range(n_dims)]
    second_terms = [output_fp32[n_dims + axis] for axis in range(n_dims)]
    mixed_terms = [output_fp32[2 * n_dims + axis] for axis in range(n_mixed)]

    wp_device, wp_stream = _warp_launch_context(field_fp32)
    _launch_forward_fused_order2(
        field_fp32=field_fp32,
        spacing_tuple=spacing_tuple,
        first_components=first_terms,
        second_components=second_terms,
        mixed_components=mixed_terms,
        include_mixed=include_mixed,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    return _restore_dtype(output_fp32, orig_dtype)


@uniform_grid_derivatives_order2_fused_impl.register_fake
def _uniform_grid_derivatives_order2_fused_impl_fake(
    field: torch.Tensor,
    spacing_meta: torch.Tensor,
    include_mixed: bool,
) -> torch.Tensor:
    """Fake tensor propagation for fused uniform order-2 custom op."""
    _ = spacing_meta
    n_mixed = _mixed_term_count(field.ndim) if include_mixed else 0
    return torch.empty(
        (2 * field.ndim + n_mixed, *field.shape),
        device=field.device,
        dtype=field.dtype,
    )


def setup_uniform_grid_derivatives_order2_fused_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    """Store backward context for fused uniform order-2 custom op."""
    field, spacing_meta, include_mixed = inputs
    _ = output
    ctx.spacing_tuple = tuple(float(v) for v in spacing_meta.tolist())
    ctx.orig_dtype = field.dtype
    ctx.n_dims = field.ndim
    ctx.include_mixed = bool(include_mixed)


def backward_uniform_grid_derivatives_order2_fused(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None]:
    """Backward pass for fused uniform order-2 custom op."""
    if grad_output is None or not ctx.needs_input_grad[0]:
        return None, None, None

    grad_output_fp32 = _to_fp32_contiguous(grad_output)
    n_dims = int(ctx.n_dims)
    include_mixed = bool(ctx.include_mixed)
    grad_first_components = [grad_output_fp32[axis] for axis in range(n_dims)]
    grad_second_components = [grad_output_fp32[n_dims + axis] for axis in range(n_dims)]
    grad_mixed_components: list[torch.Tensor] = []
    if include_mixed:
        n_mixed = _mixed_term_count(n_dims)
        grad_mixed_components = [
            grad_output_fp32[2 * n_dims + axis] for axis in range(n_mixed)
        ]
    grad_field = torch.empty_like(grad_output_fp32[0])

    wp_device, wp_stream = _warp_launch_context(grad_output_fp32)
    _launch_backward_fused_order2_no_mixed(
        grad_first_components=grad_first_components,
        grad_second_components=grad_second_components,
        grad_mixed_components=grad_mixed_components,
        spacing_tuple=ctx.spacing_tuple,
        include_mixed=include_mixed,
        grad_field=grad_field,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )
    grad_field = _restore_dtype(grad_field, ctx.orig_dtype)
    return grad_field, None, None


uniform_grid_derivatives_order2_fused_impl.register_autograd(
    backward_uniform_grid_derivatives_order2_fused,
    setup_context=setup_uniform_grid_derivatives_order2_fused_context,
)


def uniform_grid_gradient_warp(
    field: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    order: int = 2,
    derivative_order: int = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute periodic first or pure second derivatives on a uniform grid.

    Notes
    -----
    Warp backends internally compute in ``float32``. Float64 inputs are
    accepted, but derivative accuracy is limited to ``float32`` precision.
    """
    _validate_field(field)
    spacing_tuple = _normalize_spacing(spacing, field.ndim)
    _validate_positive_spacing(spacing_tuple)
    order = _validate_order(order)
    derivative_order = _validate_derivative_order(derivative_order)
    _validate_include_mixed(
        derivative_order=derivative_order,
        include_mixed=include_mixed,
    )
    spacing_meta = torch.tensor(spacing_tuple, dtype=torch.float32, device="cpu")
    return uniform_grid_gradient_impl(
        field,
        spacing_meta,
        int(order),
        int(derivative_order),
        bool(include_mixed),
    )


def uniform_grid_gradient_warp_multi(
    field: torch.Tensor,
    spacing: float | Sequence[float],
    order: int,
    derivative_orders: int | Sequence[int] = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute multiple derivative families, fusing Warp launches when possible.

    For ``order=2``, combined requests use fused kernels (with custom-op
    backward when gradients are required), including optional mixed second
    derivatives. ``order=4`` requests continue to compose single-order Warp calls.
    """
    _validate_field(field)
    spacing_tuple = _normalize_spacing(spacing, field.ndim)
    _validate_positive_spacing(spacing_tuple)
    order = _validate_order(order)
    requested_orders = normalize_derivative_orders(
        derivative_orders=derivative_orders,
        function_name="uniform_grid_gradient",
    )
    mixed_terms = normalize_include_mixed(
        include_mixed=include_mixed,
        function_name="uniform_grid_gradient",
    )
    validate_mixed_request(
        derivative_orders=requested_orders,
        include_mixed=mixed_terms,
        ndim=field.ndim,
        function_name="uniform_grid_gradient",
    )

    spacing_meta = torch.tensor(spacing_tuple, dtype=torch.float32, device="cpu")

    ### Use fused order-2 path whenever multiple derivative families are requested.
    if order == 2 and (len(requested_orders) > 1 or mixed_terms):
        fused = uniform_grid_derivatives_order2_fused_impl(
            field, spacing_meta, mixed_terms
        )
        outputs: list[torch.Tensor] = []
        n_dims = field.ndim
        if 1 in requested_orders:
            outputs.extend(fused[:n_dims].unbind(0))
        if 2 in requested_orders:
            outputs.extend(fused[n_dims : 2 * n_dims].unbind(0))
            if mixed_terms:
                n_mixed = _mixed_term_count(n_dims)
                outputs.extend(fused[2 * n_dims : 2 * n_dims + n_mixed].unbind(0))
        return torch.stack(outputs, dim=0)

    ### Compose requests through the single-order custom op entrypoint.
    return compose_derivative_outputs(
        field=field,
        requested_orders=requested_orders,
        include_mixed=mixed_terms,
        single_order_fn=lambda input_field, derivative_order: (
            uniform_grid_gradient_warp(
                field=input_field,
                spacing=spacing_tuple,
                order=order,
                derivative_order=derivative_order,
                include_mixed=False,
            )
        ),
    )
