# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch custom-op integration for Warp-backed radial-basis fields."""

from __future__ import annotations

from typing import NamedTuple

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .rbf_kernels import (
    rbf_control_and_radial_backward_f32,
    rbf_control_and_radial_backward_f64,
    rbf_forward_f32,
    rbf_forward_f64,
    rbf_point_backward_f32,
    rbf_point_backward_f64,
    rbf_polynomial_backward_f32,
    rbf_polynomial_backward_f64,
)


class _RBFKernelSet(NamedTuple):
    """Warp dtype and matching thin-plate-spline kernels."""

    warp_dtype: object
    forward: object
    point_backward: object
    control_and_radial_backward: object
    polynomial_backward: object


_RBF_KERNELS = {
    torch.float32: _RBFKernelSet(
        wp.float32,
        rbf_forward_f32,
        rbf_point_backward_f32,
        rbf_control_and_radial_backward_f32,
        rbf_polynomial_backward_f32,
    ),
    torch.float64: _RBFKernelSet(
        wp.float64,
        rbf_forward_f64,
        rbf_point_backward_f64,
        rbf_control_and_radial_backward_f64,
        rbf_polynomial_backward_f64,
    ),
}

# Each control-/polynomial-centric pullback thread reduces one bounded query
# block before issuing a single atomic update. This keeps temporary storage
# constant while providing enough parallel work for large point sets.
_RBF_BACKWARD_QUERY_BLOCK_SIZE = 256


def _rbf_kernels(dtype: torch.dtype) -> _RBFKernelSet:
    """Return one internally consistent dtype/kernel family."""

    try:
        return _RBF_KERNELS[dtype]
    except KeyError:
        raise TypeError(
            f"Warp RBF evaluation supports float32 and float64, got {dtype}"
        ) from None


def _wp_view(tensor: torch.Tensor, dtype):
    """Create a zero-copy Warp descriptor for a detached Torch tensor."""

    return wp.from_torch(
        tensor.detach(), dtype=dtype, return_ctype=True, requires_grad=False
    )


def _empty_3d(reference: torch.Tensor) -> torch.Tensor:
    """Return a rank-3 zero-size launch placeholder."""

    return torch.empty((0, 0, 0), dtype=reference.dtype, device=reference.device)


def _validate_rbf_inputs(
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> None:
    """Validate normalized RBF evaluator inputs without device synchronization."""

    tensors = (points, controls, radial_coefficients, polynomial_coefficients)
    dtype = points.dtype
    device = points.device
    _rbf_kernels(dtype)
    if any(tensor.dtype != dtype for tensor in tensors[1:]):
        raise TypeError("all RBF field tensors must have the same dtype")
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError("all RBF field tensors must be on the same device")
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("all RBF field tensors must be normalized rank-3 tensors")
    if points.shape[-1] == 0:
        raise ValueError("RBF field tensors must have at least one spatial dimension")
    if controls.shape != radial_coefficients.shape:
        raise ValueError("radial_coefficients must have the same shape as controls")
    if controls.shape[0] != points.shape[0] or controls.shape[2] != points.shape[2]:
        raise ValueError(
            "points and controls must have aligned batch/spatial dimensions"
        )
    batch, _, num_dims = points.shape
    if polynomial_coefficients.shape[0] != batch:
        raise ValueError("polynomial_coefficients must match the points batch size")
    if polynomial_coefficients.shape[2] != num_dims:
        raise ValueError(
            "polynomial_coefficients must match the points spatial dimension"
        )
    if polynomial_coefficients.shape[1] not in (0, num_dims + 1):
        raise ValueError("polynomial_coefficients must have zero or num_dims + 1 terms")


def _prepare_rbf_inputs(
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and make the evaluator inputs contiguous."""

    _validate_rbf_inputs(points, controls, radial_coefficients, polynomial_coefficients)
    return (
        points.contiguous(),
        controls.contiguous(),
        radial_coefficients.contiguous(),
        polynomial_coefficients.contiguous(),
    )


@torch.library.custom_op("physicsnemo::rbf_field_warp_impl", mutates_args=())
def rbf_field_warp_impl(
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a thin-plate-spline RBF field with Warp."""

    points_c, controls_c, radial_c, polynomial_c = _prepare_rbf_inputs(
        points, controls, radial_coefficients, polynomial_coefficients
    )
    field = torch.empty_like(points_c)
    batch, num_points, num_dims = points_c.shape
    if batch * num_points == 0:
        return field

    kernels = _rbf_kernels(points_c.dtype)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_c)
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        wp.launch(
            kernels.forward,
            dim=(batch, num_points),
            inputs=[
                _wp_view(points_c, kernels.warp_dtype),
                _wp_view(controls_c, kernels.warp_dtype),
                _wp_view(radial_c, kernels.warp_dtype),
                _wp_view(polynomial_c, kernels.warp_dtype),
                int(controls_c.shape[1]),
                int(num_dims),
                int(polynomial_c.shape[1]),
                _wp_view(field, kernels.warp_dtype),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return field


@rbf_field_warp_impl.register_fake
def _rbf_field_warp_fake(
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> torch.Tensor:
    _ = controls, radial_coefficients, polynomial_coefficients
    return torch.empty(
        points.shape,
        dtype=points.dtype,
        device=points.device,
        memory_format=torch.contiguous_format,
    )


@torch.library.custom_op(
    "physicsnemo::rbf_field_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_field, Tensor points, Tensor controls, "
        "Tensor radial_coefficients, Tensor polynomial_coefficients, "
        "bool need_points=True, bool need_controls=True, "
        "bool need_radial_coefficients=True, "
        "bool need_polynomial_coefficients=True) -> "
        "(Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def rbf_field_warp_backward_impl(
    grad_field: torch.Tensor,
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
    need_points: bool = True,
    need_controls: bool = True,
    need_radial_coefficients: bool = True,
    need_polynomial_coefficients: bool = True,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Evaluate the first-order thin-plate-spline field pullback with Warp."""

    _validate_rbf_inputs(points, controls, radial_coefficients, polynomial_coefficients)
    if grad_field.shape != points.shape:
        raise ValueError("grad_field and points must have matching shapes")
    if grad_field.dtype != points.dtype:
        raise TypeError("grad_field must have the same dtype as points")
    if grad_field.device != points.device:
        raise ValueError("grad_field must be on the same device as points")

    grad_field_c = grad_field.contiguous()
    points_c = points.contiguous()
    controls_c = controls.contiguous()
    radial_c = radial_coefficients.contiguous()
    polynomial_c = polynomial_coefficients.contiguous()
    batch, num_points, num_dims = points_c.shape
    num_controls = controls_c.shape[1]
    num_polynomial_terms = polynomial_c.shape[1]
    num_query_blocks = (
        num_points + _RBF_BACKWARD_QUERY_BLOCK_SIZE - 1
    ) // _RBF_BACKWARD_QUERY_BLOCK_SIZE

    grad_points = torch.zeros_like(points_c) if need_points else None
    grad_controls = torch.zeros_like(controls_c) if need_controls else None
    grad_radial = torch.zeros_like(radial_c) if need_radial_coefficients else None
    grad_polynomial = (
        torch.zeros_like(polynomial_c) if need_polynomial_coefficients else None
    )

    if batch == 0 or num_points == 0:
        return grad_points, grad_controls, grad_radial, grad_polynomial

    kernels = _rbf_kernels(points_c.dtype)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_field_c)
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        if need_points:
            wp.launch(
                kernels.point_backward,
                dim=(batch, num_points),
                inputs=[
                    _wp_view(grad_field_c, kernels.warp_dtype),
                    _wp_view(points_c, kernels.warp_dtype),
                    _wp_view(controls_c, kernels.warp_dtype),
                    _wp_view(radial_c, kernels.warp_dtype),
                    _wp_view(polynomial_c, kernels.warp_dtype),
                    int(num_controls),
                    int(num_dims),
                    int(num_polynomial_terms),
                    _wp_view(grad_points, kernels.warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

        if num_controls > 0 and (need_controls or need_radial_coefficients):
            grad_controls_launch = (
                grad_controls if grad_controls is not None else _empty_3d(points_c)
            )
            grad_radial_launch = (
                grad_radial if grad_radial is not None else _empty_3d(points_c)
            )
            wp.launch(
                kernels.control_and_radial_backward,
                dim=(batch, num_controls, num_dims, num_query_blocks),
                inputs=[
                    _wp_view(grad_field_c, kernels.warp_dtype),
                    _wp_view(points_c, kernels.warp_dtype),
                    _wp_view(controls_c, kernels.warp_dtype),
                    _wp_view(radial_c, kernels.warp_dtype),
                    int(num_points),
                    int(num_dims),
                    int(_RBF_BACKWARD_QUERY_BLOCK_SIZE),
                    int(need_controls),
                    int(need_radial_coefficients),
                    _wp_view(grad_controls_launch, kernels.warp_dtype),
                    _wp_view(grad_radial_launch, kernels.warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

        if need_polynomial_coefficients and num_polynomial_terms > 0:
            wp.launch(
                kernels.polynomial_backward,
                dim=(batch, num_polynomial_terms, num_dims, num_query_blocks),
                inputs=[
                    _wp_view(grad_field_c, kernels.warp_dtype),
                    _wp_view(points_c, kernels.warp_dtype),
                    int(num_points),
                    int(_RBF_BACKWARD_QUERY_BLOCK_SIZE),
                    _wp_view(grad_polynomial, kernels.warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

    return grad_points, grad_controls, grad_radial, grad_polynomial


@rbf_field_warp_backward_impl.register_fake
def _rbf_field_warp_backward_fake(
    grad_field: torch.Tensor,
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
    need_points: bool = True,
    need_controls: bool = True,
    need_radial_coefficients: bool = True,
    need_polynomial_coefficients: bool = True,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    _ = grad_field
    return (
        torch.empty_like(points, memory_format=torch.contiguous_format)
        if need_points
        else None,
        torch.empty_like(controls, memory_format=torch.contiguous_format)
        if need_controls
        else None,
        torch.empty_like(radial_coefficients, memory_format=torch.contiguous_format)
        if need_radial_coefficients
        else None,
        torch.empty_like(polynomial_coefficients, memory_format=torch.contiguous_format)
        if need_polynomial_coefficients
        else None,
    )


def _setup_rbf_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    _ = output
    ctx.save_for_backward(*(tensor.contiguous() for tensor in inputs))


def _backward_rbf(
    ctx: torch.autograd.function.FunctionCtx,
    grad_field: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    needs = ctx.needs_input_grad
    if grad_field is None or not any(needs):
        return None, None, None, None
    points, controls, radial_coefficients, polynomial_coefficients = ctx.saved_tensors
    return rbf_field_warp_backward_impl(
        grad_field,
        points,
        controls,
        radial_coefficients,
        polynomial_coefficients,
        bool(needs[0]),
        bool(needs[1]),
        bool(needs[2]),
        bool(needs[3]),
    )


rbf_field_warp_impl.register_autograd(
    _backward_rbf,
    setup_context=_setup_rbf_context,
)


def rbf_field_warp(
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_coefficients: torch.Tensor,
    polynomial_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a normalized rank-3 thin-plate-spline RBF field with Warp."""

    return rbf_field_warp_impl(
        points,
        controls,
        radial_coefficients,
        polynomial_coefficients,
    )


__all__ = [
    "rbf_field_warp",
    "rbf_field_warp_backward_impl",
    "rbf_field_warp_impl",
]
