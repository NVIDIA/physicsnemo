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

"""Torch custom-op integration for Warp-backed mesh morphing."""

from __future__ import annotations

from typing import NamedTuple

import torch
import warp as wp
from jaxtyping import Float

from physicsnemo.core.function_spec import FunctionSpec

from .._torch_impl import displace_points_torch
from .._utils import _zero_dependency
from .kernels import (
    shepard_backward_f32,
    shepard_backward_f64,
    shepard_forward_f32,
    shepard_forward_f64,
    shepard_point_backward_f32,
    shepard_point_backward_f64,
)

wp.init()
wp.config.log_level = wp.LOG_WARNING

# Borrowed Torch streams use ``warp_stream_scope`` so Warp's temporary
# allocations stay alive until queued work finishes. Entry synchronization is
# disabled at the launch sites below to preserve CUDA Graph capture.


class _ShepardKernelSet(NamedTuple):
    """Warp dtype and matching compact-Shepard kernels."""

    warp_dtype: object
    forward: object
    backward: object
    point_backward: object


_SHEPARD_KERNELS = {
    torch.float32: _ShepardKernelSet(
        wp.float32,
        shepard_forward_f32,
        shepard_backward_f32,
        shepard_point_backward_f32,
    ),
    torch.float64: _ShepardKernelSet(
        wp.float64,
        shepard_forward_f64,
        shepard_backward_f64,
        shepard_point_backward_f64,
    ),
}


def _check_common_dtype(*tensors: Float[torch.Tensor, "..."]) -> None:
    dtype = tensors[0].dtype
    device = tensors[0].device
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"deformation supports float32 and float64, got {dtype}")
    if any(t.dtype != dtype for t in tensors):
        raise TypeError("all floating deformation tensors must have the same dtype")
    if any(t.device != device for t in tensors):
        raise ValueError("all deformation tensors must be on the same device")


def _empty_contiguous_like(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate a contiguous tensor with ``tensor``'s shape, dtype, and device."""

    return torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)


def _empty_3d(reference: torch.Tensor) -> torch.Tensor:
    """Return a rank-3 zero-size launch placeholder."""

    return torch.empty((0, 0, 0), dtype=reference.dtype, device=reference.device)


def _empty_2d(reference: torch.Tensor) -> torch.Tensor:
    """Return a rank-2 zero-size launch placeholder."""

    return torch.empty((0, 0), dtype=reference.dtype, device=reference.device)


def _wp_view(tensor: torch.Tensor, dtype):
    """Create the faster zero-copy Warp array descriptor for a Torch tensor."""

    return wp.from_torch(
        tensor.detach(), dtype=dtype, return_ctype=True, requires_grad=False
    )


def _shepard_kernels(dtype: torch.dtype) -> _ShepardKernelSet:
    """Return one internally consistent dtype/kernel family."""

    try:
        return _SHEPARD_KERNELS[dtype]
    except KeyError:
        raise TypeError(
            f"Warp morphing supports float32 and float64, got {dtype}"
        ) from None


def _prepare_shepard_forward_inputs(
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and make contiguous the inputs shared by both forward ops."""

    _check_common_dtype(points, controls, control_displacements, radii)
    if points.ndim != 3 or controls.ndim != 3:
        raise ValueError("points and controls must be normalized rank-3 tensors")
    if control_displacements.shape != controls.shape:
        raise ValueError("control_displacements must match controls")
    if controls.shape[0] != points.shape[0] or controls.shape[2] != points.shape[2]:
        raise ValueError(
            "points and controls must have aligned batch/spatial dimensions"
        )
    if radii.shape != controls.shape[:2]:
        raise ValueError("radii must have shape (batch, num_controls)")

    return (
        points.contiguous(),
        controls.contiguous(),
        control_displacements.contiguous(),
        radii.contiguous(),
    )


def _launch_shepard_forward(
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor,
    radii: torch.Tensor,
    field: torch.Tensor,
    auxiliaries: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]
    | None,
) -> None:
    """Launch the forward kernel with optional backward auxiliaries."""

    batch, num_points, num_dims = points.shape
    if batch * num_points == 0:
        return

    kernels = _shepard_kernels(points.dtype)
    if auxiliaries is None:
        # The kernel ignores these aliased zero-size placeholders when
        # save_auxiliaries is false.
        empty_float = _empty_2d(points)
        empty_int = torch.empty((0, 0), dtype=torch.int32, device=points.device)
        min_q = denominator = empty_float
        exact_count = reference_index = empty_int
        correction = None
    else:
        min_q, denominator, exact_count, reference_index, correction = auxiliaries
    correction_launch = correction if correction is not None else _empty_3d(points)

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points)
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        wp.launch(
            kernels.forward,
            dim=(batch, num_points),
            inputs=[
                _wp_view(points, kernels.warp_dtype),
                _wp_view(controls, kernels.warp_dtype),
                _wp_view(control_displacements, kernels.warp_dtype),
                _wp_view(radii, kernels.warp_dtype),
                int(controls.shape[1]),
                int(num_dims),
                int(auxiliaries is not None),
                int(correction is not None),
                _wp_view(field, kernels.warp_dtype),
                _wp_view(min_q, kernels.warp_dtype),
                _wp_view(denominator, kernels.warp_dtype),
                _wp_view(exact_count, wp.int32),
                _wp_view(reference_index, wp.int32),
                _wp_view(correction_launch, kernels.warp_dtype),
            ],
            device=wp_device,
            stream=wp_stream,
        )


@torch.library.custom_op(
    "physicsnemo::compact_shepard_field_warp_impl",
    mutates_args=(),
    schema=(
        "(Tensor points, Tensor controls, Tensor control_displacements, "
        "Tensor radii, bool save_correction=True) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)
def compact_shepard_field_warp_impl(
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor,
    radii: torch.Tensor,
    save_correction: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate the compact Shepard displacement field with Warp."""
    points_c, controls_c, control_displacements_c, radii_c = (
        _prepare_shepard_forward_inputs(points, controls, control_displacements, radii)
    )
    batch, num_points, _ = points_c.shape
    field = torch.empty_like(points_c)
    min_q = torch.empty((batch, num_points), dtype=points.dtype, device=points.device)
    denominator = torch.empty_like(min_q)
    exact_count = torch.empty(
        (batch, num_points), dtype=torch.int32, device=points.device
    )
    reference_index = torch.empty_like(exact_count)
    correction = torch.empty_like(points_c) if save_correction else _empty_3d(points_c)
    _launch_shepard_forward(
        points_c,
        controls_c,
        control_displacements_c,
        radii_c,
        field,
        (
            min_q,
            denominator,
            exact_count,
            reference_index,
            correction if save_correction else None,
        ),
    )
    return field, min_q, denominator, exact_count, reference_index, correction


@compact_shepard_field_warp_impl.register_fake
def _compact_shepard_field_warp_fake(
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor,
    radii: torch.Tensor,
    save_correction: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _ = controls, control_displacements, radii
    prefix = points.shape[:2]
    return (
        _empty_contiguous_like(points),
        torch.empty(prefix, dtype=points.dtype, device=points.device),
        torch.empty(prefix, dtype=points.dtype, device=points.device),
        torch.empty(prefix, dtype=torch.int32, device=points.device),
        torch.empty(prefix, dtype=torch.int32, device=points.device),
        (_empty_contiguous_like(points) if save_correction else _empty_3d(points)),
    )


@torch.library.custom_op(
    "physicsnemo::compact_shepard_field_warp_forward_only_impl", mutates_args=()
)
def compact_shepard_field_warp_forward_only_impl(
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only the field, omitting every backward-only auxiliary."""
    points_c, controls_c, control_displacements_c, radii_c = (
        _prepare_shepard_forward_inputs(points, controls, control_displacements, radii)
    )
    field = torch.empty_like(points_c)
    _launch_shepard_forward(
        points_c,
        controls_c,
        control_displacements_c,
        radii_c,
        field,
        None,
    )
    return field


@compact_shepard_field_warp_forward_only_impl.register_fake
def _compact_shepard_field_warp_forward_only_fake(
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    _ = controls, control_displacements, radii
    return _empty_contiguous_like(points)


def _setup_shepard_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    points, controls, control_displacements, radii, _ = inputs
    _, min_q, denominator, exact_count, reference_index, correction = output
    needs = ctx.needs_input_grad
    ctx.save_geometry_values = bool(needs[0] or needs[1] or needs[3])
    saved = [
        points.contiguous(),
        controls.contiguous(),
        radii.contiguous(),
        min_q,
        denominator,
        exact_count,
        reference_index,
    ]
    if ctx.save_geometry_values:
        # Keep the separately accumulated correction: reconstructing it as
        # field-reference would lose near-handle precision.
        saved.extend([control_displacements.contiguous(), correction])
    ctx.save_for_backward(*saved)
    ctx.mark_non_differentiable(
        min_q, denominator, exact_count, reference_index, correction
    )


# This opaque pullback is the deliberate first-order autograd boundary; its
# fake implementation supports AOT tracing without promising higher derivatives.
@torch.library.custom_op(
    "physicsnemo::compact_shepard_field_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_field, Tensor points, Tensor controls, "
        "Tensor? control_displacements, Tensor radii, Tensor min_q, "
        "Tensor denominator, Tensor exact_count, Tensor reference_index, "
        "Tensor? correction, bool need_points=True, bool need_controls=True, "
        "bool need_control_displacements=True, bool need_radii=True) -> "
        "(Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def compact_shepard_field_warp_backward_impl(
    grad_field: torch.Tensor,
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor | None,
    radii: torch.Tensor,
    min_q: torch.Tensor,
    denominator: torch.Tensor,
    exact_count: torch.Tensor,
    reference_index: torch.Tensor,
    correction: torch.Tensor | None,
    need_points: bool = True,
    need_controls: bool = True,
    need_control_displacements: bool = True,
    need_radii: bool = True,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Evaluate the first-order compact-Shepard pullback with Warp."""
    floating_inputs = [grad_field, points, controls, radii, min_q, denominator]
    if control_displacements is not None:
        floating_inputs.append(control_displacements)
    if correction is not None:
        floating_inputs.append(correction)
    _check_common_dtype(*floating_inputs)
    if exact_count.dtype != torch.int32 or exact_count.device != points.device:
        raise TypeError("exact_count must be an int32 tensor on the points device")
    if reference_index.dtype != torch.int32 or reference_index.device != points.device:
        raise TypeError("reference_index must be an int32 tensor on the points device")
    if grad_field.shape != points.shape:
        raise ValueError("grad_field and points must have matching shapes")
    if controls.ndim != 3:
        raise ValueError("controls must be rank 3")
    if (
        control_displacements is not None
        and controls.shape != control_displacements.shape
    ):
        raise ValueError("controls and control_displacements must match")
    geometry_needed = need_points or need_controls or need_radii
    if geometry_needed and (control_displacements is None or correction is None):
        raise ValueError(
            "control_displacements and correction are required for geometry gradients"
        )
    if radii.shape != controls.shape[:2]:
        raise ValueError("radii must have shape (batch, num_controls)")
    if min_q.shape != points.shape[:2] or denominator.shape != points.shape[:2]:
        raise ValueError("saved normalization tensors must match the query prefix")
    if (
        exact_count.shape != points.shape[:2]
        or reference_index.shape != points.shape[:2]
    ):
        raise ValueError("saved index/count tensors must match the query prefix")
    if correction is not None and correction.shape != points.shape:
        raise ValueError("saved correction must match points")

    grad_field_c = grad_field.contiguous()
    points_c = points.contiguous()
    controls_c = controls.contiguous()
    control_displacements_c = (
        control_displacements.contiguous()
        if control_displacements is not None
        else None
    )
    radii_c = radii.contiguous()
    min_q_c = min_q.contiguous()
    denominator_c = denominator.contiguous()
    exact_count_c = exact_count.contiguous()
    reference_index_c = reference_index.contiguous()
    correction_c = correction.contiguous() if correction is not None else None
    batch, num_points, num_dims = points.shape
    num_controls = controls.shape[1]
    grad_points = (
        (
            torch.zeros(points_c.shape, dtype=points.dtype, device=points.device)
            if num_controls == 0
            else torch.empty(points_c.shape, dtype=points.dtype, device=points.device)
        )
        if need_points
        else None
    )
    grad_controls = torch.zeros_like(controls_c) if need_controls else None
    grad_control_displacements = (
        torch.zeros_like(controls_c) if need_control_displacements else None
    )
    grad_radii = torch.zeros_like(radii_c) if need_radii else None

    if batch * num_points * num_controls > 0 and (
        need_points or need_controls or need_control_displacements or need_radii
    ):
        kernels = _shepard_kernels(points.dtype)
        wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_field_c)
        with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
            control_displacements_launch = (
                control_displacements_c
                if control_displacements_c is not None
                else _empty_3d(points_c)
            )
            correction_launch = (
                correction_c if correction_c is not None else _empty_3d(points_c)
            )
            common = [
                _wp_view(points_c, kernels.warp_dtype),
                _wp_view(controls_c, kernels.warp_dtype),
                _wp_view(control_displacements_launch, kernels.warp_dtype),
                _wp_view(radii_c, kernels.warp_dtype),
                _wp_view(min_q_c, kernels.warp_dtype),
                _wp_view(denominator_c, kernels.warp_dtype),
                _wp_view(exact_count_c, wp.int32),
                _wp_view(reference_index_c, wp.int32),
                _wp_view(correction_launch, kernels.warp_dtype),
                _wp_view(grad_field_c, kernels.warp_dtype),
            ]
            if need_points:
                wp.launch(
                    kernels.point_backward,
                    dim=(batch, num_points),
                    inputs=[
                        *common,
                        int(num_controls),
                        int(num_dims),
                        _wp_view(grad_points, kernels.warp_dtype),
                    ],
                    device=wp_device,
                    stream=wp_stream,
                )
            if need_controls or need_control_displacements or need_radii:
                controls_grad_launch = (
                    grad_controls if grad_controls is not None else _empty_3d(points_c)
                )
                displacement_grad_launch = (
                    grad_control_displacements
                    if grad_control_displacements is not None
                    else _empty_3d(points_c)
                )
                radii_grad_launch = (
                    grad_radii if grad_radii is not None else _empty_2d(points_c)
                )
                wp.launch(
                    kernels.backward,
                    dim=(batch, num_points, num_controls),
                    inputs=[
                        *common,
                        int(num_dims),
                        int(need_controls),
                        int(need_control_displacements),
                        int(need_radii),
                        _wp_view(controls_grad_launch, kernels.warp_dtype),
                        _wp_view(displacement_grad_launch, kernels.warp_dtype),
                        _wp_view(radii_grad_launch, kernels.warp_dtype),
                    ],
                    device=wp_device,
                    stream=wp_stream,
                )
    return grad_points, grad_controls, grad_control_displacements, grad_radii


@compact_shepard_field_warp_backward_impl.register_fake
def _compact_shepard_field_warp_backward_fake(
    grad_field: torch.Tensor,
    points: torch.Tensor,
    controls: torch.Tensor,
    control_displacements: torch.Tensor | None,
    radii: torch.Tensor,
    min_q: torch.Tensor,
    denominator: torch.Tensor,
    exact_count: torch.Tensor,
    reference_index: torch.Tensor,
    correction: torch.Tensor | None,
    need_points: bool = True,
    need_controls: bool = True,
    need_control_displacements: bool = True,
    need_radii: bool = True,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    _ = (
        grad_field,
        control_displacements,
        min_q,
        denominator,
        exact_count,
        reference_index,
        correction,
    )
    return (
        _empty_contiguous_like(points) if need_points else None,
        _empty_contiguous_like(controls) if need_controls else None,
        _empty_contiguous_like(controls) if need_control_displacements else None,
        _empty_contiguous_like(radii) if need_radii else None,
    )


def _backward_shepard(
    ctx: torch.autograd.function.FunctionCtx,
    grad_field: torch.Tensor | None,
    grad_min_q: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
    grad_exact_count: torch.Tensor | None,
    grad_reference_index: torch.Tensor | None,
    grad_correction: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    _ = (
        grad_min_q,
        grad_denominator,
        grad_exact_count,
        grad_reference_index,
        grad_correction,
    )
    needs = ctx.needs_input_grad
    if grad_field is None or not any(needs):
        return None, None, None, None, None

    saved = list(ctx.saved_tensors)
    points = saved.pop(0)
    controls = saved.pop(0)
    radii = saved.pop(0)
    min_q = saved.pop(0)
    denominator = saved.pop(0)
    exact_count = saved.pop(0)
    reference_index = saved.pop(0)
    if ctx.save_geometry_values:
        control_displacements = saved.pop(0)
        correction = saved.pop(0)
    else:
        control_displacements = correction = None
    grad_points, grad_controls, grad_control_displacements, grad_radii = (
        compact_shepard_field_warp_backward_impl(
            grad_field,
            points,
            controls,
            control_displacements,
            radii,
            min_q,
            denominator,
            exact_count,
            reference_index,
            correction,
            bool(needs[0]),
            bool(needs[1]),
            bool(needs[2]),
            bool(needs[3]),
        )
    )

    return (
        grad_points if needs[0] else None,
        grad_controls if needs[1] else None,
        grad_control_displacements if needs[2] else None,
        grad_radii if needs[3] else None,
        None,
    )


compact_shepard_field_warp_impl.register_autograd(
    _backward_shepard, setup_context=_setup_shepard_context
)


def morph_points_warp(
    points: torch.Tensor,
    control_points: torch.Tensor,
    control_displacements: torch.Tensor,
    radius: torch.Tensor,
    point_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Normalized rank-3 Warp compact-Shepard morphing entry point."""
    if control_points.shape[1] == 0:
        # Preserve every differentiable zero dependency without paying for two
        # Warp launches or allocating field auxiliaries.
        zero = _zero_dependency(
            control_points,
            control_displacements,
            radius,
            point_weights,
        )
        return points + zero

    points_c = points.contiguous()
    controls_c = control_points.contiguous()
    control_displacements_c = control_displacements.contiguous()
    radius_c = radius.contiguous()
    point_weights_c = point_weights.contiguous() if point_weights is not None else None
    needs_field_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (points_c, controls_c, control_displacements_c, radius_c)
    )
    if needs_field_grad:
        needs_geometry_grad = (
            points_c.requires_grad or controls_c.requires_grad or radius_c.requires_grad
        )
        field, _, _, _, _, _ = compact_shepard_field_warp_impl(
            points_c,
            controls_c,
            control_displacements_c,
            radius_c,
            needs_geometry_grad,
        )
    else:
        field = compact_shepard_field_warp_forward_only_impl(
            points_c, controls_c, control_displacements_c, radius_c
        )
    # Only the compact Shepard field warrants a Warp kernel. Native Torch
    # handles the final point weighting and addition without a dense Warp API.
    return displace_points_torch(points_c, field, point_weights_c)


__all__ = [
    "compact_shepard_field_warp_backward_impl",
    "compact_shepard_field_warp_forward_only_impl",
    "compact_shepard_field_warp_impl",
    "morph_points_warp",
]
