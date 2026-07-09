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

"""Torch custom-op integration for Warp-backed Procrustes rotation."""

from __future__ import annotations

from typing import NamedTuple

import torch
import warp as wp
from jaxtyping import Float

from physicsnemo.core.function_spec import FunctionSpec

from .._procrustes_common import solve_procrustes
from .procrustes_kernels import (
    proper_rotation_backward_1d_f32,
    proper_rotation_backward_1d_f64,
    proper_rotation_backward_2d_f64,
    proper_rotation_backward_3d_f64,
    proper_rotation_forward_1d_f32,
    proper_rotation_forward_1d_f64,
    proper_rotation_forward_2d_f64,
    proper_rotation_forward_3d_f64,
)

wp.init()
wp.config.log_level = wp.LOG_WARNING


class _ProcrustesKernelSet(NamedTuple):
    """Warp composite dtypes and matching projection kernels."""

    matrix_dtype: object
    forward: object
    backward: object


_PROCRUSTES_KERNELS = {
    (torch.float32, 1): _ProcrustesKernelSet(
        wp.float32,
        proper_rotation_forward_1d_f32,
        proper_rotation_backward_1d_f32,
    ),
    (torch.float64, 1): _ProcrustesKernelSet(
        wp.float64,
        proper_rotation_forward_1d_f64,
        proper_rotation_backward_1d_f64,
    ),
    (torch.float64, 2): _ProcrustesKernelSet(
        wp.mat22d,
        proper_rotation_forward_2d_f64,
        proper_rotation_backward_2d_f64,
    ),
    (torch.float64, 3): _ProcrustesKernelSet(
        wp.mat33d,
        proper_rotation_forward_3d_f64,
        proper_rotation_backward_3d_f64,
    ),
}


def _kernel_set(
    covariance: Float[torch.Tensor, "batch num_dims num_dims"],
) -> _ProcrustesKernelSet:
    """Return the kernel family matching a covariance tensor."""

    try:
        return _PROCRUSTES_KERNELS[(covariance.dtype, covariance.shape[-1])]
    except KeyError:
        raise TypeError(
            "Warp Procrustes rotation supports float32 covariance matrices "
            "in 1D and float64 covariance matrices in 1D/2D/3D"
        ) from None


def _validate_covariance(
    covariance: Float[torch.Tensor, "batch num_dims num_dims"],
) -> None:
    """Validate the normalized covariance custom-op contract."""

    if covariance.ndim != 3 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError(
            f"covariance must have shape (B, D, D), got {tuple(covariance.shape)}"
        )
    if covariance.shape[-1] not in (1, 2, 3):
        raise ValueError(
            "covariance spatial dimension must be 1, 2, or 3, got "
            f"{covariance.shape[-1]}"
        )
    if covariance.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "covariance must have dtype torch.float32 or torch.float64, got "
            f"{covariance.dtype}"
        )
    if covariance.dtype == torch.float32 and covariance.shape[-1] in (2, 3):
        raise TypeError(
            "2D/3D Warp Procrustes rotation requires float64 covariance; "
            "the public Warp backend promotes float32 covariance internally"
        )


def _wp_view(tensor: Float[torch.Tensor, "..."], dtype, num_dims: int):
    """Create a zero-copy scalar, vector, or matrix Warp descriptor."""

    value = tensor.detach()
    if num_dims == 1:
        value = value.reshape(-1)
    return wp.from_torch(
        value,
        dtype=dtype,
        return_ctype=True,
        requires_grad=False,
    )


def _launch_forward(
    covariance: Float[torch.Tensor, "batch num_dims num_dims"],
    rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    symmetric_factor: Float[torch.Tensor, "batch num_dims num_dims"],
) -> None:
    """Launch one proper-rotation projection per covariance matrix."""

    batch_size, num_dims, _ = covariance.shape
    if batch_size == 0:
        return
    kernels = _kernel_set(covariance)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(covariance)
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        wp.launch(
            kernels.forward,
            dim=batch_size,
            inputs=[
                _wp_view(covariance, kernels.matrix_dtype, num_dims),
                _wp_view(rotation, kernels.matrix_dtype, num_dims),
                _wp_view(symmetric_factor, kernels.matrix_dtype, num_dims),
            ],
            device=wp_device,
            stream=wp_stream,
        )


def _launch_backward(
    grad_rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    symmetric_factor: Float[torch.Tensor, "batch num_dims num_dims"],
    grad_covariance: Float[torch.Tensor, "batch num_dims num_dims"],
) -> None:
    """Launch the analytic projection VJP for each batch element."""

    batch_size, num_dims, _ = rotation.shape
    if batch_size == 0:
        return
    kernels = _kernel_set(rotation)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(rotation)
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        wp.launch(
            kernels.backward,
            dim=batch_size,
            inputs=[
                _wp_view(grad_rotation, kernels.matrix_dtype, num_dims),
                _wp_view(rotation, kernels.matrix_dtype, num_dims),
                _wp_view(symmetric_factor, kernels.matrix_dtype, num_dims),
                _wp_view(grad_covariance, kernels.matrix_dtype, num_dims),
            ],
            device=wp_device,
            stream=wp_stream,
        )


@torch.library.custom_op(
    "physicsnemo::procrustes_rotation_warp_impl",
    mutates_args=(),
    schema="(Tensor covariance) -> (Tensor, Tensor)",
)
def procrustes_rotation_warp_impl(
    covariance: Float[torch.Tensor, "batch num_dims num_dims"],
) -> tuple[
    Float[torch.Tensor, "batch num_dims num_dims"],
    Float[torch.Tensor, "batch num_dims num_dims"],
]:
    """Project covariance matrices onto proper rotations with Warp."""

    _validate_covariance(covariance)
    covariance_c = covariance.contiguous()
    rotation = torch.empty_like(covariance_c)
    symmetric_factor = torch.empty_like(covariance_c)
    _launch_forward(
        covariance_c,
        rotation,
        symmetric_factor,
    )
    return rotation, symmetric_factor


@procrustes_rotation_warp_impl.register_fake
def _procrustes_rotation_warp_fake(
    covariance: Float[torch.Tensor, "batch num_dims num_dims"],
) -> tuple[
    Float[torch.Tensor, "batch num_dims num_dims"],
    Float[torch.Tensor, "batch num_dims num_dims"],
]:
    """Propagate shapes through the Warp projection custom op."""

    return (
        torch.empty_like(covariance, memory_format=torch.contiguous_format),
        torch.empty_like(covariance, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op(
    "physicsnemo::procrustes_rotation_warp_backward_impl",
    mutates_args=(),
    schema="(Tensor grad_rotation, Tensor rotation, Tensor symmetric_factor) -> Tensor",
)
def procrustes_rotation_warp_backward_impl(
    grad_rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    symmetric_factor: Float[torch.Tensor, "batch num_dims num_dims"],
) -> Float[torch.Tensor, "batch num_dims num_dims"]:
    """Apply the opaque first-order Warp projection pullback."""

    grad_rotation_c = grad_rotation.contiguous()
    rotation_c = rotation.contiguous()
    symmetric_factor_c = symmetric_factor.contiguous()
    grad_covariance = torch.empty_like(rotation_c)
    _launch_backward(
        grad_rotation_c,
        rotation_c,
        symmetric_factor_c,
        grad_covariance,
    )
    return grad_covariance


@procrustes_rotation_warp_backward_impl.register_fake
def _procrustes_rotation_warp_backward_fake(
    grad_rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    rotation: Float[torch.Tensor, "batch num_dims num_dims"],
    symmetric_factor: Float[torch.Tensor, "batch num_dims num_dims"],
) -> Float[torch.Tensor, "batch num_dims num_dims"]:
    """Propagate the covariance-gradient shape through AOT tracing."""

    _ = grad_rotation, symmetric_factor
    return torch.empty_like(rotation, memory_format=torch.contiguous_format)


def _setup_procrustes_rotation_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[Float[torch.Tensor, "batch num_dims num_dims"]],
    output: tuple[
        Float[torch.Tensor, "batch num_dims num_dims"],
        Float[torch.Tensor, "batch num_dims num_dims"],
    ],
) -> None:
    """Save the compact polar-factor state required by the VJP."""

    _ = inputs
    rotation, symmetric_factor = output
    ctx.save_for_backward(rotation, symmetric_factor)
    ctx.mark_non_differentiable(symmetric_factor)


def _backward_procrustes_rotation(
    ctx: torch.autograd.function.FunctionCtx,
    grad_rotation: Float[torch.Tensor, "batch num_dims num_dims"] | None,
    grad_symmetric_factor: Float[torch.Tensor, "batch num_dims num_dims"] | None,
) -> tuple[Float[torch.Tensor, "batch num_dims num_dims"] | None]:
    """Route the rotation cotangent through the Warp pullback op."""

    _ = grad_symmetric_factor
    if grad_rotation is None or not ctx.needs_input_grad[0]:
        return (None,)
    rotation, symmetric_factor = ctx.saved_tensors
    return (
        procrustes_rotation_warp_backward_impl(
            grad_rotation,
            rotation,
            symmetric_factor,
        ),
    )


procrustes_rotation_warp_impl.register_autograd(
    _backward_procrustes_rotation,
    setup_context=_setup_procrustes_rotation_context,
)


def _project_rotation_warp(
    covariance: Float[torch.Tensor, "batch num_dims num_dims"],
) -> Float[torch.Tensor, "batch num_dims num_dims"]:
    """Return only the public rotation result from the custom op."""

    # The matrix projector uses double-precision closed-form/Jacobi solves for
    # stable rotations and VJPs on elongated point clouds. Promote only this
    # tiny BxDxD projection, retaining float32 point reductions and restoring
    # the public dtype afterward.
    projection_input = (
        covariance.to(torch.float64)
        if covariance.dtype == torch.float32 and covariance.shape[-1] in (2, 3)
        else covariance
    )
    rotation, _ = procrustes_rotation_warp_impl(projection_input)
    return rotation.to(covariance.dtype)


def procrustes_warp(
    source: Float[torch.Tensor, "batch num_points num_dims"],
    target: Float[torch.Tensor, "batch num_points num_dims"],
    scale: bool = True,
) -> tuple[
    Float[torch.Tensor, "batch num_dims num_dims"],
    Float[torch.Tensor, "batch num_dims"],
    Float[torch.Tensor, " batch"],
]:
    """Solve normalized rank-three Procrustes registration with Warp."""

    return solve_procrustes(source, target, scale, _project_rotation_warp)


__all__ = [
    "procrustes_rotation_warp_backward_impl",
    "procrustes_rotation_warp_impl",
    "procrustes_warp",
]
