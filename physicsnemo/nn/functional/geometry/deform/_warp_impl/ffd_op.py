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

"""Torch custom-op integration for Warp-backed lattice free-form deformation."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import warp as wp
from jaxtyping import Bool, Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from .._torch_impl import displace_points_torch
from .._utils import _FFD_MIN_NODES, _ffd_window_size, _zero_dependency
from .ffd_kernels import (
    BERNSTEIN_BASIS_ID,
    BSPLINE_BASIS_ID,
    CUBIC_HERMITE_BASIS_ID,
    LINEAR_BASIS_ID,
    QUINTIC_HERMITE_BASIS_ID,
    ffd_backward_f32,
    ffd_backward_f64,
    ffd_forward_f32,
    ffd_forward_f64,
    ffd_point_backward_f32,
    ffd_point_backward_f64,
)
from .op import _check_common_dtype, _empty_contiguous_like, _wp_view

_BASIS_IDS = {
    "bernstein": BERNSTEIN_BASIS_ID,
    "bspline": BSPLINE_BASIS_ID,
    "linear": LINEAR_BASIS_ID,
    "cubic_hermite": CUBIC_HERMITE_BASIS_ID,
    "quintic_hermite": QUINTIC_HERMITE_BASIS_ID,
}


class _FFDKernelSet(NamedTuple):
    """Warp dtype and matching lattice free-form deformation kernels."""

    warp_dtype: object
    forward: object
    backward: object
    point_backward: object


_FFD_KERNELS = {
    torch.float32: _FFDKernelSet(
        wp.float32,
        ffd_forward_f32,
        ffd_backward_f32,
        ffd_point_backward_f32,
    ),
    torch.float64: _FFDKernelSet(
        wp.float64,
        ffd_forward_f64,
        ffd_backward_f64,
        ffd_point_backward_f64,
    ),
}

# Deliberately retain each device's static resolution tensor for the process
# lifetime. Captured CUDA Graphs keep its device address, so eviction or manual
# clearing could invalidate later graph replays. Each entry contains only one
# int32 value per lattice axis. Reuse also avoids host-to-device copies.
_RESOLUTION_TENSOR_REGISTRY: dict[
    tuple[torch.device, tuple[int, ...]], torch.Tensor
] = {}


def _ffd_kernels(dtype: torch.dtype) -> _FFDKernelSet:
    """Return one internally consistent dtype/kernel family."""

    try:
        return _FFD_KERNELS[dtype]
    except KeyError:
        raise TypeError(
            f"Warp free-form deformation supports float32 and float64, got {dtype}"
        ) from None


def _resolution_tensor(
    resolution: list[int], device: torch.device
) -> Int[torch.Tensor, " num_dims"]:
    key = (device, tuple(resolution))
    cached = _RESOLUTION_TENSOR_REGISTRY.get(key)
    if cached is None:
        cached = torch.tensor(resolution, dtype=torch.int32, device=device)
        _RESOLUTION_TENSOR_REGISTRY[key] = cached
    return cached


def _validate_ffd_geometry(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: list[int],
    basis: str,
) -> None:
    """Validate the normalized query/lattice geometry shared by both ops."""

    if basis not in _BASIS_IDS:
        raise ValueError(
            "basis must be 'bernstein', 'bspline', 'linear', 'cubic_hermite', "
            f"or 'quintic_hermite', got {basis!r}"
        )
    if points.ndim != 3:
        raise ValueError("points must be a normalized rank-3 tensor")
    batch, _, num_dims = points.shape
    if len(resolution) != num_dims:
        raise ValueError(
            "resolution must list one lattice size per coordinate dimension, got "
            f"{len(resolution)} sizes for {num_dims} dimensions"
        )
    min_nodes = _FFD_MIN_NODES[basis]
    if any(int(nodes) < min_nodes for nodes in resolution):
        raise ValueError(
            f"basis '{basis}' requires at least {min_nodes} lattice nodes per "
            f"axis, got lattice resolution {tuple(resolution)}"
        )
    if tuple(origin.shape) != (batch, num_dims) or tuple(extent.shape) != (
        batch,
        num_dims,
    ):
        raise ValueError("origin and extent must have shape (batch, num_dims)")


def _validate_ffd_lattice(
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    resolution: list[int],
) -> None:
    if control_displacements.ndim != 3:
        raise ValueError("control_displacements must be a normalized rank-3 tensor")
    expected = (points.shape[0], math.prod(resolution), points.shape[2])
    if tuple(control_displacements.shape) != expected:
        raise ValueError(
            "control_displacements must have one row per lattice node with shape "
            f"{expected}, got {tuple(control_displacements.shape)}"
        )


@torch.library.custom_op(
    "physicsnemo::ffd_field_warp_impl",
    mutates_args=(),
    schema=(
        "(Tensor points, Tensor control_displacements, Tensor origin, "
        "Tensor extent, int[] resolution, str basis) -> Tensor"
    ),
)
def ffd_field_warp_impl(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"],
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: list[int],
    basis: str,
) -> Float[torch.Tensor, "batch num_points num_dims"]:
    """Evaluate the lattice free-form displacement field with Warp."""
    _check_common_dtype(points, control_displacements, origin, extent)
    _validate_ffd_geometry(points, origin, extent, resolution, basis)
    _validate_ffd_lattice(control_displacements, points, resolution)

    points_c = points.contiguous()
    lattice_c = control_displacements.contiguous()
    origin_c = origin.contiguous()
    extent_c = extent.contiguous()
    field = torch.empty_like(points_c)
    batch, num_points, num_dims = points_c.shape
    if batch * num_points == 0:
        return field

    kernels = _ffd_kernels(points.dtype)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_c)
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        wp.launch(
            kernels.forward,
            dim=(batch, num_points),
            inputs=[
                _wp_view(points_c, kernels.warp_dtype),
                _wp_view(lattice_c, kernels.warp_dtype),
                _wp_view(origin_c, kernels.warp_dtype),
                _wp_view(extent_c, kernels.warp_dtype),
                _wp_view(_resolution_tensor(resolution, points.device), wp.int32),
                int(_BASIS_IDS[basis]),
                int(num_dims),
                int(lattice_c.shape[1]),
                int(_ffd_window_size(resolution, basis)),
                _wp_view(field, kernels.warp_dtype),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return field


@ffd_field_warp_impl.register_fake
def _ffd_field_warp_fake(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"],
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: list[int],
    basis: str,
) -> Float[torch.Tensor, "batch num_points num_dims"]:
    _ = control_displacements, origin, extent, resolution, basis
    return _empty_contiguous_like(points)


# This opaque pullback is the deliberate first-order autograd boundary. Its
# fake implementation supports AOT tracing without promising higher derivatives.
@torch.library.custom_op(
    "physicsnemo::ffd_field_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_field, Tensor points, Tensor? control_displacements, "
        "Tensor origin, Tensor extent, int[] resolution, str basis, "
        "bool need_points=True, bool need_control_displacements=True) -> "
        "(Tensor?, Tensor?)"
    ),
)
def ffd_field_warp_backward_impl(
    grad_field: Float[torch.Tensor, "batch num_points num_dims"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"] | None,
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: list[int],
    basis: str,
    need_points: bool = True,
    need_control_displacements: bool = True,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch lattice_nodes num_dims"] | None,
]:
    """Evaluate the first-order lattice free-form deformation pullback with Warp."""
    floating_inputs = [grad_field, points, origin, extent]
    if control_displacements is not None:
        floating_inputs.append(control_displacements)
    _check_common_dtype(*floating_inputs)
    _validate_ffd_geometry(points, origin, extent, resolution, basis)
    if grad_field.shape != points.shape:
        raise ValueError("grad_field and points must have matching shapes")
    if need_points and control_displacements is None:
        raise ValueError("control_displacements is required for point gradients")
    if control_displacements is not None:
        _validate_ffd_lattice(control_displacements, points, resolution)

    grad_field_c = grad_field.contiguous()
    points_c = points.contiguous()
    origin_c = origin.contiguous()
    extent_c = extent.contiguous()
    lattice_c = (
        control_displacements.contiguous()
        if control_displacements is not None
        else None
    )
    batch, num_points, num_dims = points_c.shape
    lattice_size = math.prod(resolution)
    # The point-backward kernel writes every row, so an uninitialized
    # allocation is sufficient.
    grad_points = torch.empty_like(points_c) if need_points else None
    grad_lattice = (
        torch.zeros(
            (batch, lattice_size, num_dims), dtype=points.dtype, device=points.device
        )
        if need_control_displacements
        else None
    )

    if batch * num_points > 0 and (need_points or need_control_displacements):
        kernels = _ffd_kernels(points.dtype)
        basis_id = int(_BASIS_IDS[basis])
        resolution_t = _resolution_tensor(resolution, points.device)
        window_total = int(_ffd_window_size(resolution, basis))
        wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_field_c)
        with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
            if need_points:
                wp.launch(
                    kernels.point_backward,
                    dim=(batch, num_points),
                    inputs=[
                        _wp_view(points_c, kernels.warp_dtype),
                        _wp_view(lattice_c, kernels.warp_dtype),
                        _wp_view(origin_c, kernels.warp_dtype),
                        _wp_view(extent_c, kernels.warp_dtype),
                        _wp_view(resolution_t, wp.int32),
                        basis_id,
                        int(num_dims),
                        int(lattice_size),
                        window_total,
                        _wp_view(grad_field_c, kernels.warp_dtype),
                        _wp_view(grad_points, kernels.warp_dtype),
                    ],
                    device=wp_device,
                    stream=wp_stream,
                )
            if need_control_displacements:
                wp.launch(
                    kernels.backward,
                    dim=(batch, num_points, window_total),
                    inputs=[
                        _wp_view(points_c, kernels.warp_dtype),
                        _wp_view(origin_c, kernels.warp_dtype),
                        _wp_view(extent_c, kernels.warp_dtype),
                        _wp_view(resolution_t, wp.int32),
                        basis_id,
                        int(num_dims),
                        int(lattice_size),
                        window_total,
                        _wp_view(grad_field_c, kernels.warp_dtype),
                        _wp_view(grad_lattice, kernels.warp_dtype),
                    ],
                    device=wp_device,
                    stream=wp_stream,
                )
    return grad_points, grad_lattice


@ffd_field_warp_backward_impl.register_fake
def _ffd_field_warp_backward_fake(
    grad_field: Float[torch.Tensor, "batch num_points num_dims"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"] | None,
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: list[int],
    basis: str,
    need_points: bool = True,
    need_control_displacements: bool = True,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch lattice_nodes num_dims"] | None,
]:
    _ = grad_field, control_displacements, origin, extent, basis
    grad_lattice = (
        torch.empty(
            (points.shape[0], math.prod(resolution), points.shape[2]),
            dtype=points.dtype,
            device=points.device,
        )
        if need_control_displacements
        else None
    )
    return (
        _empty_contiguous_like(points) if need_points else None,
        grad_lattice,
    )


def _setup_ffd_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        Float[torch.Tensor, "batch num_points num_dims"],
        Float[torch.Tensor, "batch lattice_nodes num_dims"],
        Float[torch.Tensor, "batch num_dims"],
        Float[torch.Tensor, "batch num_dims"],
        list[int],
        str,
    ],
    output: Float[torch.Tensor, "batch num_points num_dims"],
) -> None:
    points, control_displacements, origin, extent, resolution, basis = inputs
    needs = ctx.needs_input_grad
    ctx.save_lattice_values = bool(needs[0])
    saved = [points.contiguous(), origin.contiguous(), extent.contiguous()]
    if ctx.save_lattice_values:
        saved.append(control_displacements.contiguous())
    ctx.save_for_backward(*saved)
    ctx.resolution = list(resolution)
    ctx.basis = basis


def _backward_ffd(
    ctx: torch.autograd.function.FunctionCtx,
    grad_field: Float[torch.Tensor, "batch num_points num_dims"] | None,
) -> tuple[Float[torch.Tensor, "..."] | None, ...]:
    needs = ctx.needs_input_grad
    if grad_field is None or not (needs[0] or needs[1]):
        return None, None, None, None, None, None

    saved = list(ctx.saved_tensors)
    points = saved.pop(0)
    origin = saved.pop(0)
    extent = saved.pop(0)
    control_displacements = saved.pop(0) if ctx.save_lattice_values else None
    grad_points, grad_lattice = ffd_field_warp_backward_impl(
        grad_field,
        points,
        control_displacements,
        origin,
        extent,
        ctx.resolution,
        ctx.basis,
        bool(needs[0]),
        bool(needs[1]),
    )
    return (
        grad_points if needs[0] else None,
        grad_lattice if needs[1] else None,
        None,
        None,
        None,
        None,
    )


ffd_field_warp_impl.register_autograd(_backward_ffd, setup_context=_setup_ffd_context)


def ffd_points_warp(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    control_displacements: Float[torch.Tensor, "batch lattice_nodes num_dims"],
    origin: Float[torch.Tensor, "batch num_dims"],
    extent: Float[torch.Tensor, "batch num_dims"],
    resolution: tuple[int, ...],
    basis: str,
    point_weights: Bool[torch.Tensor, "batch num_points"]
    | Float[torch.Tensor, "batch num_points"]
    | None,
) -> Float[torch.Tensor, "batch num_points num_dims"]:
    """Normalized rank-3 Warp lattice free-form deformation entry point."""
    if points.shape[0] == 0 or points.shape[1] == 0:
        # The point identity supplies its gradient directly. Connect every other
        # differentiable input through one scalar zero instead of paying for a
        # Warp launch on an empty query set.
        zero = _zero_dependency(control_displacements, point_weights)
        return points + zero

    points_c = points.contiguous()
    point_weights_c = point_weights.contiguous() if point_weights is not None else None
    field = ffd_field_warp_impl(
        points_c,
        control_displacements.contiguous(),
        origin.contiguous(),
        extent.contiguous(),
        list(resolution),
        basis,
    )
    # Only the lattice field warrants a Warp kernel. Native Torch handles the
    # final point weighting and addition without a dense Warp API.
    return displace_points_torch(points_c, field, point_weights_c)


__all__ = [
    "ffd_field_warp_backward_impl",
    "ffd_field_warp_impl",
    "ffd_points_warp",
]
