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

"""Torch custom-op integration for Warp deformation-energy kernels."""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch
import warp as wp
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from .energy_kernels import (
    HINGE_BENDING_F32,
    HINGE_BENDING_F64,
    SIMPLEX_INVERSION_F32,
    SIMPLEX_INVERSION_F64,
    SIMPLEX_MEASURE_F32,
    SIMPLEX_MEASURE_F64,
    SIMPLEX_STVK_F32,
    SIMPLEX_STVK_F64,
    VOLUME_CONTRIBUTIONS_F32,
    VOLUME_CONTRIBUTIONS_F64,
)


class _EnergyKernelFamily(NamedTuple):
    warp_dtype: object
    stvk: dict[tuple[int, int], object]
    measure: dict[tuple[int, int], object]
    inversion: dict[tuple[int, int], object]
    bending: object
    volume: object


_ENERGY_KERNELS = {
    torch.float32: _EnergyKernelFamily(
        wp.float32,
        SIMPLEX_STVK_F32,
        SIMPLEX_MEASURE_F32,
        SIMPLEX_INVERSION_F32,
        HINGE_BENDING_F32,
        VOLUME_CONTRIBUTIONS_F32,
    ),
    torch.float64: _EnergyKernelFamily(
        wp.float64,
        SIMPLEX_STVK_F64,
        SIMPLEX_MEASURE_F64,
        SIMPLEX_INVERSION_F64,
        HINGE_BENDING_F64,
        VOLUME_CONTRIBUTIONS_F64,
    ),
}


def _kernel_family(dtype: torch.dtype) -> _EnergyKernelFamily:
    try:
        return _ENERGY_KERNELS[dtype]
    except KeyError:
        raise TypeError(
            f"Warp deformation energies support float32 and float64, got {dtype}"
        ) from None


def _wp_view(
    tensor: Float[torch.Tensor, "..."] | Int[torch.Tensor, "..."],
    dtype,
):
    return wp.from_torch(
        tensor.detach(), dtype=dtype, return_ctype=True, requires_grad=False
    )


def _validate_coordinates(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
) -> None:
    _kernel_family(points.dtype)
    if points.ndim != 3 or reference_points.ndim != 3:
        raise ValueError(
            "points and reference_points must be normalized rank-3 tensors"
        )
    if points.shape != reference_points.shape:
        raise ValueError("points and reference_points must have identical shapes")
    if points.dtype != reference_points.dtype:
        raise TypeError("points and reference_points must have the same dtype")
    if points.device != reference_points.device:
        raise ValueError("points and reference_points must be on the same device")
    if points.shape[-1] not in (1, 2, 3):
        raise ValueError(
            "Warp deformation energies support coordinate dimensions 1, 2, and 3, "
            f"got D={points.shape[-1]}"
        )


def _validate_topology(
    topology: Int[torch.Tensor, "num_primitives vertices_per_primitive"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    *,
    name: str,
    width: int | None = None,
) -> None:
    if topology.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if topology.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64")
    if topology.device != points.device:
        raise ValueError(f"{name} and points must be on the same device")
    if width is not None and topology.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns")


def _validate_simplex_inputs(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    *,
    full_dimensional: bool = False,
) -> tuple[_EnergyKernelFamily, tuple[int, int]]:
    _validate_coordinates(points, reference_points)
    _validate_topology(simplices, points, name="simplices")
    coordinate_dimension = points.shape[-1]
    simplex_dimension = simplices.shape[1] - 1
    if simplex_dimension not in (1, 2, 3) or simplex_dimension > coordinate_dimension:
        raise ValueError(
            "simplices must have dimension 1, 2, or 3 not exceeding the "
            f"coordinate dimension, got m={simplex_dimension}, D={coordinate_dimension}"
        )
    if full_dimensional and simplex_dimension != coordinate_dimension:
        raise ValueError("simplex inversion energy requires full-dimensional simplices")
    return _kernel_family(points.dtype), (coordinate_dimension, simplex_dimension)


def _validate_output_gradient(
    gradient: Float[torch.Tensor, "..."],
    output: Float[torch.Tensor, "..."],
    name: str,
) -> Float[torch.Tensor, "..."]:
    if gradient.shape != output.shape:
        raise ValueError(f"{name} must have shape {tuple(output.shape)}")
    if gradient.dtype != output.dtype or gradient.device != output.device:
        raise ValueError(f"{name} must match the output dtype and device")
    return gradient.contiguous()


def _launch_forward(
    kernel,
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    topology: Int[torch.Tensor, "num_primitives vertices_per_primitive"],
    scalar_inputs: tuple[object, ...],
    outputs: tuple[Float[torch.Tensor, "batch num_primitives"], ...],
    detached_inputs: tuple[Float[torch.Tensor, "batch num_dims"], ...] = (),
) -> None:
    batch = points.shape[0]
    num_primitives = topology.shape[0]
    if batch * num_primitives == 0:
        return
    family = _kernel_family(points.dtype)
    device_scope = torch.cuda.device(points.device) if points.is_cuda else nullcontext()
    with device_scope:
        wp_device, wp_stream = FunctionSpec.warp_launch_context(points)
        with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
            wp.launch(
                kernel,
                dim=(batch, num_primitives),
                inputs=[
                    _wp_view(points, family.warp_dtype),
                    _wp_view(reference_points, family.warp_dtype),
                    _wp_view(topology, wp.int64),
                    int(points.shape[1]),
                    *(
                        _wp_view(tensor, family.warp_dtype)
                        for tensor in detached_inputs
                    ),
                    *scalar_inputs,
                ],
                outputs=[_wp_view(output, family.warp_dtype) for output in outputs],
                device=wp_device,
                stream=wp_stream,
                record_tape=False,
            )


def _launch_backward(
    kernel,
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    topology: Int[torch.Tensor, "num_primitives vertices_per_primitive"],
    scalar_inputs: tuple[object, ...],
    outputs: tuple[Float[torch.Tensor, "batch num_primitives"], ...],
    output_gradients: tuple[Float[torch.Tensor, "batch num_primitives"], ...],
    need_points: bool,
    need_reference_points: bool,
    detached_inputs: tuple[Float[torch.Tensor, "batch num_dims"], ...] = (),
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    grad_points = torch.zeros_like(points) if need_points else None
    grad_reference = (
        torch.zeros_like(reference_points) if need_reference_points else None
    )
    batch = points.shape[0]
    num_primitives = topology.shape[0]
    if batch * num_primitives == 0 or not (need_points or need_reference_points):
        return grad_points, grad_reference

    family = _kernel_family(points.dtype)
    scalar_adjoints = tuple(type(value)(0.0) for value in scalar_inputs)
    device_scope = torch.cuda.device(points.device) if points.is_cuda else nullcontext()
    with device_scope:
        wp_device, wp_stream = FunctionSpec.warp_launch_context(points)
        with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
            wp.launch(
                kernel,
                dim=(batch, num_primitives),
                inputs=[
                    _wp_view(points, family.warp_dtype),
                    _wp_view(reference_points, family.warp_dtype),
                    _wp_view(topology, wp.int64),
                    int(points.shape[1]),
                    *(
                        _wp_view(tensor, family.warp_dtype)
                        for tensor in detached_inputs
                    ),
                    *scalar_inputs,
                ],
                outputs=[_wp_view(output, family.warp_dtype) for output in outputs],
                adj_inputs=[
                    _wp_view(grad_points, family.warp_dtype)
                    if grad_points is not None
                    else None,
                    _wp_view(grad_reference, family.warp_dtype)
                    if grad_reference is not None
                    else None,
                    None,
                    0,
                    *(None for _ in detached_inputs),
                    *scalar_adjoints,
                ],
                adj_outputs=[
                    _wp_view(gradient, family.warp_dtype)
                    for gradient in output_gradients
                ],
                device=wp_device,
                stream=wp_stream,
                adjoint=True,
                record_tape=False,
            )
    return grad_points, grad_reference


@torch.library.custom_op(
    "physicsnemo::simplex_stvk_terms_warp_impl",
    mutates_args=(),
    schema=(
        "(Tensor points, Tensor reference_points, Tensor simplices, "
        "float lame_lambda, float shear_modulus) -> Tensor"
    ),
)
def simplex_stvk_terms_warp_impl(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    lame_lambda: float,
    shear_modulus: float,
) -> Float[torch.Tensor, "batch num_simplices"]:
    """Evaluate normalized StVK simplex terms with Warp."""

    family, dimensions = _validate_simplex_inputs(points, reference_points, simplices)
    points_c = points.contiguous()
    reference_c = reference_points.contiguous()
    simplices_c = simplices.contiguous()
    output = torch.empty(
        (points.shape[0], simplices.shape[0]),
        dtype=points.dtype,
        device=points.device,
    )
    scalars = (
        family.warp_dtype(lame_lambda),
        family.warp_dtype(shear_modulus),
    )
    _launch_forward(
        family.stvk[dimensions],
        points_c,
        reference_c,
        simplices_c,
        scalars,
        (output,),
    )
    return output


@simplex_stvk_terms_warp_impl.register_fake
def _simplex_stvk_fake(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    lame_lambda: float,
    shear_modulus: float,
) -> Float[torch.Tensor, "batch num_simplices"]:
    _ = reference_points, lame_lambda, shear_modulus
    return torch.empty(
        (points.shape[0], simplices.shape[0]), dtype=points.dtype, device=points.device
    )


@torch.library.custom_op(
    "physicsnemo::simplex_stvk_terms_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_terms, Tensor points, Tensor reference_points, "
        "Tensor simplices, Tensor terms, float lame_lambda, "
        "float shear_modulus, bool need_points, "
        "bool need_reference_points) -> (Tensor?, Tensor?)"
    ),
)
def simplex_stvk_terms_warp_backward_impl(
    grad_terms: Float[torch.Tensor, "batch num_simplices"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    terms: Float[torch.Tensor, "batch num_simplices"],
    lame_lambda: float,
    shear_modulus: float,
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    """Evaluate the first-order Warp pullback for StVK simplex terms."""

    family, dimensions = _validate_simplex_inputs(points, reference_points, simplices)
    grad_terms_c = _validate_output_gradient(grad_terms, terms, "grad_terms")
    scalars = (
        family.warp_dtype(lame_lambda),
        family.warp_dtype(shear_modulus),
    )
    return _launch_backward(
        family.stvk[dimensions],
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        scalars,
        (terms.contiguous(),),
        (grad_terms_c,),
        need_points,
        need_reference_points,
    )


@simplex_stvk_terms_warp_backward_impl.register_fake
def _simplex_stvk_backward_fake(
    grad_terms: Float[torch.Tensor, "batch num_simplices"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    terms: Float[torch.Tensor, "batch num_simplices"],
    lame_lambda: float,
    shear_modulus: float,
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    _ = grad_terms, simplices, terms, lame_lambda, shear_modulus
    return (
        torch.empty_like(points) if need_points else None,
        torch.empty_like(reference_points) if need_reference_points else None,
    )


def _setup_stvk_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        Float[torch.Tensor, "batch num_points num_dims"],
        Float[torch.Tensor, "batch num_points num_dims"],
        Int[torch.Tensor, "num_simplices vertices_per_simplex"],
        float,
        float,
    ],
    output: Float[torch.Tensor, "batch num_simplices"],
) -> None:
    points, reference_points, simplices, lame_lambda, shear_modulus = inputs
    ctx.save_for_backward(
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        output.contiguous(),
    )
    ctx.lame_lambda = lame_lambda
    ctx.shear_modulus = shear_modulus


def _backward_stvk(
    ctx: torch.autograd.function.FunctionCtx,
    grad_terms: Float[torch.Tensor, "batch num_simplices"] | None,
) -> tuple[Float[torch.Tensor, "batch num_points num_dims"] | None, ...]:
    needs = ctx.needs_input_grad
    if grad_terms is None or not (needs[0] or needs[1]):
        return None, None, None, None, None
    points, reference_points, simplices, terms = ctx.saved_tensors
    grad_points, grad_reference = simplex_stvk_terms_warp_backward_impl(
        grad_terms,
        points,
        reference_points,
        simplices,
        terms,
        ctx.lame_lambda,
        ctx.shear_modulus,
        bool(needs[0]),
        bool(needs[1]),
    )
    return grad_points, grad_reference, None, None, None


simplex_stvk_terms_warp_impl.register_autograd(
    _backward_stvk, setup_context=_setup_stvk_context
)


@torch.library.custom_op(
    "physicsnemo::simplex_measure_components_warp_impl",
    mutates_args=(),
    schema=(
        "(Tensor points, Tensor reference_points, Tensor simplices) -> (Tensor, Tensor)"
    ),
)
def simplex_measure_components_warp_impl(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
) -> tuple[
    Float[torch.Tensor, "batch num_simplices"],
    Float[torch.Tensor, "batch num_simplices"],
]:
    """Evaluate normalized simplex measure components with Warp."""

    family, dimensions = _validate_simplex_inputs(points, reference_points, simplices)
    shape = (points.shape[0], simplices.shape[0])
    ratio = torch.empty(shape, dtype=points.dtype, device=points.device)
    reference_measure = torch.empty_like(ratio)
    _launch_forward(
        family.measure[dimensions],
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        (),
        (ratio, reference_measure),
    )
    return ratio, reference_measure


@simplex_measure_components_warp_impl.register_fake
def _simplex_measure_fake(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
) -> tuple[
    Float[torch.Tensor, "batch num_simplices"],
    Float[torch.Tensor, "batch num_simplices"],
]:
    _ = reference_points
    ratio = torch.empty(
        (points.shape[0], simplices.shape[0]), dtype=points.dtype, device=points.device
    )
    return ratio, torch.empty_like(ratio)


@torch.library.custom_op(
    "physicsnemo::simplex_measure_components_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_ratio, Tensor grad_reference_measure, Tensor points, "
        "Tensor reference_points, Tensor simplices, Tensor ratio, "
        "Tensor reference_measure, bool need_points, "
        "bool need_reference_points) -> (Tensor?, Tensor?)"
    ),
)
def simplex_measure_components_warp_backward_impl(
    grad_ratio: Float[torch.Tensor, "batch num_simplices"],
    grad_reference_measure: Float[torch.Tensor, "batch num_simplices"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    ratio: Float[torch.Tensor, "batch num_simplices"],
    reference_measure: Float[torch.Tensor, "batch num_simplices"],
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    """Evaluate the first-order Warp pullback for measure components."""

    family, dimensions = _validate_simplex_inputs(points, reference_points, simplices)
    return _launch_backward(
        family.measure[dimensions],
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        (),
        (ratio.contiguous(), reference_measure.contiguous()),
        (
            _validate_output_gradient(grad_ratio, ratio, "grad_ratio"),
            _validate_output_gradient(
                grad_reference_measure,
                reference_measure,
                "grad_reference_measure",
            ),
        ),
        need_points,
        need_reference_points,
    )


@simplex_measure_components_warp_backward_impl.register_fake
def _simplex_measure_backward_fake(
    grad_ratio: Float[torch.Tensor, "batch num_simplices"],
    grad_reference_measure: Float[torch.Tensor, "batch num_simplices"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    ratio: Float[torch.Tensor, "batch num_simplices"],
    reference_measure: Float[torch.Tensor, "batch num_simplices"],
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    _ = grad_ratio, grad_reference_measure, simplices, ratio, reference_measure
    return (
        torch.empty_like(points) if need_points else None,
        torch.empty_like(reference_points) if need_reference_points else None,
    )


def _setup_measure_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        Float[torch.Tensor, "batch num_points num_dims"],
        Float[torch.Tensor, "batch num_points num_dims"],
        Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    ],
    output: tuple[
        Float[torch.Tensor, "batch num_simplices"],
        Float[torch.Tensor, "batch num_simplices"],
    ],
) -> None:
    points, reference_points, simplices = inputs
    ratio, reference_measure = output
    ctx.save_for_backward(
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        ratio.contiguous(),
        reference_measure.contiguous(),
    )


def _backward_measure(
    ctx: torch.autograd.function.FunctionCtx,
    grad_ratio: Float[torch.Tensor, "batch num_simplices"] | None,
    grad_reference_measure: Float[torch.Tensor, "batch num_simplices"] | None,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    None,
]:
    needs = ctx.needs_input_grad
    if not (needs[0] or needs[1]):
        return None, None, None
    points, reference_points, simplices, ratio, reference_measure = ctx.saved_tensors
    if grad_ratio is None:
        grad_ratio = torch.zeros_like(ratio)
    if grad_reference_measure is None:
        grad_reference_measure = torch.zeros_like(reference_measure)
    grad_points, grad_reference = simplex_measure_components_warp_backward_impl(
        grad_ratio,
        grad_reference_measure,
        points,
        reference_points,
        simplices,
        ratio,
        reference_measure,
        bool(needs[0]),
        bool(needs[1]),
    )
    return grad_points, grad_reference, None


simplex_measure_components_warp_impl.register_autograd(
    _backward_measure, setup_context=_setup_measure_context
)


@torch.library.custom_op(
    "physicsnemo::simplex_inversion_terms_warp_impl",
    mutates_args=(),
    schema=(
        "(Tensor points, Tensor reference_points, Tensor simplices, "
        "float minimum_jacobian) -> Tensor"
    ),
)
def simplex_inversion_terms_warp_impl(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    minimum_jacobian: float,
) -> Float[torch.Tensor, "batch num_simplices"]:
    """Evaluate normalized simplex inversion terms with Warp."""

    family, dimensions = _validate_simplex_inputs(
        points, reference_points, simplices, full_dimensional=True
    )
    terms = torch.empty(
        (points.shape[0], simplices.shape[0]),
        dtype=points.dtype,
        device=points.device,
    )
    scalars = (family.warp_dtype(minimum_jacobian),)
    _launch_forward(
        family.inversion[dimensions],
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        scalars,
        (terms,),
    )
    return terms


@simplex_inversion_terms_warp_impl.register_fake
def _simplex_inversion_fake(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    minimum_jacobian: float,
) -> Float[torch.Tensor, "batch num_simplices"]:
    _ = reference_points, minimum_jacobian
    return torch.empty(
        (points.shape[0], simplices.shape[0]), dtype=points.dtype, device=points.device
    )


@torch.library.custom_op(
    "physicsnemo::simplex_inversion_terms_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_terms, Tensor points, Tensor reference_points, "
        "Tensor simplices, Tensor terms, float minimum_jacobian, "
        "bool need_points, bool need_reference_points) -> (Tensor?, Tensor?)"
    ),
)
def simplex_inversion_terms_warp_backward_impl(
    grad_terms: Float[torch.Tensor, "batch num_simplices"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    terms: Float[torch.Tensor, "batch num_simplices"],
    minimum_jacobian: float,
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    """Evaluate the first-order Warp pullback for inversion terms."""

    family, dimensions = _validate_simplex_inputs(
        points, reference_points, simplices, full_dimensional=True
    )
    return _launch_backward(
        family.inversion[dimensions],
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        (family.warp_dtype(minimum_jacobian),),
        (terms.contiguous(),),
        (_validate_output_gradient(grad_terms, terms, "grad_terms"),),
        need_points,
        need_reference_points,
    )


@simplex_inversion_terms_warp_backward_impl.register_fake
def _simplex_inversion_backward_fake(
    grad_terms: Float[torch.Tensor, "batch num_simplices"],
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    terms: Float[torch.Tensor, "batch num_simplices"],
    minimum_jacobian: float,
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points num_dims"] | None,
    Float[torch.Tensor, "batch num_points num_dims"] | None,
]:
    _ = grad_terms, simplices, terms, minimum_jacobian
    return (
        torch.empty_like(points) if need_points else None,
        torch.empty_like(reference_points) if need_reference_points else None,
    )


def _setup_inversion_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        Float[torch.Tensor, "batch num_points num_dims"],
        Float[torch.Tensor, "batch num_points num_dims"],
        Int[torch.Tensor, "num_simplices vertices_per_simplex"],
        float,
    ],
    output: Float[torch.Tensor, "batch num_simplices"],
) -> None:
    points, reference_points, simplices, minimum_jacobian = inputs
    ctx.save_for_backward(
        points.contiguous(),
        reference_points.contiguous(),
        simplices.contiguous(),
        output.contiguous(),
    )
    ctx.minimum_jacobian = minimum_jacobian


def _backward_inversion(
    ctx: torch.autograd.function.FunctionCtx,
    grad_terms: Float[torch.Tensor, "batch num_simplices"] | None,
) -> tuple[Float[torch.Tensor, "batch num_points num_dims"] | None, ...]:
    needs = ctx.needs_input_grad
    if grad_terms is None or not (needs[0] or needs[1]):
        return None, None, None, None
    points, reference_points, simplices, terms = ctx.saved_tensors
    grad_points, grad_reference = simplex_inversion_terms_warp_backward_impl(
        grad_terms,
        points,
        reference_points,
        simplices,
        terms,
        ctx.minimum_jacobian,
        bool(needs[0]),
        bool(needs[1]),
    )
    return grad_points, grad_reference, None, None


simplex_inversion_terms_warp_impl.register_autograd(
    _backward_inversion, setup_context=_setup_inversion_context
)


def _validate_hinge_inputs(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
) -> _EnergyKernelFamily:
    _validate_coordinates(points, reference_points)
    if points.shape[-1] != 3:
        raise ValueError("hinge bending requires three-dimensional coordinates")
    _validate_topology(hinges, points, name="hinges", width=4)
    return _kernel_family(points.dtype)


@torch.library.custom_op(
    "physicsnemo::hinge_bending_terms_warp_impl",
    mutates_args=(),
    schema="(Tensor points, Tensor reference_points, Tensor hinges) -> Tensor",
)
def hinge_bending_terms_warp_impl(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
) -> Float[torch.Tensor, "batch num_hinges"]:
    """Evaluate normalized hinge-bending terms with Warp."""

    family = _validate_hinge_inputs(points, reference_points, hinges)
    terms = torch.empty(
        (points.shape[0], hinges.shape[0]), dtype=points.dtype, device=points.device
    )
    _launch_forward(
        family.bending,
        points.contiguous(),
        reference_points.contiguous(),
        hinges.contiguous(),
        (),
        (terms,),
    )
    return terms


@hinge_bending_terms_warp_impl.register_fake
def _hinge_bending_fake(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
) -> Float[torch.Tensor, "batch num_hinges"]:
    _ = reference_points
    return torch.empty(
        (points.shape[0], hinges.shape[0]), dtype=points.dtype, device=points.device
    )


@torch.library.custom_op(
    "physicsnemo::hinge_bending_terms_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_terms, Tensor points, Tensor reference_points, "
        "Tensor hinges, Tensor terms, bool need_points, "
        "bool need_reference_points) -> (Tensor?, Tensor?)"
    ),
)
def hinge_bending_terms_warp_backward_impl(
    grad_terms: Float[torch.Tensor, "batch num_hinges"],
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
    terms: Float[torch.Tensor, "batch num_hinges"],
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"] | None,
    Float[torch.Tensor, "batch num_points 3"] | None,
]:
    """Evaluate the first-order Warp pullback for hinge-bending terms."""

    family = _validate_hinge_inputs(points, reference_points, hinges)
    return _launch_backward(
        family.bending,
        points.contiguous(),
        reference_points.contiguous(),
        hinges.contiguous(),
        (),
        (terms.contiguous(),),
        (_validate_output_gradient(grad_terms, terms, "grad_terms"),),
        need_points,
        need_reference_points,
    )


@hinge_bending_terms_warp_backward_impl.register_fake
def _hinge_bending_backward_fake(
    grad_terms: Float[torch.Tensor, "batch num_hinges"],
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
    terms: Float[torch.Tensor, "batch num_hinges"],
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"] | None,
    Float[torch.Tensor, "batch num_points 3"] | None,
]:
    _ = grad_terms, hinges, terms
    return (
        torch.empty_like(points) if need_points else None,
        torch.empty_like(reference_points) if need_reference_points else None,
    )


def _setup_hinge_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        Float[torch.Tensor, "batch num_points 3"],
        Float[torch.Tensor, "batch num_points 3"],
        Int[torch.Tensor, "num_hinges 4"],
    ],
    output: Float[torch.Tensor, "batch num_hinges"],
) -> None:
    points, reference_points, hinges = inputs
    ctx.save_for_backward(
        points.contiguous(),
        reference_points.contiguous(),
        hinges.contiguous(),
        output.contiguous(),
    )


def _backward_hinge(
    ctx: torch.autograd.function.FunctionCtx,
    grad_terms: Float[torch.Tensor, "batch num_hinges"] | None,
) -> tuple[Float[torch.Tensor, "batch num_points 3"] | None, ...]:
    needs = ctx.needs_input_grad
    if grad_terms is None or not (needs[0] or needs[1]):
        return None, None, None
    points, reference_points, hinges, terms = ctx.saved_tensors
    grad_points, grad_reference = hinge_bending_terms_warp_backward_impl(
        grad_terms,
        points,
        reference_points,
        hinges,
        terms,
        bool(needs[0]),
        bool(needs[1]),
    )
    return grad_points, grad_reference, None


hinge_bending_terms_warp_impl.register_autograd(
    _backward_hinge, setup_context=_setup_hinge_context
)


def _validate_volume_inputs(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> _EnergyKernelFamily:
    _validate_coordinates(points, reference_points)
    if points.shape[-1] != 3:
        raise ValueError("closed-surface volume requires three-dimensional coordinates")
    _validate_topology(triangles, points, name="triangles", width=3)
    return _kernel_family(points.dtype)


def _volume_origins(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> tuple[
    Float[torch.Tensor, "batch 3"],
    Float[torch.Tensor, "batch 3"],
]:
    """Return detached, validly indexable common origins for volume kernels."""

    if points.shape[1] == 0 or triangles.shape[0] == 0:
        shape = (points.shape[0], 3)
        return points.new_zeros(shape), reference_points.new_zeros(shape)
    origin_index = triangles[0, 0].clamp(0, points.shape[1] - 1).reshape(1)
    return (
        points.index_select(1, origin_index).squeeze(1).detach().contiguous(),
        reference_points.index_select(1, origin_index).squeeze(1).detach().contiguous(),
    )


@torch.library.custom_op(
    "physicsnemo::closed_surface_volume_contributions_warp_impl",
    mutates_args=(),
    schema=(
        "(Tensor points, Tensor reference_points, Tensor triangles) -> (Tensor, Tensor)"
    ),
)
def closed_surface_volume_contributions_warp_impl(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> tuple[
    Float[torch.Tensor, "batch num_triangles"],
    Float[torch.Tensor, "batch num_triangles"],
]:
    """Evaluate normalized closed-surface volume contributions with Warp."""

    family = _validate_volume_inputs(points, reference_points, triangles)
    current = torch.empty(
        (points.shape[0], triangles.shape[0]),
        dtype=points.dtype,
        device=points.device,
    )
    reference = torch.empty_like(current)
    origins = _volume_origins(points, reference_points, triangles)
    _launch_forward(
        family.volume,
        points.contiguous(),
        reference_points.contiguous(),
        triangles.contiguous(),
        (),
        (current, reference),
        detached_inputs=origins,
    )
    return current, reference


@closed_surface_volume_contributions_warp_impl.register_fake
def _volume_fake(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> tuple[
    Float[torch.Tensor, "batch num_triangles"],
    Float[torch.Tensor, "batch num_triangles"],
]:
    _ = reference_points
    current = torch.empty(
        (points.shape[0], triangles.shape[0]),
        dtype=points.dtype,
        device=points.device,
    )
    return current, torch.empty_like(current)


@torch.library.custom_op(
    "physicsnemo::closed_surface_volume_contributions_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor grad_current, Tensor grad_reference, Tensor points, "
        "Tensor reference_points, Tensor triangles, Tensor current, "
        "Tensor reference, bool need_points, bool need_reference_points) "
        "-> (Tensor?, Tensor?)"
    ),
)
def closed_surface_volume_contributions_warp_backward_impl(
    grad_current: Float[torch.Tensor, "batch num_triangles"],
    grad_reference: Float[torch.Tensor, "batch num_triangles"],
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
    current: Float[torch.Tensor, "batch num_triangles"],
    reference: Float[torch.Tensor, "batch num_triangles"],
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"] | None,
    Float[torch.Tensor, "batch num_points 3"] | None,
]:
    """Evaluate the Warp pullback for closed-surface volume contributions."""

    family = _validate_volume_inputs(points, reference_points, triangles)
    origins = _volume_origins(points, reference_points, triangles)
    return _launch_backward(
        family.volume,
        points.contiguous(),
        reference_points.contiguous(),
        triangles.contiguous(),
        (),
        (current.contiguous(), reference.contiguous()),
        (
            _validate_output_gradient(grad_current, current, "grad_current"),
            _validate_output_gradient(grad_reference, reference, "grad_reference"),
        ),
        need_points,
        need_reference_points,
        detached_inputs=origins,
    )


@closed_surface_volume_contributions_warp_backward_impl.register_fake
def _volume_backward_fake(
    grad_current: Float[torch.Tensor, "batch num_triangles"],
    grad_reference: Float[torch.Tensor, "batch num_triangles"],
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
    current: Float[torch.Tensor, "batch num_triangles"],
    reference: Float[torch.Tensor, "batch num_triangles"],
    need_points: bool,
    need_reference_points: bool,
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"] | None,
    Float[torch.Tensor, "batch num_points 3"] | None,
]:
    _ = grad_current, grad_reference, triangles, current, reference
    return (
        torch.empty_like(points) if need_points else None,
        torch.empty_like(reference_points) if need_reference_points else None,
    )


def _setup_volume_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[
        Float[torch.Tensor, "batch num_points 3"],
        Float[torch.Tensor, "batch num_points 3"],
        Int[torch.Tensor, "num_triangles 3"],
    ],
    output: tuple[
        Float[torch.Tensor, "batch num_triangles"],
        Float[torch.Tensor, "batch num_triangles"],
    ],
) -> None:
    points, reference_points, triangles = inputs
    current, reference = output
    ctx.save_for_backward(
        points.contiguous(),
        reference_points.contiguous(),
        triangles.contiguous(),
        current.contiguous(),
        reference.contiguous(),
    )


def _backward_volume(
    ctx: torch.autograd.function.FunctionCtx,
    grad_current: Float[torch.Tensor, "batch num_triangles"] | None,
    grad_reference: Float[torch.Tensor, "batch num_triangles"] | None,
) -> tuple[
    Float[torch.Tensor, "batch num_points 3"] | None,
    Float[torch.Tensor, "batch num_points 3"] | None,
    None,
]:
    needs = ctx.needs_input_grad
    if not (needs[0] or needs[1]):
        return None, None, None
    points, reference_points, triangles, current, reference = ctx.saved_tensors
    if grad_current is None:
        grad_current = torch.zeros_like(current)
    if grad_reference is None:
        grad_reference = torch.zeros_like(reference)
    grad_points, grad_reference_points = (
        closed_surface_volume_contributions_warp_backward_impl(
            grad_current,
            grad_reference,
            points,
            reference_points,
            triangles,
            current,
            reference,
            bool(needs[0]),
            bool(needs[1]),
        )
    )
    return grad_points, grad_reference_points, None


closed_surface_volume_contributions_warp_impl.register_autograd(
    _backward_volume, setup_context=_setup_volume_context
)


def simplex_stvk_terms_warp(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    lame_lambda: float,
    shear_modulus: float,
) -> Float[torch.Tensor, "batch num_simplices"]:
    """Return StVK simplex terms from the Warp custom operator."""

    return simplex_stvk_terms_warp_impl(
        points, reference_points, simplices, lame_lambda, shear_modulus
    )


def simplex_measure_components_warp(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
) -> tuple[
    Float[torch.Tensor, "batch num_simplices"],
    Float[torch.Tensor, "batch num_simplices"],
]:
    """Return simplex measure components from the Warp custom operator."""

    return simplex_measure_components_warp_impl(points, reference_points, simplices)


def simplex_inversion_terms_warp(
    points: Float[torch.Tensor, "batch num_points num_dims"],
    reference_points: Float[torch.Tensor, "batch num_points num_dims"],
    simplices: Int[torch.Tensor, "num_simplices vertices_per_simplex"],
    minimum_jacobian: float,
) -> Float[torch.Tensor, "batch num_simplices"]:
    """Return simplex inversion terms from the Warp custom operator."""

    return simplex_inversion_terms_warp_impl(
        points, reference_points, simplices, minimum_jacobian
    )


def hinge_bending_terms_warp(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    hinges: Int[torch.Tensor, "num_hinges 4"],
) -> Float[torch.Tensor, "batch num_hinges"]:
    """Return hinge-bending terms from the Warp custom operator."""

    return hinge_bending_terms_warp_impl(points, reference_points, hinges)


def closed_surface_volume_contributions_warp(
    points: Float[torch.Tensor, "batch num_points 3"],
    reference_points: Float[torch.Tensor, "batch num_points 3"],
    triangles: Int[torch.Tensor, "num_triangles 3"],
) -> tuple[
    Float[torch.Tensor, "batch num_triangles"],
    Float[torch.Tensor, "batch num_triangles"],
]:
    """Return volume contributions from the Warp custom operator."""

    return closed_surface_volume_contributions_warp_impl(
        points, reference_points, triangles
    )


__all__ = [
    "closed_surface_volume_contributions_warp",
    "hinge_bending_terms_warp",
    "simplex_inversion_terms_warp",
    "simplex_measure_components_warp",
    "simplex_stvk_terms_warp",
]
