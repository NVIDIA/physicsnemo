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

from physicsnemo.core.function_spec import FunctionSpec

from ...uniform_grid_gradient._warp_impl.utils import _normalize_spacing
from ..utils import validate_vector_field
from .launch_backward import _launch_backward
from .launch_forward import _launch_forward

_SUPPORTED_ORDERS = (2, 4)


def _validate_order(order: int) -> int:
    if not isinstance(order, int):
        raise TypeError(f"order must be an integer, got {type(order)}")
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            "uniform_grid_curl supports central orders "
            f"{list(_SUPPORTED_ORDERS)}, got order={order}"
        )
    return order


def _validate_positive_spacing(spacing_tuple: tuple[float, ...]) -> None:
    for dx in spacing_tuple:
        if dx <= 0.0:
            raise ValueError("all spacing entries must be strictly positive")


def _to_fp32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=torch.float32).contiguous()


def _restore_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


@torch.library.custom_op("physicsnemo::uniform_grid_curl_warp_impl", mutates_args=())
def uniform_grid_curl_impl(
    vector_field: torch.Tensor,
    spacing_meta: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Evaluate uniform-grid curl with fused Warp kernels."""
    grid_ndim = validate_vector_field(vector_field)
    spacing_tuple = tuple(float(v) for v in spacing_meta.tolist())
    _validate_positive_spacing(spacing_tuple)
    order = _validate_order(int(order))
    orig_dtype = vector_field.dtype
    vector_field_fp32 = _to_fp32_contiguous(vector_field)
    output_shape = (
        vector_field_fp32.shape[1:] if grid_ndim == 2 else vector_field_fp32.shape
    )
    output_fp32 = torch.empty(
        output_shape,
        device=vector_field_fp32.device,
        dtype=torch.float32,
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(vector_field_fp32)
    _launch_forward(
        vector_field_fp32=vector_field_fp32,
        spacing_tuple=spacing_tuple[:grid_ndim],
        order=order,
        output_fp32=output_fp32,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )
    return _restore_dtype(output_fp32, orig_dtype)


@uniform_grid_curl_impl.register_fake
def _uniform_grid_curl_impl_fake(
    vector_field: torch.Tensor,
    spacing_meta: torch.Tensor,
    order: int,
) -> torch.Tensor:
    _ = (spacing_meta, order)
    output_shape = (
        vector_field.shape[1:] if vector_field.ndim == 3 else vector_field.shape
    )
    return torch.empty(
        output_shape,
        device=vector_field.device,
        dtype=vector_field.dtype,
    )


def setup_uniform_grid_curl_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    """Save uniform-grid curl metadata for the backward pass."""
    vector_field, spacing_meta, order = inputs
    _ = output
    ctx.spacing_tuple = tuple(float(v) for v in spacing_meta.tolist())
    ctx.order = int(order)
    ctx.orig_dtype = vector_field.dtype
    ctx.grid_ndim = vector_field.ndim - 1


def backward_uniform_grid_curl(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, None, None]:
    if grad_output is None or not ctx.needs_input_grad[0]:
        return None, None, None
    grad_output_fp32 = _to_fp32_contiguous(grad_output)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    if ctx.grid_ndim == 2:
        grad_vector_fp32 = torch.empty(
            (2, *grad_output_fp32.shape),
            device=grad_output_fp32.device,
            dtype=torch.float32,
        )
        _launch_backward(
            grad_output_fp32=grad_output_fp32,
            spacing_tuple=ctx.spacing_tuple[:2],
            order=ctx.order,
            grad_vector_fp32=grad_vector_fp32,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
    else:
        grad_vector_fp32 = torch.empty_like(grad_output_fp32)
        _launch_forward(
            vector_field_fp32=grad_output_fp32,
            spacing_tuple=ctx.spacing_tuple[:3],
            order=ctx.order,
            output_fp32=grad_vector_fp32,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
    return _restore_dtype(grad_vector_fp32, ctx.orig_dtype), None, None


uniform_grid_curl_impl.register_autograd(
    backward_uniform_grid_curl,
    setup_context=setup_uniform_grid_curl_context,
)


def uniform_grid_curl_warp(
    vector_field: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    order: int = 2,
) -> torch.Tensor:
    """Compute periodic uniform-grid curl with a fused Warp custom op."""
    grid_ndim = vector_field.ndim - 1
    spacing_tuple = _normalize_spacing(spacing, grid_ndim)
    spacing_meta = torch.tensor(spacing_tuple, dtype=torch.float32, device="cpu")
    return uniform_grid_curl_impl(vector_field, spacing_meta, _validate_order(order))
