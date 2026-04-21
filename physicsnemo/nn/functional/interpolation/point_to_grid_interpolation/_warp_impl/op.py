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

"""Torch custom-op registration for Warp point-to-grid interpolation.

Kernel Summary
==============

| Kernel Group | Purpose |
|---|---|
| ``kernels.py`` | Consolidated forward/backward kernels grouped by dimensionality and stencil width. |
| ``launch_forward`` | Forward launch helpers that choose stride-specific kernels by dimensionality. |
| ``launch_backward`` | Backward dispatcher that routes to manual per-dimension launch helpers. |
"""

import warnings
from typing import List, Tuple

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.interpolation._warp_common import (
    _INTERP_ID_TO_STRIDE,
    _INTERP_NAME_TO_ID,
    interpolation_geometry,
    parse_grid_metadata,
)

from .launch_backward import launch_backward
from .launch_forward import launch_forward


# Register the warp-backed point-to-grid op with torch custom ops.
@torch.library.custom_op(
    "physicsnemo::point_to_grid_interpolation_warp", mutates_args=()
)
def point_to_grid_interpolation_impl(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    # Keep signature parity with the torch implementation API.
    _ = mem_speed_trade

    # Validate device contract.
    if query_points.device != point_values.device:
        raise ValueError("query_points and point_values must be on the same device")

    # Parse grid metadata and normalize query-point/value shapes.
    grid = parse_grid_metadata(grid_meta, op_name="warp point-to-grid interpolation")
    dims = len(grid)
    if query_points.ndim == 1 and dims == 1:
        query_points = query_points.unsqueeze(-1)
    if query_points.ndim != 2 or query_points.shape[-1] != dims:
        raise ValueError(
            f"query_points must have shape (num_points, {dims}), got {tuple(query_points.shape)}"
        )
    if point_values.ndim == 1:
        point_values = point_values.unsqueeze(-1)
    if point_values.ndim != 2:
        raise ValueError(
            "point_values must have shape (num_points, channels) or (num_points,)"
        )
    if query_points.shape[0] != point_values.shape[0]:
        raise ValueError(
            "query_points and point_values must have the same leading dimension"
        )

    # Build launch geometry.
    stride = _INTERP_ID_TO_STRIDE.get(interp_id)
    if stride is None:
        raise ValueError(f"Unsupported interpolation id {interp_id}")
    start_vals, dx_vals, sizes, center_offset = interpolation_geometry(
        grid, stride, pad_grid=False
    )

    # Normalize to float32 for warp kernels and keep original dtype for output cast.
    input_dtype = point_values.dtype
    query_fp32 = (
        query_points
        if query_points.dtype == torch.float32
        else query_points.to(torch.float32)
    )
    values_fp32 = (
        point_values
        if point_values.dtype == torch.float32
        else point_values.to(torch.float32)
    )

    # Allocate output tensor.
    output = torch.zeros(
        (values_fp32.shape[1], *sizes),
        device=query_fp32.device,
        dtype=torch.float32,
    )

    # Resolve Warp device/stream from torch inputs and launch.
    wp_device, wp_stream = FunctionSpec.warp_launch_context(query_fp32)
    with wp.ScopedStream(wp_stream):
        launch_forward(
            query_points=query_fp32,
            point_values=values_fp32,
            out_grid=output,
            dims=dims,
            start_vals=start_vals,
            dx_vals=dx_vals,
            sizes=sizes,
            center_offset=center_offset,
            interp_id=interp_id,
            stride=stride,
            num_points=query_fp32.shape[0],
            wp_device=wp_device,
            wp_stream=wp_stream,
        )

    # Cast outputs back to input dtype for API consistency.
    if input_dtype != torch.float32:
        output = output.to(input_dtype)
    return output


# Register fake tensor propagation for torch compile/fake mode.
@point_to_grid_interpolation_impl.register_fake
def _(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    _ = (query_points, interp_id, mem_speed_trade)
    channels = point_values.shape[1] if point_values.ndim == 2 else 1
    dims = int(grid_meta.shape[0]) if grid_meta.ndim == 2 else 1
    sizes = [1] * dims
    if (
        grid_meta.device.type != "meta"
        and grid_meta.ndim == 2
        and grid_meta.shape[1] == 3
    ):
        try:
            sizes = [int(v) for v in grid_meta[:, 2].tolist()]
        except Exception:
            sizes = [1] * dims
    return torch.empty(
        (channels, *sizes),
        device=point_values.device,
        dtype=point_values.dtype,
    )


# Setup tensors and metadata required for custom-op backward.
def setup_point_to_grid_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    _ = output
    query_points, point_values, grid_meta, interp_id, mem_speed_trade = inputs
    _ = mem_speed_trade
    ctx.save_for_backward(query_points, point_values, grid_meta)
    ctx.interp_id = int(interp_id)


# Compute backward with manual Warp kernels.
def backward_point_to_grid(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
    query_points, point_values, grid_meta = ctx.saved_tensors
    if grad_output is None:
        return None, None, None, None, None
    grad_query, grad_values = launch_backward(
        query_points=query_points,
        point_values=point_values,
        grid_meta=grid_meta,
        interp_id=ctx.interp_id,
        grad_grid_output=grad_output,
        needs_input_grad=ctx.needs_input_grad,
    )
    return grad_query, grad_values, None, None, None


# Register custom-op backward.
point_to_grid_interpolation_impl.register_autograd(
    backward_point_to_grid, setup_context=setup_point_to_grid_context
)


# Public warp entry point used by the point-to-grid FunctionSpec.
def point_to_grid_interpolation_warp(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str = "smooth_step_2",
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    if query_points.dtype != torch.float32:
        raise TypeError("query_points must be float32")
    if point_values.dtype != torch.float32:
        raise TypeError("point_values must be float32")
    if not mem_speed_trade:
        warnings.warn(
            "The Warp backend ignores mem_speed_trade and always runs the same kernel path.",
            UserWarning,
            stacklevel=2,
        )

    interp_id = _INTERP_NAME_TO_ID.get(interpolation_type)
    if interp_id is None:
        raise ValueError(
            "interpolation_type must be one of "
            f"{list(_INTERP_NAME_TO_ID)}, got {interpolation_type}"
        )

    grid_meta = torch.tensor(grid, dtype=torch.float32, device="cpu")
    return point_to_grid_interpolation_impl(
        query_points,
        point_values,
        grid_meta,
        int(interp_id),
        mem_speed_trade,
    )


__all__ = ["point_to_grid_interpolation_warp"]
