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

"""Torch custom-op registration for Warp grid-to-point interpolation.

Kernel Summary
==============

| Kernel Group | Purpose |
|---|---|
| ``kernels.py`` | Consolidated forward/backward kernels grouped by dimensionality and stencil width. |
| ``launch_forward`` | Forward launch helpers that choose a stride-specific kernel for each dimensionality. |
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
    pad_grid_for_stride,
    parse_grid_metadata,
)

from .launch_backward import launch_backward
from .launch_forward import launch_forward


# Register the warp-backed interpolation op with torch custom ops.
@torch.library.custom_op("physicsnemo::interpolation_warp", mutates_args=())
def interpolation_impl(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    # Keep signature parity with the torch implementation API.
    _ = mem_speed_trade

    # Validate device contract.
    if query_points.device != context_grid.device:
        raise ValueError("query_points and context_grid must be on the same device")

    # Parse grid metadata and normalize query-point shape to (N, dims).
    grid = parse_grid_metadata(grid_meta, op_name="warp interpolation")
    dims = len(grid)
    if query_points.ndim == 1 and dims == 1:
        query_points = query_points.unsqueeze(-1)
    if query_points.shape[-1] != dims:
        raise ValueError(
            f"query_points must have last dimension {dims}, got {query_points.shape}"
        )
    num_points = query_points.shape[0]

    # Validate context-grid spatial dimensions against grid metadata.
    grid_sizes = [g[2] for g in grid]
    if list(context_grid.shape[1:]) != grid_sizes:
        raise ValueError(
            "context_grid shape must match grid sizes: "
            f"expected {grid_sizes}, got {list(context_grid.shape[1:])}"
        )

    # Prepare padded input grid and launch geometry.
    stride = _INTERP_ID_TO_STRIDE.get(interp_id)
    if stride is None:
        raise ValueError(f"Unsupported interpolation id {interp_id}")
    context_grid, _ = pad_grid_for_stride(context_grid, dims, stride)
    start_vals, dx_vals, padded_sizes, center_offset = interpolation_geometry(
        grid, stride, pad_grid=True
    )

    # Normalize to float32 for warp kernels and keep original dtype for output cast.
    input_dtype = context_grid.dtype
    if input_dtype != torch.float32:
        context_grid = context_grid.to(torch.float32)
    if query_points.dtype != torch.float32:
        query_points = query_points.to(torch.float32)

    # Allocate output tensor.
    output = torch.empty(
        (num_points, context_grid.shape[0]),
        device=query_points.device,
        dtype=torch.float32,
    )

    # Resolve Warp device/stream from torch inputs.
    wp_device, wp_stream = FunctionSpec.warp_launch_context(query_points)

    # Launch the specialized interpolation kernel for the input dimensionality.
    with wp.ScopedStream(wp_stream):
        launch_forward(
            query_points=query_points,
            context_grid=context_grid,
            output=output,
            dims=dims,
            start_vals=start_vals,
            dx_vals=dx_vals,
            padded_sizes=padded_sizes,
            center_offset=center_offset,
            interp_id=interp_id,
            stride=stride,
            num_points=num_points,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )

    # Cast outputs back to the input grid dtype for API consistency.
    if input_dtype != torch.float32:
        output = output.to(input_dtype)
    return output


# Register fake tensor propagation for torch compile/fake mode.
@interpolation_impl.register_fake
def _(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    return torch.empty(
        query_points.shape[0],
        context_grid.shape[0],
        device=query_points.device,
        dtype=context_grid.dtype,
    )


# Setup tensors and metadata required for custom-op backward.
def setup_interpolation_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    query_points, context_grid, grid_meta, interp_id, mem_speed_trade = inputs
    ctx.save_for_backward(query_points, context_grid, grid_meta)
    ctx.interp_id = int(interp_id)
    ctx.mem_speed_trade = bool(mem_speed_trade)


# Compute backward for all interpolation modes via manual formulas.
def backward_interpolation(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
    query_points, context_grid, grid_meta = ctx.saved_tensors
    if grad_output is None:
        return None, None, None, None, None
    grad_query, grad_grid = launch_backward(
        query_points=query_points,
        context_grid=context_grid,
        grid_meta=grid_meta,
        interp_id=ctx.interp_id,
        grad_output=grad_output,
        needs_input_grad=ctx.needs_input_grad,
    )
    return grad_query, grad_grid, None, None, None


# Register custom-op backward.
interpolation_impl.register_autograd(
    backward_interpolation, setup_context=setup_interpolation_context
)


# Public warp entry point used by the interpolation FunctionSpec.
def interpolation_warp(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str = "smooth_step_2",
    mem_speed_trade: bool = True,
) -> torch.Tensor:
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
    return interpolation_impl(
        query_points,
        context_grid,
        grid_meta,
        int(interp_id),
        mem_speed_trade,
    )


__all__ = ["interpolation_warp"]
