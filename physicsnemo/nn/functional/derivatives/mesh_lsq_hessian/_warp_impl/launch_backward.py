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

"""Backward Warp launch surface for mesh LSQ Hessians."""

from __future__ import annotations

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ._kernels import backward_points_kernel, backward_values_kernel


def launch_backward_values(
    *,
    points_fp32: torch.Tensor,
    offsets_i32: torch.Tensor,
    indices_i32: torch.Tensor,
    weight_power: float,
    distance_epsilon: float,
    q_coefficients: torch.Tensor,
    r_factor: torch.Tensor,
    permutation: torch.Tensor,
    fit_info: torch.Tensor,
    full_rank: torch.Tensor,
    grad_output_flat: torch.Tensor,
    grad_values_flat: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Launch the explicit value adjoint."""
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            kernel=backward_values_kernel,
            dim=(points_fp32.shape[0], grad_values_flat.shape[1]),
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(offsets_i32, dtype=wp.int32),
                wp.from_torch(indices_i32, dtype=wp.int32),
                int(points_fp32.shape[1]),
                float(weight_power),
                float(distance_epsilon),
                wp.from_torch(q_coefficients, dtype=wp.float32),
                wp.from_torch(r_factor, dtype=wp.float32),
                wp.from_torch(permutation, dtype=wp.int32),
                wp.from_torch(fit_info, dtype=wp.float32),
                wp.from_torch(full_rank, dtype=wp.int32),
                wp.from_torch(grad_output_flat, dtype=wp.float32),
                wp.from_torch(grad_values_flat, dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )


def launch_backward_points(
    *,
    points_fp32: torch.Tensor,
    values_flat_fp32: torch.Tensor,
    offsets_i32: torch.Tensor,
    indices_i32: torch.Tensor,
    weight_power: float,
    distance_epsilon: float,
    q_coefficients: torch.Tensor,
    r_factor: torch.Tensor,
    permutation: torch.Tensor,
    fit_info: torch.Tensor,
    full_rank: torch.Tensor,
    grad_output_flat: torch.Tensor,
    grad_points_fp32: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Launch the explicit coordinate adjoint."""
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            kernel=backward_points_kernel,
            dim=(points_fp32.shape[0], values_flat_fp32.shape[1]),
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(values_flat_fp32, dtype=wp.float32),
                wp.from_torch(offsets_i32, dtype=wp.int32),
                wp.from_torch(indices_i32, dtype=wp.int32),
                int(points_fp32.shape[1]),
                float(weight_power),
                float(distance_epsilon),
                wp.from_torch(q_coefficients, dtype=wp.float32),
                wp.from_torch(r_factor, dtype=wp.float32),
                wp.from_torch(permutation, dtype=wp.int32),
                wp.from_torch(fit_info, dtype=wp.float32),
                wp.from_torch(full_rank, dtype=wp.int32),
                wp.from_torch(grad_output_flat, dtype=wp.float32),
                wp.from_torch(grad_points_fp32, dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )


__all__ = ["launch_backward_points", "launch_backward_values"]
