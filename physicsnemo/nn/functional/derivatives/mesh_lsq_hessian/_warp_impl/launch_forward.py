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

"""Forward Warp launch surface for mesh LSQ Hessians."""

from __future__ import annotations

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ._kernels import factorize_kernel, forward_kernel


def launch_factorization(
    *,
    points_fp32: torch.Tensor,
    offsets_i32: torch.Tensor,
    indices_i32: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    distance_epsilon: float,
    requested_rcond: float,
    wp_device,
    wp_stream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate and build one geometry-only CPQR factorization per entity."""
    n_entities = points_fp32.shape[0]
    q_coefficients = torch.zeros(
        (n_entities, 9, 9),
        dtype=torch.float32,
        device=points_fp32.device,
    )
    r_factor = torch.zeros_like(q_coefficients)
    permutation = torch.zeros(
        (n_entities, 9),
        dtype=torch.int32,
        device=points_fp32.device,
    )
    fit_info = torch.zeros(
        (n_entities, 3),
        dtype=torch.float32,
        device=points_fp32.device,
    )
    full_rank = torch.zeros(
        (n_entities,),
        dtype=torch.int32,
        device=points_fp32.device,
    )
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            kernel=factorize_kernel,
            dim=points_fp32.shape[0],
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(offsets_i32, dtype=wp.int32),
                wp.from_torch(indices_i32, dtype=wp.int32),
                int(points_fp32.shape[1]),
                float(weight_power),
                int(min_neighbors),
                float(distance_epsilon),
                float(requested_rcond),
                wp.from_torch(q_coefficients, dtype=wp.float32),
                wp.from_torch(r_factor, dtype=wp.float32),
                wp.from_torch(permutation, dtype=wp.int32),
                wp.from_torch(fit_info, dtype=wp.float32),
                wp.from_torch(full_rank, dtype=wp.int32),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return q_coefficients, r_factor, permutation, fit_info, full_rank


def launch_forward(
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
    hessians_flat: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Apply stored QR factors to every entity and value component."""
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            kernel=forward_kernel,
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
                wp.from_torch(hessians_flat, dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )


__all__ = ["launch_factorization", "launch_forward"]
