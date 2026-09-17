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

"""Warp-accelerated farthest-point sampling."""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .kernels import fps_fused
from .utils import validate_inputs

wp.config.log_level = wp.LOG_WARNING
wp.init()

# Threads per block (one block per cloud). Lanes scan the points strided, so
# this works for any ``num_points``; the launch caps it at ``min(_BLOCK_SIZE,
# num_points)``. Larger blocks give more parallelism per cloud (fewer points
# per lane), which matters most for large clouds.
_BLOCK_SIZE = 512


@torch.library.custom_op("physicsnemo::farthest_point_sampling_warp", mutates_args=())
def farthest_point_sampling(
    points: torch.Tensor,
    num_samples: int,
    random_start: bool = False,
) -> torch.Tensor:
    """Warp-accelerated greedy farthest-point sampling.

    Runs the entire selection in a single fused kernel launch — one thread
    block per cloud, with the ``num_samples`` steps looped inside the kernel —
    so there is no per-iteration kernel launch or host round-trip. See
    :class:`FarthestPointSampling` for the public contract.
    """
    points_b, was_unbatched = validate_inputs(points, num_samples)
    batch_size, num_points, point_dim = (
        int(points_b.shape[0]),
        int(points_b.shape[1]),
        int(points_b.shape[2]),
    )
    device = points_b.device
    if device.type != "cuda":
        # The fused kernel relies on Warp tile block-reductions, which only
        # cooperate across lanes on the CUDA backend. Use the torch baseline
        # on CPU (auto-dispatch routes there automatically).
        raise ValueError(
            "The Warp farthest_point_sampling backend requires CUDA tensors; "
            "use implementation='torch' on CPU."
        )

    points_f = points_b.detach().to(torch.float32).contiguous()
    block_size = min(_BLOCK_SIZE, max(1, num_points))
    if random_start:
        start = torch.randint(
            0, num_points, (batch_size,), device=device, dtype=torch.int32
        )
    else:
        start = torch.zeros(batch_size, device=device, dtype=torch.int32)
    selected = torch.empty(batch_size, num_samples, device=device, dtype=torch.int32)
    min_dist = torch.full(
        (batch_size, num_points), 1.0e30, device=device, dtype=torch.float32
    )

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(points_f)
    with FunctionSpec.warp_stream_scope(wp_launch_stream):
        wp.launch(
            fps_fused,
            dim=(batch_size, block_size),
            block_dim=block_size,
            inputs=[
                wp.from_torch(points_f, return_ctype=True),
                wp.from_torch(start, return_ctype=True),
                wp.from_torch(selected, return_ctype=True),
                wp.from_torch(min_dist, return_ctype=True),
                num_points,
                num_samples,
                point_dim,
                block_size,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

    selected = selected.to(torch.int64)
    if was_unbatched:
        selected = selected.squeeze(0)
    return selected


@farthest_point_sampling.register_fake
def _(
    points: torch.Tensor,
    num_samples: int,
    random_start: bool = False,
) -> torch.Tensor:
    if points.ndim == 2:
        return torch.empty((num_samples,), dtype=torch.int64, device=points.device)
    return torch.empty(
        (int(points.shape[0]), num_samples),
        dtype=torch.int64,
        device=points.device,
    )
