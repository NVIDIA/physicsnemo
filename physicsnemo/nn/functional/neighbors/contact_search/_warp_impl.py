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

"""PyTorch integration for the Warp contact-search kernels."""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.neighbors.radius_search._warp_impl import (
    apply_grad_to_points,
)
from physicsnemo.nn.functional.neighbors.radius_search.utils import format_returns

from .kernels import (
    contact_search_count,
    contact_search_limited_select,
    contact_search_limited_select_batched,
    contact_search_unlimited_select,
)
from .utils import validate_contact_inputs

wp.config.log_level = wp.LOG_WARNING
wp.init()

_SMALL_QUERY_BLOCK_DIM = 128
_LARGE_QUERY_BLOCK_DIM = 256
_LARGE_QUERY_THRESHOLD = 65_536


def _query_block_dim(num_queries: int) -> int:
    """Select the measured occupancy sweet spot for the query launch."""
    if num_queries < _LARGE_QUERY_THRESHOLD:
        return _SMALL_QUERY_BLOCK_DIM
    return _LARGE_QUERY_BLOCK_DIM


def _count_neighbors(
    grid: wp.HashGrid,
    wp_points: wp.array(dtype=wp.vec3),
    wp_queries: wp.array(dtype=wp.vec3),
    wp_exclude_neighbors: wp.array(dtype=wp.int32),
    wp_exclude_offsets: wp.array(dtype=wp.int32),
    exclusion_row_base: int,
    launch_device: wp.Device | None,
    launch_stream: wp.Stream | None,
    radius: float,
    num_queries: int,
    sync: bool,
) -> tuple[int | torch.Tensor, torch.Tensor]:
    """Count valid neighbors and return their exclusive-scan offsets."""
    wp_result_count = wp.zeros(num_queries, device=wp_points.device, dtype=wp.int32)
    wp.launch(
        kernel=contact_search_count,
        dim=num_queries,
        inputs=[
            grid.id,
            wp_points,
            wp_queries,
            wp_exclude_neighbors,
            wp_exclude_offsets,
            exclusion_row_base,
            radius,
        ],
        outputs=[wp_result_count],
        stream=launch_stream,
        device=launch_device,
        block_dim=_query_block_dim(num_queries),
    )

    result_count = wp.to_torch(wp_result_count)
    offsets = torch.empty(
        num_queries + 1, device=result_count.device, dtype=torch.int64
    )
    offsets[0] = 0
    torch.cumsum(result_count, dim=0, dtype=torch.int64, out=offsets[1:])

    if sync:
        pinned = torch.zeros(
            1,
            dtype=torch.int64,
            pin_memory=torch.cuda.is_available(),
        )
        pinned.copy_(offsets[-1:])
        return pinned.item(), offsets

    return offsets[-1:], offsets


def _gather_neighbors(
    grid: wp.HashGrid,
    output_device: torch.device,
    wp_points: wp.array(dtype=wp.vec3),
    wp_queries: wp.array(dtype=wp.vec3),
    wp_exclude_neighbors: wp.array(dtype=wp.int32),
    wp_exclude_offsets: wp.array(dtype=wp.int32),
    exclusion_row_base: int,
    wp_offsets: wp.array(dtype=wp.int32),
    launch_device: wp.Device | None,
    launch_stream: wp.Stream | None,
    radius: float,
    num_queries: int,
    return_dists: bool,
    return_points: bool,
    total_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather valid neighbors into dynamic COO output arrays."""
    indices = torch.zeros((2, total_count), dtype=torch.int32, device=output_device)
    if return_dists:
        distances = torch.zeros(
            (total_count,), dtype=torch.float32, device=output_device
        )
    else:
        distances = torch.empty((0,), dtype=torch.float32, device=output_device)
    if return_points:
        points_out = torch.zeros(
            (total_count, 3), dtype=torch.float32, device=output_device
        )
    else:
        points_out = torch.empty((0, 3), dtype=torch.float32, device=output_device)

    wp.launch(
        kernel=contact_search_unlimited_select,
        dim=num_queries,
        inputs=[
            grid.id,
            wp_points,
            wp_queries,
            wp_exclude_neighbors,
            wp_exclude_offsets,
            exclusion_row_base,
            wp_offsets,
            wp.from_torch(indices, return_ctype=True),
            radius,
            return_dists,
            wp.from_torch(distances, return_ctype=True),
            return_points,
            wp.from_torch(points_out, return_ctype=True),
        ],
        stream=launch_stream,
        device=launch_device,
        block_dim=_query_block_dim(num_queries),
    )
    return indices, points_out, distances


@torch.library.custom_op("physicsnemo::contact_search_warp", mutates_args=())
def contact_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute topology-filtered radius search using Warp hash grids."""
    points, queries, was_unbatched, exclusion_row_stride = validate_contact_inputs(
        points,
        queries,
        radius,
        exclude_neighbors,
        exclude_offsets,
        max_points,
    )
    batch_size, num_points = points.shape[:2]
    num_queries = queries.shape[1]
    input_dtype = points.dtype

    if input_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "Warp contact_search supports float16, bfloat16, and float32 "
            f"coordinates, got {input_dtype}. Use implementation='torch' "
            "for float64 coordinates to preserve search and output precision."
        )

    # Warp hash-grid kernels operate on fp32 coordinates. CSR tensors remain
    # int32 and are shared directly with Warp without copies when contiguous.
    if points.dtype != torch.float32:
        points = points.to(torch.float32)
    if queries.dtype != torch.float32:
        queries = queries.to(torch.float32)
    points = points.contiguous()
    queries = queries.contiguous()
    exclude_neighbors = exclude_neighbors.contiguous()
    exclude_offsets = exclude_offsets.contiguous()

    launch_device, launch_stream = FunctionSpec.warp_launch_context(points)
    with FunctionSpec.warp_stream_scope(launch_stream):
        wp_exclude_neighbors = wp.from_torch(
            exclude_neighbors, dtype=wp.int32, return_ctype=True
        )
        wp_exclude_offsets = wp.from_torch(
            exclude_offsets, dtype=wp.int32, return_ctype=True
        )

        grids: list[wp.HashGrid] = []
        wp_points_per_batch = []
        wp_queries_per_batch = []
        for batch_index in range(batch_size):
            points_batch = points[batch_index]
            queries_batch = queries[batch_index]
            # The registered Torch backward scatters selected-point gradients;
            # discovery must not allocate Warp-owned .grad buffers as well.
            wp_points = wp.from_torch(points_batch, dtype=wp.vec3, requires_grad=False)
            wp_queries = wp.from_torch(
                queries_batch, dtype=wp.vec3, requires_grad=False, return_ctype=True
            )
            grid = wp.HashGrid(
                dim_x=128,
                dim_y=128,
                dim_z=128,
                device=wp_points.device,
            )
            grid.reserve(num_points)
            grid.build(points=wp_points, radius=0.5 * radius)
            grids.append(grid)
            wp_points_per_batch.append(wp_points)
            wp_queries_per_batch.append(wp_queries)

        if max_points is None:
            count_tensors = []
            offset_tensors = []
            for batch_index in range(batch_size):
                exclusion_row_base = batch_index * exclusion_row_stride
                count, offsets = _count_neighbors(
                    grids[batch_index],
                    wp_points_per_batch[batch_index],
                    wp_queries_per_batch[batch_index],
                    wp_exclude_neighbors,
                    wp_exclude_offsets,
                    exclusion_row_base,
                    launch_device,
                    launch_stream,
                    radius,
                    num_queries,
                    sync=batch_size == 1,
                )
                count_tensors.append(count)
                offset_tensors.append(offsets)

            if batch_size == 1:
                total_counts = [count_tensors[0]]
            else:
                gpu_counts = torch.cat(count_tensors, dim=0)
                cpu_counts = torch.zeros(
                    batch_size,
                    dtype=torch.int64,
                    pin_memory=torch.cuda.is_available(),
                )
                cpu_counts.copy_(gpu_counts)
                total_counts = cpu_counts.tolist()

            for total_count in total_counts:
                if total_count >= torch.iinfo(torch.int32).max:
                    raise RuntimeError(
                        "Total found contact candidates is too large: "
                        f"{total_count} >= 2**31 - 1"
                    )

            offset_tensors_int32 = [
                offsets.to(torch.int32) for offsets in offset_tensors
            ]
            wp_offsets = [
                wp.from_torch(offsets, dtype=wp.int32)
                for offsets in offset_tensors_int32
            ]
            all_indices = []
            all_points = []
            all_distances = []
            for batch_index in range(batch_size):
                exclusion_row_base = batch_index * exclusion_row_stride
                batch_indices, batch_points, batch_distances = _gather_neighbors(
                    grids[batch_index],
                    points.device,
                    wp_points_per_batch[batch_index],
                    wp_queries_per_batch[batch_index],
                    wp_exclude_neighbors,
                    wp_exclude_offsets,
                    exclusion_row_base,
                    wp_offsets[batch_index],
                    launch_device,
                    launch_stream,
                    radius,
                    num_queries,
                    return_dists,
                    return_points,
                    total_counts[batch_index],
                )
                batch_row = torch.full(
                    (1, batch_indices.shape[1]),
                    batch_index,
                    dtype=batch_indices.dtype,
                    device=batch_indices.device,
                )
                all_indices.append(torch.cat([batch_row, batch_indices], dim=0))
                all_points.append(batch_points)
                all_distances.append(batch_distances)

            indices = torch.cat(all_indices, dim=1)
            points_out = (
                torch.cat(all_points, dim=0) if return_points else all_points[0]
            )
            distances_out = (
                torch.cat(all_distances, dim=0) if return_dists else all_distances[0]
            )
            num_neighbors = torch.empty((0,), dtype=torch.int32, device=points.device)
            if was_unbatched:
                indices = indices[1:]
        else:
            if was_unbatched:
                index_shape = (num_queries, max_points)
                count_shape = (num_queries,)
            else:
                index_shape = (batch_size, num_queries, max_points)
                count_shape = (batch_size, num_queries)

            indices = torch.full(
                index_shape, -1, dtype=torch.int32, device=points.device
            )
            num_neighbors = torch.zeros(
                count_shape, dtype=torch.int32, device=points.device
            )
            if return_dists:
                distances_out = torch.zeros(
                    index_shape,
                    dtype=torch.float32,
                    device=points.device,
                )
            else:
                distances_out = torch.empty(
                    (0,), dtype=torch.float32, device=points.device
                )
            # The bounded heap needs squared distances even for index-only
            # queries. Reuse the public distance buffer when requested; kernels
            # convert only valid retained entries to Euclidean distances last.
            selection_distances = (
                distances_out
                if return_dists
                else torch.empty(index_shape, dtype=torch.float32, device=points.device)
            )
            if return_points:
                points_out = torch.zeros(
                    (*index_shape, 3),
                    dtype=torch.float32,
                    device=points.device,
                )
            else:
                points_out = torch.empty(
                    (0, max_points, 3),
                    dtype=torch.float32,
                    device=points.device,
                )

            if was_unbatched:
                wp.launch(
                    kernel=contact_search_limited_select,
                    dim=num_queries,
                    inputs=[
                        grids[0].id,
                        wp_points_per_batch[0],
                        wp_queries_per_batch[0],
                        wp_exclude_neighbors,
                        wp_exclude_offsets,
                        max_points,
                        radius,
                        wp.from_torch(indices.view(-1), return_ctype=True),
                        wp.from_torch(num_neighbors, return_ctype=True),
                        return_dists,
                        wp.from_torch(selection_distances.view(-1), return_ctype=True),
                        return_points,
                        wp.from_torch(points_out, return_ctype=True)
                        if return_points
                        else None,
                    ],
                    stream=launch_stream,
                    device=launch_device,
                    block_dim=_query_block_dim(num_queries),
                )
            else:
                grid_ids_host = torch.tensor(
                    [grid.id for grid in grids],
                    dtype=torch.int64,
                    pin_memory=torch.cuda.is_available(),
                )
                grid_ids = grid_ids_host.to(points.device, non_blocking=True)
                wp_grid_ids = wp.from_torch(
                    grid_ids, dtype=wp.uint64, return_ctype=True
                )
                wp_points = wp.from_torch(
                    points, dtype=wp.vec3, requires_grad=False, return_ctype=True
                )
                wp_queries = wp.from_torch(
                    queries, dtype=wp.vec3, requires_grad=False, return_ctype=True
                )
                wp.launch(
                    kernel=contact_search_limited_select_batched,
                    dim=(batch_size, num_queries),
                    inputs=[
                        wp_grid_ids,
                        wp_points,
                        wp_queries,
                        wp_exclude_neighbors,
                        wp_exclude_offsets,
                        exclusion_row_stride,
                        max_points,
                        radius,
                        wp.from_torch(indices.view(-1), return_ctype=True),
                        wp.from_torch(num_neighbors, return_ctype=True),
                        return_dists,
                        wp.from_torch(selection_distances.view(-1), return_ctype=True),
                        return_points,
                        wp.from_torch(points_out, return_ctype=True)
                        if return_points
                        else None,
                    ],
                    stream=launch_stream,
                    device=launch_device,
                    block_dim=_query_block_dim(num_queries),
                )

    return (
        indices,
        points_out.to(input_dtype),
        distances_out.to(input_dtype),
        num_neighbors,
    )


@contact_search_impl.register_fake
def _contact_search_impl_fake(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for ``torch.compile`` shape propagation."""
    if max_points is None:
        # Dynamic output shapes cannot be represented by this fake kernel.
        torch._dynamo.graph_break()
        return None

    if points.ndim == 3:
        index_shape = (points.shape[0], queries.shape[1], max_points)
        count_shape = (points.shape[0], queries.shape[1])
    else:
        index_shape = (queries.shape[0], max_points)
        count_shape = (queries.shape[0],)

    indices = torch.empty(index_shape, dtype=torch.int32, device=queries.device)
    num_neighbors = torch.empty(count_shape, dtype=torch.int32, device=queries.device)
    if return_dists:
        distances = torch.empty(index_shape, dtype=points.dtype, device=queries.device)
    else:
        distances = torch.empty((0,), dtype=points.dtype, device=queries.device)
    if return_points:
        points_out = torch.empty(
            *index_shape, 3, dtype=points.dtype, device=queries.device
        )
    else:
        points_out = torch.empty(
            (0, max_points, 3), dtype=points.dtype, device=queries.device
        )
    return indices, points_out, distances, num_neighbors


def _setup_contact_search_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: tuple,
) -> None:
    (
        points,
        _queries,
        _radius,
        _exclude_neighbors,
        _exclude_offsets,
        max_points,
        _return_dists,
        return_points,
    ) = inputs
    indices, _points_out, _distances, num_neighbors = output
    ctx.return_points = return_points
    ctx.max_points = max_points
    if return_points:
        ctx.points_shape = points.shape
        ctx.save_for_backward(indices, num_neighbors)


def _backward_contact_search(
    ctx: torch.autograd.function.FunctionCtx,
    _grad_indices: torch.Tensor,
    grad_points_out: torch.Tensor | None,
    _grad_distances: torch.Tensor | None,
    _grad_num_neighbors: torch.Tensor | None,
) -> tuple:
    if ctx.return_points and grad_points_out is not None:
        indices, num_neighbors = ctx.saved_tensors
        output_dtype = grad_points_out.dtype
        # The shared Warp scatter kernel operates on vec3<float32>. Preserve
        # reduced-precision API behavior by accumulating in fp32 and casting
        # the completed input gradient back to the caller's dtype.
        if output_dtype != torch.float32:
            grad_points_out = grad_points_out.to(torch.float32)
        point_grads = apply_grad_to_points(
            indices,
            num_neighbors,
            grad_points_out,
            ctx.points_shape,
            ctx.max_points,
        )
        if output_dtype != torch.float32:
            point_grads = point_grads.to(output_dtype)
    else:
        point_grads = None
    return point_grads, None, None, None, None, None, None, None


contact_search_impl.register_autograd(
    _backward_contact_search,
    setup_context=_setup_contact_search_context,
)


def contact_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """Warp-backend entry point with public return formatting."""
    indices, points_out, distances, _ = contact_search_impl(
        points,
        queries,
        radius,
        exclude_neighbors,
        exclude_offsets,
        max_points,
        return_dists,
        return_points,
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)


__all__ = ["contact_search", "contact_search_impl"]
