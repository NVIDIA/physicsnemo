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

"""
This file contains the interface between PyTorch and Warp kernels.

It uses a mix of utilities, such that it needs to be opaque to pure PyTorch.
At the same time, we want to rely on PyTorch's memory allocation as much as possible
and not warp.  So, tensor creation and allocation is driven by torch, and
passed to warp for computation.
"""

from typing import List

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .kernels import (
    radius_search_count,
    radius_search_limited_select_batched,
    radius_search_unlimited_select,
    scatter_add_batched,
    scatter_add_unlimited,
)
from .utils import format_returns, validate_inputs

wp.config.quiet = True

wp.init()

BLOCK_DIM = 32


def count_neighbors(
    grid: wp.HashGrid,
    wp_points: wp.array(dtype=wp.vec3),
    wp_queries: wp.array(dtype=wp.vec3),
    wp_launch_device: wp.Device | None,
    wp_launch_stream: wp.Stream | None,
    radius: float,
    N_queries: int,
    sync: bool = True,
) -> tuple:
    """
    Count the number of neighbors within a given radius for each query point.

    Args:
        grid (wp.HashGrid): The hash grid to use for the search.
        wp_points (wp.array): The points to search in, as a warp array.
        wp_queries (wp.array): The queries to search for, as a warp array.
        wp_launch_device (wp.Device | None): The device to launch the kernel on.
        wp_launch_stream (wp.Stream | None): The stream to launch the kernel on.
        radius (float): The radius that bounds the search.
        N_queries (int): Total number of query points.
        sync (bool): If True, copies count to CPU and returns (int, wp_offset).
            If False, returns (gpu_count_tensor, wp_offset) for batched sync.

    Returns:
        When sync=True: tuple[int, wp.array] -- total count and offset array.
        When sync=False: tuple[torch.Tensor, wp.array] -- GPU-side count tensor and offset array.
    """
    wp_result_count = wp.zeros(N_queries, device=wp_points.device, dtype=wp.int32)

    wp.launch(
        kernel=radius_search_count,
        dim=N_queries,
        inputs=[grid.id, wp_points, wp_queries, radius],
        outputs=[wp_result_count],
        stream=wp_launch_stream,
        device=wp_launch_device,
        block_dim=BLOCK_DIM,
    )

    wp_offset = wp.zeros(N_queries + 1, device=wp_points.device, dtype=wp.int32)
    torch_offset = wp.to_torch(wp_offset)
    torch_result_count = wp.to_torch(wp_result_count)
    torch.cumsum(torch_result_count, dim=0, out=torch_offset[1:])

    if sync:
        pin_memory = torch.cuda.is_available()
        pinned_buffer = torch.zeros(1, dtype=torch.int32, pin_memory=pin_memory)
        pinned_buffer.copy_(torch_offset[-1:])
        return pinned_buffer.item(), wp_offset

    # Return the last element as a 1-element GPU tensor for batch-sync later
    return torch_offset[-1:], wp_offset


def gather_neighbors(
    grid: wp.HashGrid,
    output_device: torch.device,
    wp_points: wp.array(dtype=wp.vec3),
    wp_queries: wp.array(dtype=wp.vec3),
    wp_offset: wp.array(dtype=wp.int32),
    wp_launch_device: wp.Device | None,
    wp_launch_stream: wp.Stream | None,
    radius: float,
    N_queries: int,
    return_dists: bool,
    return_points: bool,
    total_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gather the neighbors for each query point.

    Args:
        grid (wp.HashGrid): The hash grid to use for the search.
        output_device (torch.device): The device to allocate output tensors on.
        wp_points (wp.array): The points to search in, as a warp array.
        wp_queries (wp.array): The queries to search for, as a warp array.
        wp_offset (wp.array): The offset in output for each input point, as a warp array.
        wp_launch_device (wp.Device | None): The device to launch the kernel on.
        wp_launch_stream (wp.Stream | None): The stream to launch the kernel on.
        radius (float): The radius that bounds the search.
        N_queries (int): Total number of query points.
        return_dists (bool): Whether to return the distances of the neighbors.
        return_points (bool): Whether to return the points of the neighbors.
        total_count (int): The total number of neighbors found.

    Returns:
        tuple[torch.Tensor, ...]: Indices, points, distances, and num_neighbors tensors.
    """
    indices = torch.zeros((2, total_count), dtype=torch.int32, device=output_device)

    if return_dists:
        distances = torch.zeros(
            (total_count,), dtype=torch.float32, device=output_device
        )
    else:
        distances = torch.empty(0, dtype=torch.float32, device=output_device)

    if return_points:
        points = torch.zeros(
            (total_count, 3), dtype=torch.float32, device=output_device
        )
    else:
        points = torch.empty(0, 3, dtype=torch.float32, device=output_device)

    wp.launch(
        kernel=radius_search_unlimited_select,
        dim=N_queries,
        inputs=[
            grid.id,
            wp_points,
            wp_queries,
            wp_offset,
            wp.from_torch(indices, return_ctype=True),
            radius,
            return_dists,
            wp.from_torch(distances, return_ctype=True),
            return_points,
            wp.from_torch(points, return_ctype=True),
        ],
        stream=wp_launch_stream,
        device=wp_launch_device,
        block_dim=BLOCK_DIM,
    )

    num_neighbors = torch.empty(0, dtype=torch.int32, device=output_device)
    return indices, points, distances, num_neighbors


@torch.library.custom_op("physicsnemo::radius_search_warp", mutates_args=())
def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find and return the nearest neighbors in `points` using locations from `queries`.

    Implemented with warp kernels.  Make sure points and queries are on the same device.
    Accepts both unbatched (N, 3) and batched (B, N, 3) inputs.

    Always returns indices, points, distances.  If return_points is False, points is an empty tensor.
    If return_dists is False, distances is an empty tensor.

    Args:
        points (torch.Tensor): The points to search in, (N, 3) or (B, N, 3).
        queries (torch.Tensor): The queries to search for, (M, 3) or (B, M, 3).
        radius (float): The radius that bounds the search.
        max_points (int | None, optional): The maximum number of points to return per query. If None, unlimited.
        return_dists (bool, optional): Whether to return the distances of the neighbors.
        return_points (bool, optional): Whether to return the points of the neighbors.

    Returns:
        tuple[torch.Tensor, ...]: (indices, points, distances, num_neighbors)
    """

    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")

    points, queries, was_unbatched = validate_inputs(points, queries)
    B = points.shape[0]
    N_queries = queries.shape[1]

    input_dtype = points.dtype

    # Warp supports only fp32, so we have to cast:
    if points.dtype != torch.float32:
        points = points.to(torch.float32)
    if queries.dtype != torch.float32:
        queries = queries.to(torch.float32)

    # Compute follows data.
    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(points)

    with wp.ScopedStream(wp_launch_stream):
        # Build one hash grid per batch element (Python loop)
        grids = []
        wp_points_per_b = []
        wp_queries_per_b = []
        for b in range(B):
            pts_b = points[b].contiguous()
            qrs_b = queries[b].contiguous()
            wp_pts_b = wp.from_torch(pts_b, dtype=wp.vec3)
            wp_qrs_b = wp.from_torch(qrs_b, dtype=wp.vec3, return_ctype=True)
            grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=wp_pts_b.device)
            grid.reserve(N_queries)
            grid.build(points=wp_pts_b, radius=0.5 * radius)
            grids.append(grid)
            wp_points_per_b.append(wp_pts_b)
            wp_queries_per_b.append(wp_qrs_b)

        if max_points is None:
            # ---------------------------------------------------------------
            # Dynamic output path: per-element count, single sync, then gather
            # ---------------------------------------------------------------

            # Count pass: collect all counts without syncing individually
            count_tensors = []
            wp_offsets = []
            for b in range(B):
                count_t, wp_off = count_neighbors(
                    grids[b],
                    wp_points_per_b[b],
                    wp_queries_per_b[b],
                    wp_launch_device,
                    wp_launch_stream,
                    radius,
                    N_queries,
                    sync=(B == 1),
                )
                count_tensors.append(count_t)
                wp_offsets.append(wp_off)

            # Sync: for B==1 count_tensors[0] is already an int;
            # for B>1 we batch-sync all GPU tensors at once
            if B == 1:
                total_counts = [count_tensors[0]]
            else:
                gpu_counts = torch.cat(count_tensors, dim=0)
                pin_memory = torch.cuda.is_available()
                cpu_counts = torch.zeros(B, dtype=torch.int32, pin_memory=pin_memory)
                cpu_counts.copy_(gpu_counts)
                total_counts = cpu_counts.tolist()

            for tc in total_counts:
                if not tc < 2**31 - 1:
                    raise RuntimeError(
                        f"Total found neighbors is too large: {tc} >= 2**31 - 1"
                    )

            # Gather per batch element, concatenate with batch indices
            all_indices = []
            all_pts = []
            all_dists = []
            for b in range(B):
                idx_b, pts_b, dist_b, _ = gather_neighbors(
                    grids[b],
                    points.device,
                    wp_points_per_b[b],
                    wp_queries_per_b[b],
                    wp_offsets[b],
                    wp_launch_device,
                    wp_launch_stream,
                    radius,
                    N_queries,
                    return_dists,
                    return_points,
                    total_counts[b],
                )
                # idx_b: (2, count_b); prepend batch-index row
                batch_row = torch.full(
                    (1, idx_b.shape[1]),
                    b,
                    dtype=idx_b.dtype,
                    device=idx_b.device,
                )
                all_indices.append(torch.cat([batch_row, idx_b], dim=0))
                all_pts.append(pts_b)
                all_dists.append(dist_b)

            indices = torch.cat(all_indices, dim=1)  # (3, total)
            pts_out = torch.cat(all_pts, dim=0) if return_points else all_pts[0]
            dists_out = torch.cat(all_dists, dim=0) if return_dists else all_dists[0]
            num_neighbors = torch.empty(0, dtype=torch.int32, device=points.device)

            if was_unbatched:
                # Strip the batch-index row to restore (2, count) format
                indices = indices[1:]

        else:
            # ---------------------------------------------------------------
            # Deterministic output path: always use batched 2D kernel launch
            # ---------------------------------------------------------------

            # Build warp array of grid IDs
            grid_ids_tensor = torch.tensor(
                [g.id for g in grids], dtype=torch.int64, device=points.device
            )
            wp_grid_ids = wp.from_torch(
                grid_ids_tensor, dtype=wp.uint64, return_ctype=True
            )

            # Convert batched points/queries to warp 2D arrays
            wp_points_2d = wp.from_torch(
                points.contiguous(), dtype=wp.vec3, return_ctype=True
            )
            wp_queries_2d = wp.from_torch(
                queries.contiguous(), dtype=wp.vec3, return_ctype=True
            )

            # Allocate outputs with batch dimension
            indices = torch.full(
                (B, N_queries, max_points),
                0,
                dtype=torch.int32,
                device=points.device,
            )
            num_neighbors = torch.zeros(
                (B, N_queries),
                dtype=torch.int32,
                device=points.device,
            )
            if return_dists:
                dists_out = torch.zeros(
                    (B, N_queries, max_points),
                    dtype=torch.float32,
                    device=points.device,
                )
            else:
                dists_out = torch.empty(
                    0,
                    dtype=torch.float32,
                    device=points.device,
                )
            if return_points:
                pts_out = torch.zeros(
                    (B, N_queries, max_points, 3),
                    dtype=torch.float32,
                    device=points.device,
                )
            else:
                pts_out = torch.empty(
                    (0, max_points, 3),
                    dtype=torch.float32,
                    device=points.device,
                )

            wp.launch(
                kernel=radius_search_limited_select_batched,
                dim=(B, N_queries),
                inputs=[
                    wp_grid_ids,
                    wp_points_2d,
                    wp_queries_2d,
                    max_points,
                    radius,
                    wp.from_torch(indices, return_ctype=True),
                    wp.from_torch(num_neighbors, return_ctype=True),
                    return_dists,
                    wp.from_torch(dists_out, return_ctype=True),
                    return_points,
                    wp.from_torch(pts_out, return_ctype=True)
                    if return_points
                    else None,
                ],
                stream=wp_launch_stream,
                device=wp_launch_device,
            )

            if was_unbatched:
                indices = indices.squeeze(0)
                num_neighbors = num_neighbors.squeeze(0)
                if return_dists:
                    dists_out = dists_out.squeeze(0)
                if return_points:
                    pts_out = pts_out.squeeze(0)

    # Handle the matrix of return values:
    pts_out = pts_out.to(input_dtype)
    dists_out = dists_out.to(input_dtype)
    return indices, pts_out, dists_out, num_neighbors


# This is to enable torch.compile:
@radius_search_impl.register_fake
def radius_search_impl_fake(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for torch.compile/fake tensor support.
    Handles both unbatched (N, 3) and batched (B, N, 3) inputs.
    """

    if max_points is not None:
        # Determine shape prefix based on input rank
        if points.ndim == 3:
            idx_shape = (points.shape[0], queries.shape[1], max_points)
            nn_shape = (points.shape[0], queries.shape[1])
        else:
            idx_shape = (queries.shape[0], max_points)
            nn_shape = (queries.shape[0],)

        indices = torch.empty(idx_shape, dtype=torch.int32, device=queries.device)
        num_neighbors = torch.empty(nn_shape, dtype=torch.int32, device=queries.device)

        # Dtype must match the real op, which returns tensors cast to points.dtype.
        # Hard-coding fp32 here causes Inductor to emit kernels with wrong
        # strides/byte-counts under bf16/fp16, triggering cudaErrorIllegalAddress.
        if return_dists:
            distances = torch.empty(
                idx_shape,
                dtype=points.dtype,
                device=queries.device,
            )
        else:
            distances = torch.empty(0, dtype=points.dtype, device=queries.device)

        if return_points:
            out_points = torch.empty(
                *idx_shape,
                3,
                dtype=points.dtype,
                device=queries.device,
            )
        else:
            # Real op returns rank with (0, max_points, 3); keep consistent
            out_points = torch.empty(
                0,
                max_points,
                3,
                dtype=points.dtype,
                device=queries.device,
            )

        return indices, out_points, distances, num_neighbors

    else:
        torch._dynamo.graph_break()


# This is for the autograd context creation.
def setup_radius_search_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: tuple
) -> None:
    """
    Set up the autograd context for the radius search operation.

    Args:
        ctx (torch.autograd.function.FunctionCtx): The autograd context.
        inputs (tuple): The input arguments to the forward function.
        output (tuple): The output tensors from the forward function.
    """
    points, queries, radius, max_points, return_dists, return_points = inputs

    indexes, ret_points, distances, num_neighbors = output

    # For the backward pass, we need to know how many neighbors
    # per index _if_ max points isn't none

    ctx.return_points = return_points
    ctx.max_points = max_points

    # save the indexes if we return points:
    if return_points:
        ctx.grad_points_shape = points.shape
        ctx.points_dtype = points.dtype
        ctx.save_for_backward(indexes, num_neighbors)


def backward_radius_search(
    ctx: torch.autograd.function.FunctionCtx,
    grad_idx: torch.Tensor,
    grad_points: torch.Tensor | None,
    grad_dists: torch.Tensor | None,
    grad_num_neighbors: torch.Tensor | None,
) -> tuple:
    """
    Backward function for the radius search operation.

    Args:
        ctx (torch.autograd.function.FunctionCtx): The autograd context.
        grad_idx (torch.Tensor): The gradient of the indices.
        grad_points (torch.Tensor | None): The gradient of the points - usually None
        grad_dists (torch.Tensor | None): The gradient of the distances - usually None
        grad_num_neighbors (torch.Tensor | None): The gradient of the number of neighbors - usually None

    Returns:
        tuple: Gradients of the inputs.
    """

    if ctx.return_points:
        (indexes, num_neighbors) = ctx.saved_tensors
        point_grads = apply_grad_to_points(
            indexes,
            num_neighbors,
            grad_points,
            ctx.grad_points_shape,
            ctx.max_points,
        )
    else:
        point_grads = None

    return point_grads, None, None, None, None, None


@torch.library.custom_op(
    "physicsnemo::radius_search_apply_grad_to_points", mutates_args=()
)
def apply_grad_to_points(
    indexes: torch.Tensor,
    num_neighbors: torch.Tensor,
    grad_points_out: torch.Tensor,
    points_shape: List[int],
    max_points: int | None = None,
) -> torch.Tensor:
    """
    Apply the gradient from the output points to the input points using the provided indices.
    Handles both unbatched and batched tensors.

    Args:
        indexes (torch.Tensor): The indices mapping output points to input points.
        num_neighbors (torch.Tensor): Per-query neighbor counts (max_points path).
        grad_points_out (torch.Tensor): The gradient of the output points.
        points_shape (List[int]): The shape of the input points tensor.
        max_points (int | None): Max neighbors per query, or None for unlimited.

    Returns:
        torch.Tensor: The gradient with respect to the input points.
    """
    point_grads = torch.zeros(
        points_shape, dtype=grad_points_out.dtype, device=grad_points_out.device
    )

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(
        grad_points_out
    )

    # Make sure the inputs are contiguous:
    if not grad_points_out.is_contiguous():
        grad_points_out = grad_points_out.contiguous()
    if not indexes.is_contiguous():
        indexes = indexes.contiguous()
    if not point_grads.is_contiguous():
        point_grads = point_grads.contiguous()
    if max_points is not None and not num_neighbors.is_contiguous():
        num_neighbors = num_neighbors.contiguous()

    if max_points is None:
        # Dynamic path: indexes is (2, total) unbatched or (3, total) batched.
        # scatter_add_unlimited works on flat (N, 3) grad tensors, so we loop
        # over batch elements (trivially 1 iteration for unbatched).
        if indexes.shape[-1] > 0:
            if indexes.shape[0] == 3:
                # Batched: row 0 is batch index, rows 1-2 are query/point indices
                B = point_grads.shape[0]
                for b_idx in range(B):
                    mask = indexes[0] == b_idx
                    b_indexes = indexes[1:, mask]
                    b_grad_out = grad_points_out[mask]
                    if b_indexes.shape[1] > 0:
                        wp.launch(
                            kernel=scatter_add_unlimited,
                            dim=b_indexes.shape[1],
                            inputs=[
                                wp.from_torch(
                                    b_indexes, dtype=wp.int32, return_ctype=True
                                ),
                                wp.from_torch(
                                    b_grad_out, dtype=wp.vec3, return_ctype=True
                                ),
                                wp.from_torch(
                                    point_grads[b_idx],
                                    dtype=wp.vec3,
                                    return_ctype=True,
                                ),
                            ],
                            device=wp_launch_device,
                            stream=wp_launch_stream,
                            block_dim=BLOCK_DIM,
                        )
            else:
                # Unbatched: rows 0-1 are query/point indices
                wp.launch(
                    kernel=scatter_add_unlimited,
                    dim=indexes.shape[1],
                    inputs=[
                        wp.from_torch(indexes, dtype=wp.int32, return_ctype=True),
                        wp.from_torch(
                            grad_points_out, dtype=wp.vec3, return_ctype=True
                        ),
                        wp.from_torch(point_grads, dtype=wp.vec3, return_ctype=True),
                    ],
                    device=wp_launch_device,
                    stream=wp_launch_stream,
                    block_dim=BLOCK_DIM,
                )

    else:
        # Deterministic path: always use batched kernel.
        # Unsqueeze 2D tensors to 3D so we can use a single kernel variant.
        if indexes.ndim == 2:
            indexes = indexes.unsqueeze(0)
            num_neighbors = num_neighbors.unsqueeze(0)
            grad_points_out = grad_points_out.unsqueeze(0)
            point_grads = point_grads.unsqueeze(0)

        B = indexes.shape[0]
        wp.launch(
            kernel=scatter_add_batched,
            dim=(B, indexes.shape[1]),
            inputs=[
                wp.from_torch(indexes, dtype=wp.int32, return_ctype=True),
                wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
                wp.from_torch(grad_points_out, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(point_grads, dtype=wp.vec3, return_ctype=True),
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
            block_dim=BLOCK_DIM,
        )

        if point_grads.shape[0] == 1 and len(points_shape) == 2:
            point_grads = point_grads.squeeze(0)

    return point_grads


@apply_grad_to_points.register_fake
def apply_grad_to_points_fake(
    indexes: torch.Tensor,
    num_neighbors: torch.Tensor,
    grad_points_out: torch.Tensor,
    points_shape: List[int],
    max_points: int | None = None,
) -> torch.Tensor:
    """
    Fake implementation for apply_grad_to_points for torch.compile/fake tensor support.

    Args:
        indexes (torch.Tensor): The indices mapping output points to input points.
        num_neighbors (torch.Tensor): The per-query neighbor counts (only used when
            ``max_points`` is not None, but always present to match the real op signature).
        grad_points_out (torch.Tensor): The gradient of the output points.
        points_shape (List[int]): The shape of the input points tensor.

    Returns:
        torch.Tensor: The gradient with respect to the input points.
    """
    point_grads = torch.empty(
        points_shape, dtype=grad_points_out.dtype, device=grad_points_out.device
    )

    return point_grads


radius_search_impl.register_autograd(
    backward_radius_search, setup_context=setup_radius_search_context
)


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """
    Perform a radius search between points and queries.

    Accepts both unbatched (N, 3) and batched (B, N, 3) inputs.

    Args:
        points (torch.Tensor): The input points tensor, (N, 3) or (B, N, 3).
        queries (torch.Tensor): The query points tensor, (M, 3) or (B, M, 3).
        radius (float): The search radius.
        max_points (int | None): The maximum number of neighbors per query, or
            None for unlimited.
        return_dists (bool): Whether to return distances between query and
            neighbor points.
        return_points (bool): Whether to return the neighbor points themselves.

    Returns:
        The formatted radius search results, whose contents depend on
        ``return_dists`` and ``return_points``.
    """
    indices, points_out, distances, _ = radius_search_impl(
        points, queries, radius, max_points, return_dists, return_points
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
