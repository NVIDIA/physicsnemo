# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.utils.version_check import check_min_version

WARP_AVAILABLE = check_min_version("warp", "0.6.0")

if WARP_AVAILABLE:
    import warp as wp

    wp.config.quiet = True

    wp.init()

    BLOCK_DIM = 32

    from .kernels import (
        radius_search_count,
        radius_search_limited_select,
        radius_search_unlimited_select,
        radius_search_unlimited_select_with_dists,
        radius_search_unlimited_select_with_dists_and_points,
        radius_search_unlimited_select_with_points,
        scatter_add,
        scatter_add_unlimited,
    )

    def count_neighbors(
        grid: wp.HashGrid,
        wp_points: wp.array(dtype=wp.vec3),
        wp_queries: wp.array(dtype=wp.vec3),
        wp_launch_device: wp.context.Device | None,
        wp_launch_stream: wp.Stream | None,
        radius: float,
        N_queries: int,
    ) -> tuple[int, wp.array]:
        """
        Count the number of neighbors within a given radius for each query point.

        Args:
            grid (wp.HashGrid): The hash grid to use for the search.
            wp_points (wp.array): The points to search in, as a warp array.
            wp_queries (wp.array): The queries to search for, as a warp array.
            wp_launch_device (wp.context.Device | None): The device to launch the kernel on.
            wp_launch_stream (wp.Stream | None): The stream to launch the kernel on.
            radius (float): The radius that bounds the search.
            N_queries (int): Total number of query points.

        Returns:
            tuple[int, wp.array]: The total count of neighbors and the offset array.
        """
        # For unlimited output points, we have to go through and count once:
        wp_result_count = wp.zeros(N_queries, device=wp_points.device, dtype=wp.int32)

        wp.launch(
            kernel=radius_search_count,
            dim=N_queries,
            inputs=[grid.id, wp_points, wp_queries, radius],
            outputs=[
                wp_result_count,
            ],
            stream=wp_launch_stream,
            device=wp_launch_device,
            block_dim=BLOCK_DIM,
        )

        # The offset tensor is owned by warp
        wp_offset = wp.zeros(N_queries + 1, device=wp_points.device, dtype=wp.int32)

        # Compute the offset from each point to the next point in terms of num neighbors:
        torch_offset = wp.to_torch(wp_offset)
        torch_result_count = wp.to_torch(wp_result_count)

        torch.cumsum(torch_result_count, dim=0, out=torch_offset[1:])

        # Create a pinned buffer on CPU to receive the count
        pinned_buffer = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        # Copy the last element to pinned memory
        pinned_buffer.copy_(torch_offset[-1:])
        total_count = pinned_buffer.item()

        # Return the count and the offsets:
        return total_count, wp_offset

    def gather_neighbors(
        grid: wp.HashGrid,
        output_device: torch.device,
        wp_points: wp.array(dtype=wp.vec3),
        wp_queries: wp.array(dtype=wp.vec3),
        wp_offset: wp.array(dtype=wp.int32),
        wp_launch_device: wp.context.Device | None,
        wp_launch_stream: wp.Stream | None,
        radius: float,
        N_queries: int,
        return_dists: bool,
        return_points: bool,
        total_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather the neighbors for each query point.

        Args:
            grid (wp.HashGrid): The hash grid to use for the search.
            output_device (torch.device): The device to allocate output tensors on.
            wp_points (wp.array): The points to search in, as a warp array.
            wp_queries (wp.array): The queries to search for, as a warp array.
            wp_offset (wp.array): The offset in output for each input point, as a warp array.
            wp_launch_device (wp.context.Device | None): The device to launch the kernel on.
            wp_launch_stream (wp.Stream | None): The stream to launch the kernel on.
            radius (float): The radius that bounds the search.
            N_queries (int): Total number of query points.
            return_dists (bool): Whether to return the distances of the neighbors.
            return_points (bool): Whether to return the points of the neighbors.
            total_count (int): The total number of neighbors found.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Indices, points, and distances tensors.
        """
        # These three tensors need to persist outside this function, potentially,
        # So they are allocated via torch:
        indices = torch.zeros(
            (
                2,
                total_count,
            ),
            dtype=torch.int32,
            device=output_device,
        )

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

        # Now, kernel selection:
        if not return_dists and not return_points:
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
                ],
                stream=wp_launch_stream,
                device=wp_launch_device,
                block_dim=BLOCK_DIM,
            )

        elif return_dists and not return_points:
            wp.launch(
                kernel=radius_search_unlimited_select_with_dists,
                dim=N_queries,
                inputs=[
                    grid.id,
                    wp_points,
                    wp_queries,
                    wp_offset,
                    wp.from_torch(indices, return_ctype=True),
                    wp.from_torch(distances, return_ctype=True),
                    radius,
                ],
                stream=wp_launch_stream,
                device=wp_launch_device,
                block_dim=BLOCK_DIM,
            )

        elif not return_dists and return_points:
            wp.launch(
                kernel=radius_search_unlimited_select_with_points,
                dim=N_queries,
                inputs=[
                    grid.id,
                    wp_points,
                    wp_queries,
                    wp_offset,
                    wp.from_torch(indices, return_ctype=True),
                    wp.from_torch(points, return_ctype=True),
                    radius,
                ],
                stream=wp_launch_stream,
                device=wp_launch_device,
                block_dim=BLOCK_DIM,
            )

        else:
            wp.launch(
                kernel=radius_search_unlimited_select_with_dists_and_points,
                dim=N_queries,
                inputs=[
                    grid.id,
                    wp_points,
                    wp_queries,
                    wp_offset,
                    wp.from_torch(indices, return_ctype=True),
                    wp.from_torch(distances, return_ctype=True),
                    wp.from_torch(points, return_ctype=True),
                    radius,
                ],
                stream=wp_launch_stream,
                device=wp_launch_device,
                block_dim=BLOCK_DIM,
            )

        # Return all three + one empty tensor for consistency
        # (We could return the proper tensor but it's not needed, and anyways
        # warp is allocating it, not torch, so need to be careful...)
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

        Always returns indices, points, distances.  If return_points is False, points is an empty tensor.
        If return_dists is False, distances is an empty tensor.

        Args:
            points (torch.Tensor): The points to search in.
            queries (torch.Tensor): The queries to search for.
            radius (float): The radius that bounds the search.
            max_points (int | None, optional): The maximum number of points to return per query. If None, unlimited.
            return_dists (bool, optional): Whether to return the distances of the neighbors.
            return_points (bool, optional): Whether to return the points of the neighbors.

        Returns:
            List[torch.Tensor]: [indices, points, distances]
        """

        if points.device != queries.device:
            raise ValueError("points and queries must be on the same device")

        # We're in the warp-backended regime.  So, the first thing to do is to convert these torch tensors to warp
        # These are readonly in warp, allocated with pytorch.
        wp_points = wp.from_torch(points, dtype=wp.vec3)
        wp_queries = wp.from_torch(queries, dtype=wp.vec3, return_ctype=True)

        N_queries = len(queries)

        # Compute follows data.
        # Get the device from queries and the stream from torch
        # This is meant to ensure if this kernel is called from a torch stream context, it uses it.
        if points.device.type == "cuda":
            wp_launch_stream = wp.stream_from_torch(
                torch.cuda.current_stream(points.device)
            )
            wp_launch_device = None  # We explicitly pass None if using the stream.
        else:
            wp_launch_stream = None
            wp_launch_device = "cpu"  # CPUs have no streams

        # We need to create a hash grid:
        grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=wp_points.device)
        grid.reserve(N_queries)
        grid.build(points=wp_points, radius=0.5 * radius)

        # Now, the situations diverge based on max_points.

        if max_points is None:
            total_count, wp_offset = count_neighbors(
                grid,
                wp_points,
                wp_queries,
                wp_launch_device,
                wp_launch_stream,
                radius,
                N_queries,
            )

            if not total_count < 2**31 - 1:
                raise RuntimeError(
                    f"Total found neighbors is too large: {total_count} > 2**31 - 1"
                )

            return gather_neighbors(
                grid,
                points.device,
                wp_points,
                wp_queries,
                wp_offset,
                wp_launch_device,
                wp_launch_stream,
                radius,
                N_queries,
                return_dists,
                return_points,
                total_count,
            )

        else:
            # With a fixed number of output points, we have no need for a second kernel.
            indices = torch.full(
                (N_queries, max_points), 0, dtype=torch.int32, device=points.device
            )
            if return_dists:
                distances = torch.zeros(
                    (N_queries, max_points), dtype=torch.float32, device=points.device
                )
            else:
                distances = torch.empty(0, dtype=torch.float32, device=points.device)
            num_neighbors = torch.zeros(
                (N_queries,), dtype=torch.int32, device=points.device
            )

            if return_points:
                points = torch.zeros(
                    (len(queries), max_points, 3),
                    dtype=torch.float32,
                    device=points.device,
                )
            else:
                points = torch.empty(
                    (0, max_points, 3), dtype=torch.float32, device=points.device
                )
            # This kernel selects up to max_points hits per query.
            # It is not necessarily deterministic.
            # If the number of matches > max_points, you may get different results.

            wp.launch(
                kernel=radius_search_limited_select,
                dim=N_queries,
                inputs=[
                    grid.id,
                    wp_points,
                    wp_queries,
                    max_points,
                    radius,
                    wp.from_torch(indices, return_ctype=True),
                    wp.from_torch(num_neighbors, return_ctype=True),
                    return_dists,
                    wp.from_torch(distances, return_ctype=True),
                    return_points,
                    wp.from_torch(points, return_ctype=True) if return_points else None,
                ],
                stream=wp_launch_stream,
                device=wp_launch_device,
            )

        # Handle the matrix of return values:
        return indices, points, distances, num_neighbors

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

        Args:
            points (torch.Tensor): The points to search in.
            queries (torch.Tensor): The queries to search for.
            radius (float): The radius that bounds the search.
            max_points (int | None, optional): The maximum number of points to return per query. If None, unlimited.
            return_dists (bool, optional): Whether to return the distances of the neighbors.
            return_points (bool, optional): Whether to return the points of the neighbors.

        Returns:
            List[torch.Tensor]: [indices, points, distances]
        """

        if max_points is not None:
            indices = torch.empty(
                queries.shape[0], max_points, dtype=torch.int32, device=queries.device
            )
            if max_points is not None:
                num_neighbors = torch.empty(
                    queries.shape[0], dtype=torch.int32, device=queries.device
                )
            else:
                num_neighbors = torch.empty(0, dtype=torch.int32, device=queries.device)

            if return_dists:
                distances = torch.empty(
                    queries.shape[0],
                    max_points,
                    dtype=torch.float32,
                    device=queries.device,
                )
            else:
                distances = torch.empty(0, dtype=torch.float32, device=queries.device)

            if return_points:
                out_points = torch.empty(
                    queries.shape[0],
                    max_points,
                    3,
                    dtype=torch.float32,
                    device=queries.device,
                )
            else:
                out_points = torch.empty(
                    0, 3, dtype=torch.float32, device=queries.device
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

        Args:
            indexes (torch.Tensor): The indices mapping output points to input points.
            grad_points_out (torch.Tensor): The gradient of the output points.
            points_shape (torch.Size): The shape of the input points tensor.

        Returns:
            torch.Tensor: The gradient with respect to the input points.
        """
        point_grads = torch.zeros(
            points_shape, dtype=grad_points_out.dtype, device=grad_points_out.device
        )

        if grad_points_out.device.type == "cuda":
            wp_launch_stream = wp.stream_from_torch(
                torch.cuda.current_stream(grad_points_out.device)
            )
            wp_launch_device = None  # We explicitly pass None if using the stream.
        else:
            wp_launch_stream = None
            wp_launch_device = "cpu"  # CPUs have no streams

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
            # Flatten the indexes and grad_points.  Launch one thread per element.

            # Don't launch a kernel if there are not points to work on!
            if indexes.shape[1] > 0:
                wp.launch(
                    kernel=scatter_add_unlimited,
                    dim=indexes.shape[1],  # one thread per col of indexes/point_grads
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
            wp.launch(
                kernel=scatter_add,
                dim=indexes.shape[0],  # one thread per row of indexes/point_grads
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

        return point_grads

    @apply_grad_to_points.register_fake
    def apply_grad_to_points_fake(
        indexes: torch.Tensor,
        grad_points_out: torch.Tensor,
        points_shape: List[int],
        max_points: int | None = None,
    ) -> torch.Tensor:
        """
        Fake implementation for apply_grad_to_points for torch.compile/fake tensor support.

        Args:
            indexes (torch.Tensor): The indices mapping output points to input points.
            grad_points_out (torch.Tensor): The gradient of the output points.
            points_shape (torch.Size): The shape of the input points tensor.

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

else:

    def radius_search_impl(
        points: torch.Tensor,
        queries: torch.Tensor,
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ) -> None:
        """
        Dummy implementation for when warp is not available.

        Args:
            points (torch.Tensor): The points to search in.
            queries (torch.Tensor): The queries to search for.
            radius (float): The radius that bounds the search.
            max_points (int | None, optional): The maximum number of points to return per query. If None, unlimited.
            return_dists (bool, optional): Whether to return the distances of the neighbors.
            return_points (bool, optional): Whether to return the points of the neighbors.

        Raises:
            ImportError: If warp is not installed.
        """

        raise ImportError(
            "warp is not installed, can not be used as a backend for a radius search"
        )
