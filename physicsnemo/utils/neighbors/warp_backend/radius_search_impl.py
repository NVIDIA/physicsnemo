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
This file contains the interface between pytorch and warp kernels.

It uses a mix of utilities, such that it needs to be opaque to pure pytorch.
At the same time, we want to rely on pytorch's memory allocation as much as possible 
and not warp.  So, tensor creation and allocation is driven by torch, and 
passed to warp for computation.
"""

import torch

from physicsnemo.utils.version_check import check_min_version

WARP_AVAILABLE = check_min_version("warp", "0.6.0")

if WARP_AVAILABLE:

    import warp as wp

    wp.config.quiet = True
    # wp.config.lineinfo = True

    wp.init()

    BLOCK_DIM = 32

    from .kernels import (
        radius_search_count,
        radius_search_limited_select,
        radius_search_unlimited_select,
        radius_search_unlimited_select_with_dists,
        radius_search_unlimited_select_with_dists_and_points,
        radius_search_unlimited_select_with_points,
    )

    def count_neighbors(
        grid: wp.HashGrid,
        wp_points: wp.array(dtype=wp.vec3),
        wp_queries: wp.array(dtype=wp.vec3),
        wp_launch_device: wp.context.Device | None,
        wp_launch_stream: wp.Stream | None,
        radius: float,
        N_queries: int,
    ) -> tuple[int, wp.array(dtype=wp.int32)]:

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
    ) -> wp.array | tuple[wp.array]:
        """
        Do the actual gathering of neighbors.

        Select a kernel based on the return_dists and return_points flags.

        Args:
            grid: The hash grid to use for the search.
            wp_points: The points to search in, warp array.
            wp_queries: The queries to search for, warp array.
            wp_offset: The offset in output for each input point , warp array.
            wp_launch_device: The device to launch the kernel on.
            wp_launch_stream: The stream to launch the kernel on.
            radius: The radius that bounds the search.
            N_queries: Total number of query points
            return_dists: Whether to return the distances of the neighbors.
            return_points: Whether to return the points of the neighbors.
            total_count: The total number of neighbors found.
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

        if return_points:
            points = torch.zeros(
                (total_count, 3), dtype=torch.float32, device=output_device
            )

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

            return indices

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

            return indices, distances
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

            return indices, points

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

            return indices, points, distances

    def radius_search_impl(
        points: torch.Tensor,
        queries: torch.Tensor,
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ):
        """
        Find and return the nearest neighbors in `points` using locations from `queries`.

        Implemented with warp kernels.  Make sure points and queries are on the same device.
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
        grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64, device=wp_points.device)
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

            # if not total_count < 2**31 - 1:
            #     raise RuntimeError(
            #         f"Total result count is too large: {total_count} > 2**31 - 1"
            #     )

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
                (N_queries, max_points), -1, dtype=torch.int32, device=points.device
            )
            distances = torch.zeros(
                (N_queries, max_points), dtype=torch.float32, device=points.device
            )
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
        if return_points:

            if return_dists:
                # Everything
                return indices, points, distances

            return indices, points

        if return_dists:
            return indices, distances

        # Always indices
        return indices

else:

    def radius_search_impl(
        points: torch.Tensor,
        queries: torch.Tensor,
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ):
        """ """

        raise ImportError(
            "warp is not installed, can not be used as a backend for a radius search"
        )
