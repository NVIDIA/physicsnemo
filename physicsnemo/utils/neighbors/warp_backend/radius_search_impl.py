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

    wp.init()

    from .kernels import (
        radius_search_count,
        radius_search_limited_select,
        radius_search_unlimited_select,
    )

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

        Implemented with warp kernels.

        Has to return
        """

        # We're in the warp-backended regime.  So, the first thing to do is to convert these torch tensors to warp
        wp_points = wp.from_torch(points, dtype=wp.vec3)
        wp_queries = wp.from_torch(queries, dtype=wp.vec3)

        # Compute follows data.
        # Get the device from queries and the stream from torch
        # This is meant to ensure if this kernel is called from a torch stream context, it uses it.
        wp_device = wp.device_from_torch(queries.device)
        if queries.device.type == "cuda":
            wp_stream = wp.stream_from_torch(torch.cuda.current_stream(queries.device))
            wp_device = None  # We explicitly pass None if using the stream.
        else:
            wp_stream = None
            wp_device = "cpu"  # CPUs have no streams

        # We need to create a hash grid:
        grid = wp.HashGrid(dim_x=100, dim_y=100, dim_z=100, device=wp_device)
        grid.build(points=wp_points, radius=2 * radius)

        # Now, the situations diverge based on max_points.

        if max_points is None:
            # For unlimited output points, we have to go through and count once:
            result_count = torch.zeros(
                len(queries), device=queries.device, dtype=torch.int32
            )

            wp_result_count = wp.from_torch(result_count)

            wp.launch(
                kernel=radius_search_count,
                dim=len(queries),
                inputs=[grid.id, wp_points, wp_queries, wp_result_count, radius],
                stream=wp_stream,
                device=wp_device,
            )

            # Compute the offset from each point to the next point in terms of num neighbors:
            torch_offset = torch.zeros(
                len(result_count) + 1, device=queries.device, dtype=torch.int32
            )

            # Sum the torch output counts:
            torch.cumsum(result_count, dim=0, out=torch_offset[1:])

            # ** This is a GPU -> CPU transfer **
            # ONLY transfer the final index.
            total_count = torch_offset[-1].item()

            if not total_count < 2**31 - 1:
                raise RuntimeError(
                    f"Total result count is too large: {total_count} > 2**31 - 1"
                )

            indices = torch.zeros(
                (
                    2,
                    total_count,
                ),
                dtype=torch.int32,
                device=queries.device,
            )
            distances = torch.zeros(
                (total_count,), dtype=torch.float32, device=queries.device
            )

            if return_points:
                points = torch.zeros(
                    (total_count, 3), dtype=torch.float32, device=queries.device
                )
            else:
                points = torch.empty((0, 3), dtype=torch.float32, device=queries.device)

            wp.launch(
                kernel=radius_search_unlimited_select,
                dim=len(queries),
                inputs=[
                    grid.id,
                    wp_points,
                    wp_queries,
                    wp.from_torch(torch_offset, return_ctype=True),
                    wp.from_torch(indices, return_ctype=True),
                    return_dists,
                    wp.from_torch(distances, return_ctype=True),
                    return_points,
                    wp.from_torch(points, return_ctype=True),
                    radius,
                ],
                stream=wp_stream,
                device=wp_device,
            )

        else:

            # With a fixed number of output points, we have no need for a second kernel.
            indices = torch.full(
                (len(queries), max_points), -1, dtype=torch.int32, device=queries.device
            )
            distances = torch.zeros(
                (len(queries), max_points), dtype=torch.float32, device=queries.device
            )
            num_neighbors = torch.zeros(
                (len(queries),), dtype=torch.int32, device=queries.device
            )

            if return_points:
                points = torch.zeros(
                    (len(queries), max_points, 3),
                    dtype=torch.float32,
                    device=queries.device,
                )
            else:
                points = torch.empty(
                    (0, max_points, 3), dtype=torch.float32, device=queries.device
                )

            wp.launch(
                kernel=radius_search_limited_select,
                dim=len(queries),
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
                stream=wp_stream,
                device=wp_device,
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
