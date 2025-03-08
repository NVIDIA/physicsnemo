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

import torch
import warp as wp


@wp.kernel
def radius_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_count: wp.array(dtype=wp.int32),
    radius: wp.float32,
):
    """
    Warp kernel for counting the number of points within a specified radius
    for each query point, using a hash grid for spatial queries.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_count: An array to store the count of neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count_tid = int(0)

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_count_tid += 1

    result_count[tid] = result_count_tid


@wp.kernel
def radius_search_query(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    radius: wp.float32,
):
    """
    Warp kernel for performing radius search queries on a set of points,
    storing the results of neighboring points within a specified radius.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_offset: An array to store the offset in the results array for each query point.
        result_point_idx: An array to store the indices of neighboring points found within the radius for each query point.
        result_point_dist: An array to store the distances to neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_point_idx[offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_count += 1


def radius_search(
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: float,
    grid_dim: int | tuple[int, int, int] = (128, 128, 128),
    device: str = "cuda",
):
    """
    Performs a radius search for each query point within a specified radius,
    using a hash grid for efficient spatial querying.

    Args:
        points: An array of points in space.
        queries: An array of query points.
        radius: The search radius around each query point.
        grid_dim: The dimensions of the hash grid, either as an integer or a tuple of three integers.
        device: The device (e.g., 'cuda' or 'cpu') on which computations are performed.

    Returns:
        A tuple containing the indices of neighboring points, their distances to the query points, and an offset array for result indexing.
    """
    # convert grid_dim to Tuple if it is int
    if isinstance(grid_dim, int):
        grid_dim = (grid_dim, grid_dim, grid_dim)

    result_count = wp.zeros(shape=queries.shape, dtype=wp.int32)
    grid = wp.HashGrid(
        dim_x=grid_dim[0],
        dim_y=grid_dim[1],
        dim_z=grid_dim[2],
        device=device,
    )
    grid.build(points=points, radius=2 * radius)

    wp.launch(
        kernel=radius_search_count,
        dim=len(queries),
        inputs=[grid.id, points, queries, result_count, radius],
        device=device,
    )

    torch_offset = torch.zeros(len(result_count) + 1, device=device, dtype=torch.int32)
    result_count_torch = wp.to_torch(result_count)
    torch.cumsum(result_count_torch, dim=0, out=torch_offset[1:])
    total_count = torch_offset[-1].item()
    if not total_count < 2**31 - 1:
        raise RuntimeError(
            f"Total result count is too large: {total_count} > 2**31 - 1"
        )
    offset = wp.from_torch(torch_offset)

    result_point_idx = wp.zeros(shape=(total_count,), dtype=wp.int32)
    result_point_dist = wp.zeros(shape=(total_count,), dtype=wp.float32)

    wp.launch(
        kernel=radius_search_query,
        dim=len(queries),
        inputs=[
            grid.id,
            points,
            queries,
            offset,
            result_point_idx,
            result_point_dist,
            radius,
        ],
        device=device,
    )

    return (
        result_point_idx,
        result_point_dist,
        wp.from_torch(torch_offset, dtype=wp.int32),
    )
