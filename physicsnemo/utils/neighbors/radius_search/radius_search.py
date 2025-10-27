# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Literal

import torch

from ._torch_impl import radius_search_impl as radius_search_torch
from ._warp_impl import radius_search_impl as radius_search_warp


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
    backend: Literal["warp", "torch"] = "warp",
) -> tuple[torch.Tensor]:
    """Performs radius-based neighbor search to find points within a specified radius of query points.

    Can use brute-force methods with PyTorch, or an accelerated spatial decomposition method with Warp.

    This function does not currently accept a batch index.

    This function has differing behavior based on the argument for max_points.  If max_points is None,
    the function will find ALL points within the radius and return a flattened list of indices,
    (optionally) distances, and (optionally) points.  The indices will have a shape of
    (2, N) where N is the aggregate number of neighbors found for all queries.  The 0th index of the
    output represents the index of the query points, and the 1st index represents the index of the
    neighbor points within the search space.

    If max_points is not None, the function will find the max_points closest points within the radius
    and return a statically sized array of indices, (optionally) distances, and (optionally) points.
    The indices will have a shape of (queries.shape[0], max_points).  Each row i of the indices will be
    neighbors of queries[i]. If there are fewer points than max_points, then the unused indices will be
    set to -1 and the distances and points will be set to 0 for unused points.

    Because the shape when max_points=None is dynamic, this function is incompatible with torch.compile
    in that case.  When max_points is set, this function is compatible with torch.compile regardless of
    backend.

    The different backends are not necessarily certain to provide identical output, for two reasons:
    first, if max_points is lower than the number of neighbors found, the selected points may be
    stochastic.  Second, when max_points is None or max_points is greater than the number of neighbors,
    the outputs may be ordered differently by the two backends.  Do not rely on the exact order of
    the neighbors in the outputs.

    Args:
        points (torch.Tensor): The reference point cloud tensor of shape (N, 3) where N is the number
            of points.
        queries (torch.Tensor): The query points tensor of shape (M, 3) where M is the number of
            query points.
        radius (float): The search radius. Points within or at this radius of a query point will be
            considered neighbors.
        max_points (int | None, optional): Maximum number of neighbors to return for each query point.
            If None, returns all neighbors within radius. Defaults to None.  See documentation for details.
        return_dists (bool, optional): If True, returns the distances to the neighbor points.
            Defaults to False.
        return_points (bool, optional): If True, returns the actual neighbor points in addition to
            their indices. Defaults to False.
        backend (Literal["warp", "torch"], optional): The backend implementation to use for the search.
            Either "warp" or "torch". Defaults to "warp".

    Returns:
        tuple: A tuple containing:
            - indices (torch.Tensor): Indices of neighbor points for each query point
            - counts (torch.Tensor): Number of neighbors found for each query point
            - distances (torch.Tensor, optional): Distances to neighbor points if return_dists=True
            - neighbor_points (torch.Tensor, optional): Actual neighbor points if return_points=True

    Raises:
        ValueError: If backend is not "warp" or "torch"

    """

    if backend not in ["warp", "torch"]:
        raise ValueError(
            f"`radius_search` backend must be either 'warp' or 'torch', got {backend=}"
        )

    # Num neighbors is returned, because in the warp version
    # it's essential to get the backwards pass right.

    # We never actually return it from here.
    # (If you update to use it in the future, check it carefully!)

    if backend == "warp":
        indices, points, distances, num_neighbors = radius_search_warp(
            points, queries, radius, max_points, return_dists, return_points
        )
    elif backend == "torch":
        indices, points, distances = radius_search_torch(
            points, queries, radius, max_points, return_dists, return_points
        )

    # Handle return values
    if return_points:
        if return_dists:
            return indices, points, distances
        return indices, points

    if return_dists:
        return indices, distances

    return indices
