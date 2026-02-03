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

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import radius_search as radius_search_torch
from ._warp_impl import radius_search as radius_search_warp


class RadiusSearch(FunctionSpec):
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

    Note:
        With the Warp backend, there will be an automatic casting of inputs to float32 from reduced precision,
        and results will be returned in their original precision.

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
        implementation (str, optional): Explicit implementation name ("warp" or "torch").
            Defaults to None, which selects by rank.

    Returns:
        tuple: A tuple containing:
            - indices (torch.Tensor): Indices of neighbor points for each query point
            - counts (torch.Tensor): Number of neighbors found for each query point
            - distances (torch.Tensor, optional): Distances to neighbor points if return_dists=True
            - neighbor_points (torch.Tensor, optional): Actual neighbor points if return_points=True

    Raises:
        KeyError: If an explicit implementation name is not registered.
        ImportError: If the selected implementation is unavailable.

    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: torch.Tensor,
        queries: torch.Tensor,
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ):
        return radius_search_warp(
            points, queries, radius, max_points, return_dists, return_points
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        queries: torch.Tensor,
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ):
        return radius_search_torch(
            points, queries, radius, max_points, return_dists, return_points
        )

    @classmethod
    def make_inputs(
        cls,
        device: torch.device | str = "cpu",
    ):
        # TODO(ASV): Populate benchmark inputs in a follow-up PR.
        raise NotImplementedError

    @classmethod
    def compare(
        cls,
        output,
        reference,
    ) -> None:
        # TODO(ASV): Populate output comparison in a follow-up PR.
        raise NotImplementedError


radius_search = RadiusSearch.make_function("radius_search")
