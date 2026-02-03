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

from .utils import format_returns


def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of the radius search.

    This is a brute force implementation that is not memory efficient.
    """

    # Without the compute mode set, this is numerically unstable.
    dists = torch.cdist(
        points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
    )

    if max_points is None:
        # Find all points within radius
        selection = dists <= radius
        selected_indices = torch.nonzero(selection, as_tuple=False).t().contiguous()
        selected_indices = selected_indices[[1, 0], :]

        if return_points:
            points = torch.index_select(points, 0, selected_indices[1])
        else:
            points = torch.empty(
                (0, points.shape[1]), device=points.device, dtype=points.dtype
            )

        if return_dists:
            dists = dists[selection]
        else:
            dists = torch.empty((0,), device=dists.device, dtype=dists.dtype)

    else:
        # Take the max_points lowest distances for each query
        closest_points = torch.topk(
            dists, k=min(max_points, dists.shape[0]), dim=0, largest=False
        )
        values, indices = closest_points
        # Values and indices have shape [max_points, n_queries]
        # The first dim of indices represents the index into input points

        # Filter to points within radius
        selection = values <= radius
        selected_indices = torch.where(selection, indices, 0).t()

        if return_dists:
            dists = torch.where(selection, values, 0).t()
        else:
            dists = torch.empty(
                (0, values.shape[1]), device=values.device, dtype=values.dtype
            )

        if return_points:
            # selected_indices: (num_queries, max_points)
            # points: (num_points, point_dim)
            # We want: selected_points: (num_queries, max_points, point_dim)

            safe_indices = torch.where(selection)
            max_points_loc, queries_loc = safe_indices

            # Use these to get the input points locations:
            input_point_locs = indices[max_points_loc, queries_loc]
            selected_points = points[input_point_locs]
            # Construct default output points:
            output_points = torch.zeros(
                queries.shape[0],
                max_points,
                3,
                device=queries.device,
                dtype=points.dtype,
            )
            # Put the selected points in:
            output_points[queries_loc, max_points_loc] = selected_points

            points = output_points
        else:
            points = torch.empty(
                (0, points.shape[1]), device=points.device, dtype=points.dtype
            )

    return selected_indices, points, dists


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    indices, points_out, distances = radius_search_impl(
        points, queries, radius, max_points, return_dists, return_points
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
