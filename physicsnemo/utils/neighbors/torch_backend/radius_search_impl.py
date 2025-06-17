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


def radius_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """
    Pure pytorch implementation of the radius search.

    This is a brute force implementation that is not memory efficient.
    """

    # Without the compute mode set, this is numerically unstable.
    dists = torch.cdist(points, queries, p=2.0, compute_mode="use_mm_for_euclid_dist")

    if max_points is None:
        # Find all points within radius
        selection = dists <= radius
        selected_indices = torch.nonzero(selection, as_tuple=False).t().contiguous()

        if return_points:
            points = torch.index_select(points, 0, selected_indices[1])

        if return_dists:
            dists = dists[selection]
    else:

        # Take the max_points lowest distances for each query
        closest_points = torch.topk(
            dists, k=min(max_points, dists.shape[0]), dim=0, largest=False
        )
        values, indices = closest_points

        # Filter to points within radius
        selection = values <= radius
        selected_indices = torch.where(selection, indices, -1).t()

        if return_dists:
            dists = torch.where(selection, values, 0).t()

        if return_points:

            points = points[indices].transpose(0, 1)

        print(f"poitns shape: {points.shape}")

        # # Pad with -1 if fewer points found than max_points
        # if selected_indices.shape[0] < max_points:
        #     pad_size = max_points - selected_indices.shape[0]
        #     selected_indices = torch.nn.functional.pad(selected_indices, (0, pad_size), value=-1)
        #     if return_dists:
        #         dists = torch.nn.functional.pad(dists, (0, pad_size), value=0)
        #     if return_points:
        #         points = torch.nn.functional.pad(points, (0, 0, 0, pad_size), value=0)

    # Handle return values
    if return_points:
        if return_dists:
            return selected_indices, points, dists
        return selected_indices, points

    if return_dists:
        return selected_indices, dists

    return selected_indices
