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


import torch

from .utils import format_returns, validate_inputs


def _radius_search_dynamic(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    return_dists: bool,
    return_points: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-element dynamic radius search on (N, 3) and (Q, 3) tensors.

    Finds ALL neighbors within radius (variable-length output).
    """

    # Without the compute mode set, this is numerically unstable.
    dists = torch.cdist(
        points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
    )

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

    return selected_indices, points, dists


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

    Accepts both unbatched (N, 3) and batched (B, N, 3) inputs.
    This is a brute force implementation that is not memory efficient.
    """

    points, queries, was_unbatched = validate_inputs(points, queries)
    B = points.shape[0]

    if max_points is None:
        # Dynamic output: loop over batch, concatenate with batch indices
        all_indices = []
        all_pts = []
        all_dists = []
        for b in range(B):
            idx_b, pts_b, dists_b = _radius_search_dynamic(
                points[b], queries[b], radius, return_dists, return_points
            )
            # idx_b is (2, count_b); prepend a batch-index row
            batch_row = torch.full(
                (1, idx_b.shape[1]), b, dtype=idx_b.dtype, device=idx_b.device
            )
            all_indices.append(torch.cat([batch_row, idx_b], dim=0))
            all_pts.append(pts_b)
            all_dists.append(dists_b)

        selected_indices = torch.cat(all_indices, dim=1)  # (3, total_count)
        pts_out = torch.cat(all_pts, dim=0) if return_points else all_pts[0]
        dists_out = torch.cat(all_dists, dim=0) if return_dists else all_dists[0]

        if was_unbatched:
            # Strip the batch-index row to restore (2, count) format
            selected_indices = selected_indices[1:]

        return selected_indices, pts_out, dists_out

    # Deterministic output: fully batched via cdist + topk
    # dists: (B, N, Q)
    dists = torch.cdist(
        points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
    )

    # topk along dim=1 (points dim): (B, max_points, Q)
    k = min(max_points, dists.shape[1])
    values, indices = torch.topk(dists, k=k, dim=1, largest=False)

    # Pad if k < max_points (fewer points than requested)
    if k < max_points:
        pad_size = max_points - k
        values = torch.nn.functional.pad(values, (0, 0, 0, pad_size), value=0.0)
        indices = torch.nn.functional.pad(indices, (0, 0, 0, pad_size), value=0)

    # Filter to within radius: (B, max_points, Q)
    selection = values <= radius
    selected_indices = torch.where(selection, indices, 0)
    # Transpose to (B, Q, max_points)
    selected_indices = selected_indices.permute(0, 2, 1)

    if return_dists:
        dists_out = torch.where(selection, values, 0).permute(0, 2, 1)
    else:
        dists_out = torch.empty(0, dtype=dists.dtype, device=dists.device)

    if return_points:
        # Gather points for each (batch, query, neighbor) triple
        # selection: (B, max_points, Q), indices: (B, max_points, Q)
        safe_locs = torch.where(selection)
        batch_loc, mp_loc, query_loc = safe_locs
        input_point_locs = indices[batch_loc, mp_loc, query_loc]
        selected_points = points[batch_loc, input_point_locs]
        output_points = torch.zeros(
            B,
            queries.shape[1],
            max_points,
            3,
            device=queries.device,
            dtype=points.dtype,
        )
        output_points[batch_loc, query_loc, mp_loc] = selected_points
        pts_out = output_points
    else:
        pts_out = torch.empty(
            0, max_points, 3, device=points.device, dtype=points.dtype
        )

    if was_unbatched:
        selected_indices = selected_indices.squeeze(0)
        if return_dists:
            dists_out = dists_out.squeeze(0)
        if return_points:
            pts_out = pts_out.squeeze(0)

    return selected_indices, pts_out, dists_out


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """Torch-backend entry point for radius search with formatted returns."""
    indices, points_out, distances = radius_search_impl(
        points, queries, radius, max_points, return_dists, return_points
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)
