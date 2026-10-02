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

"""Pure-PyTorch reference implementation of contact search."""

import torch

from physicsnemo.nn.functional.neighbors.radius_search.utils import format_returns

from .utils import validate_contact_inputs


def _remove_csr_entries(
    selection: torch.Tensor,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    exclusion_row_stride: int,
) -> None:
    """Clear CSR-addressed entries of a ``(B, N, Q)`` boolean tensor."""
    if exclude_neighbors.numel() == 0:
        return

    row_counts = exclude_offsets[1:] - exclude_offsets[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(
            row_counts.numel(),
            dtype=torch.int32,
            device=exclude_offsets.device,
        ),
        row_counts,
    )

    point_ids = exclude_neighbors.to(torch.int64)
    if exclusion_row_stride == 0:
        query_ids = row_ids.to(torch.int64)
        selection[:, point_ids, query_ids] = False
    else:
        batch_ids = torch.div(row_ids, exclusion_row_stride, rounding_mode="floor").to(
            torch.int64
        )
        query_ids = torch.remainder(row_ids, exclusion_row_stride).to(torch.int64)
        selection[batch_ids, point_ids, query_ids] = False


def contact_search_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Euclidean-radius neighbors after applying CSR exclusions."""
    points, queries, was_unbatched, exclusion_row_stride = validate_contact_inputs(
        points,
        queries,
        radius,
        exclude_neighbors,
        exclude_offsets,
        max_points,
    )

    # Shape (B, N, Q), matching the radius-search reference implementation.
    distances = torch.cdist(
        points, queries, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
    )
    selection = distances <= radius
    _remove_csr_entries(
        selection,
        exclude_neighbors,
        exclude_offsets,
        exclusion_row_stride,
    )

    if max_points is None:
        # nonzero order is (batch, point, query); transpose to public
        # (batch, query, point) COO order without changing pair ordering.
        selected = torch.nonzero(selection, as_tuple=False)
        indices = selected[:, [0, 2, 1]].t().contiguous()

        if return_points:
            points_out = points[selected[:, 0], selected[:, 1]]
        else:
            points_out = torch.empty((0, 3), dtype=points.dtype, device=points.device)

        if return_dists:
            distances_out = distances[selection]
        else:
            distances_out = torch.empty((0,), dtype=points.dtype, device=points.device)

        if was_unbatched:
            indices = indices[1:]

        return indices, points_out, distances_out

    # Exclusions are applied before top-k so excluded points never consume a
    # fixed output slot. Invalid entries use infinity during selection.
    filtered_distances = distances.masked_fill(~selection, torch.inf)
    k = min(max_points, points.shape[1])
    values, indices = torch.topk(
        filtered_distances, k=k, dim=1, largest=False, sorted=True
    )

    if k < max_points:
        pad_size = max_points - k
        values = torch.nn.functional.pad(values, (0, 0, 0, pad_size), value=torch.inf)
        indices = torch.nn.functional.pad(indices, (0, 0, 0, pad_size), value=0)

    valid = torch.isfinite(values)
    # Zero is a valid point index, including for coincident contact pairs.
    # Keep padding distinguishable even when distances and coordinates are zero.
    indices = torch.where(valid, indices, -1).permute(0, 2, 1)

    if return_dists:
        distances_out = torch.where(valid, values, 0).permute(0, 2, 1)
    else:
        distances_out = torch.empty((0,), dtype=points.dtype, device=points.device)

    if return_points:
        safe_locs = torch.where(valid)
        batch_ids, neighbor_slots, query_ids = safe_locs
        point_ids = indices.permute(0, 2, 1)[batch_ids, neighbor_slots, query_ids]
        selected_points = points[batch_ids, point_ids]
        points_out = torch.zeros(
            points.shape[0],
            queries.shape[1],
            max_points,
            3,
            dtype=points.dtype,
            device=points.device,
        )
        points_out[batch_ids, query_ids, neighbor_slots] = selected_points
    else:
        points_out = torch.empty(
            (0, max_points, 3), dtype=points.dtype, device=points.device
        )

    if was_unbatched:
        indices = indices.squeeze(0)
        if return_dists:
            distances_out = distances_out.squeeze(0)
        if return_points:
            points_out = points_out.squeeze(0)

    return indices, points_out, distances_out


def contact_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    exclude_neighbors: torch.Tensor,
    exclude_offsets: torch.Tensor,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """Torch-backend entry point with public return formatting."""
    indices, points_out, distances = contact_search_impl(
        points,
        queries,
        radius,
        exclude_neighbors,
        exclude_offsets,
        max_points,
        return_dists,
        return_points,
    )
    return format_returns(indices, points_out, distances, return_dists, return_points)


__all__ = ["contact_search", "contact_search_impl"]
