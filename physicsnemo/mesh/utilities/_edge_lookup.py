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

"""Edge lookup utilities for efficient edge matching in mesh operations.

This module provides hash-based lookup for finding edges within reference sets,
used throughout physicsnemo.mesh for operations like computing dual volumes,
exterior derivatives, and sharp/flat operators.
"""

import torch


def find_edges_in_reference(
    reference_edges: torch.Tensor,
    query_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find indices of query edges within a reference edge set.

    Uses hash-based lookup with O(n log n) complexity for sorting
    and O(m log n) for queries, where n = len(reference_edges)
    and m = len(query_edges).

    Edge order within each edge is ignored (edges are canonicalized
    to [min_vertex, max_vertex] internally).

    Parameters
    ----------
    reference_edges : torch.Tensor
        Reference edge set, shape (n_ref, 2). Each row is [v0, v1].
    query_edges : torch.Tensor
        Query edges to find, shape (n_query, 2). Each row is [v0, v1].

    Returns
    -------
    indices : torch.Tensor
        Shape (n_query,). For each query edge, the index in reference_edges
        where it was found. For unmatched edges, the value is undefined
        (use the matches mask to filter).
    matches : torch.Tensor
        Shape (n_query,) bool. True if query edge was found in reference_edges.

    Examples
    --------
    >>> ref = torch.tensor([[0, 1], [1, 2], [2, 3]])
    >>> query = torch.tensor([[2, 1], [5, 6], [3, 2]])  # [2,1] matches [1,2]
    >>> indices, matches = find_edges_in_reference(ref, query)
    >>> # indices[0] = 1 (matched), indices[2] = 2 (matched)
    >>> # matches = [True, False, True]
    """
    device = reference_edges.device

    ### Handle empty edge cases
    if len(reference_edges) == 0 or len(query_edges) == 0:
        return (
            torch.zeros(len(query_edges), dtype=torch.long, device=device),
            torch.zeros(len(query_edges), dtype=torch.bool, device=device),
        )

    ### Canonicalize edges to [min_vertex, max_vertex] order
    sorted_reference, _ = torch.sort(reference_edges, dim=-1)
    sorted_query, _ = torch.sort(query_edges, dim=-1)

    ### Compute integer hash for each edge
    # hash = v0 * (max_vertex + 1) + v1
    # This creates a unique mapping for edges with non-negative vertex indices
    max_vertex = max(reference_edges.max().item(), query_edges.max().item()) + 1
    reference_hash = sorted_reference[:, 0] * max_vertex + sorted_reference[:, 1]
    query_hash = sorted_query[:, 0] * max_vertex + sorted_query[:, 1]

    ### Sort reference hashes to enable binary search via searchsorted
    reference_hash_sorted, sort_indices = torch.sort(reference_hash)

    ### Find positions of query hashes in sorted reference
    positions = torch.searchsorted(reference_hash_sorted, query_hash)

    ### Clamp positions to valid range (handles queries beyond max reference)
    positions = positions.clamp(max=len(reference_hash_sorted) - 1)

    ### Verify that found positions are exact matches (not just insertion points)
    matches = reference_hash_sorted[positions] == query_hash

    ### Map back to original reference indices
    indices = sort_indices[positions]

    return indices, matches
