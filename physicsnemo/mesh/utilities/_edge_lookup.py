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
from jaxtyping import Bool, Int


def find_edges_in_reference(
    reference_edges: Int[torch.Tensor, "n_ref 2"],
    query_edges: Int[torch.Tensor, "n_queries 2"],
    index_bound: int | None = None,
) -> tuple[Int[torch.Tensor, " n_queries"], Bool[torch.Tensor, " n_queries"]]:
    r"""Find indices of query edges within a reference edge set.

    Uses hash-based lookup with :math:`O(n \log n)` complexity for sorting and
    :math:`O(m \log n)` for queries, where ``n = len(reference_edges)`` and
    ``m = len(query_edges)``.

    Edge order within each edge is ignored (edges are canonicalized to
    ascending vertex-index order internally).

    Parameters
    ----------
    reference_edges : Int[torch.Tensor, "n_ref 2"]
        Reference edge set. Each row is ``[v0, v1]``.
    query_edges : Int[torch.Tensor, "n_queries 2"]
        Query edges to find. Each row is ``[v0, v1]``.
    index_bound : int, optional
        Strict upper bound for vertex indices in both edge tensors. Passing
        this avoids a GPU synchronization when hashing the edges.

    Returns
    -------
    indices : Int[torch.Tensor, " n_queries"]
        For each query edge, the index in ``reference_edges`` where it was
        found. For unmatched edges, the value is undefined (use the
        ``matches`` mask to filter).
    matches : Bool[torch.Tensor, " n_queries"]
        ``True`` if query edge was found in ``reference_edges``.

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

    ### Canonicalize edges to ascending vertex-index order
    sorted_reference, _ = torch.sort(reference_edges, dim=-1)
    sorted_query, _ = torch.sort(query_edges, dim=-1)

    ### Compute integer hash for each edge
    # hash = v0 * index_bound + v1
    # This creates a unique mapping for edges with non-negative vertex indices
    if index_bound is None:
        index_bound = (
            int(max(reference_edges.max().item(), query_edges.max().item())) + 1
        )
    reference_hash = sorted_reference[:, 0] * index_bound + sorted_reference[:, 1]
    query_hash = sorted_query[:, 0] * index_bound + sorted_query[:, 1]

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
