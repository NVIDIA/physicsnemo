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

"""Core data structure for storing ragged adjacency relationships in meshes.

This module provides the Adjacency tensorclass for representing ragged arrays
using offset-indices encoding, commonly used in graph and mesh processing.
"""

import torch
from tensordict import tensorclass


@tensorclass
class Adjacency:
    """Ragged adjacency list stored with offset-indices encoding.

    This structure efficiently represents variable-length neighbor lists using two
    arrays: offsets and indices. This is a standard format for sparse graph data
    structures and enables GPU-compatible operations on ragged data.

    Attributes:
        offsets: Indices into the indices array marking the start of each neighbor list.
            Shape (n_sources + 1,), dtype int64. The i-th source's neighbors are
            indices[offsets[i]:offsets[i+1]].
        indices: Flattened array of all neighbor indices.
            Shape (total_neighbors,), dtype int64.

    Example:
        >>> # Represent [[0,1,2], [3,4], [5], [6,7,8]]
        >>> adj = Adjacency(
        ...     offsets=torch.tensor([0, 3, 5, 6, 9]),
        ...     indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ... )
        >>> adj.to_list()
        [[0, 1, 2], [3, 4], [5], [6, 7, 8]]

        >>> # Empty neighbor list for source 2
        >>> adj = Adjacency(
        ...     offsets=torch.tensor([0, 2, 2, 4]),
        ...     indices=torch.tensor([10, 11, 12, 13]),
        ... )
        >>> adj.to_list()
        [[10, 11], [], [12, 13]]
    """

    offsets: torch.Tensor  # shape: (n_sources + 1,), dtype: int64
    indices: torch.Tensor  # shape: (total_neighbors,), dtype: int64

    def __post_init__(self):
        if not torch.compiler.is_compiling():
            ### Validate offsets is non-empty
            # Offsets must have length (n_sources + 1), so minimum length is 1 (for n_sources=0)
            if len(self.offsets) < 1:
                raise ValueError(
                    f"Offsets array must have length >= 1 (n_sources + 1), but got {len(self.offsets)=}. "
                    f"Even for 0 sources, offsets should be [0]."
                )

            ### Validate offsets starts at 0
            if self.offsets[0].item() != 0:
                raise ValueError(
                    f"First offset must be 0, but got {self.offsets[0].item()=}. "
                    f"The offset-indices encoding requires offsets[0] == 0."
                )

            ### Validate last offset equals length of indices
            last_offset = self.offsets[-1].item()
            indices_length = len(self.indices)
            if last_offset != indices_length:
                raise ValueError(
                    f"Last offset must equal length of indices, but got "
                    f"{last_offset=} != {indices_length=}. "
                    f"The offset-indices encoding requires offsets[-1] == len(indices)."
                )

    def to_list(self) -> list[list[int]]:
        """Convert adjacency to a ragged list-of-lists representation.

        This method is primarily for testing and comparison with other libraries.
        The order of neighbors within each sublist is preserved (not sorted).

        This is, in general, much less efficient than directly using the sparse encoding
        itself -- all internal library operations use Adjacency objects directly.

        Returns:
            Ragged list where result[i] contains all neighbors of source i.
            Empty sublists represent sources with no neighbors.

        Example:
            >>> adj = Adjacency(
            ...     offsets=torch.tensor([0, 3, 3, 5]),
            ...     indices=torch.tensor([1, 2, 0, 4, 3]),
            ... )
            >>> adj.to_list()
            [[1, 2, 0], [], [4, 3]]
        """
        ### Convert to CPU numpy for Python list operations
        offsets_np = self.offsets.cpu().numpy()
        indices_np = self.indices.cpu().numpy()

        ### Build ragged list structure
        n_sources = len(offsets_np) - 1
        result = []
        for i in range(n_sources):
            start = offsets_np[i]
            end = offsets_np[i + 1]
            neighbors = indices_np[start:end].tolist()
            result.append(neighbors)

        return result

    @property
    def n_sources(self) -> int:
        """Number of source elements (points or cells) in the adjacency."""
        return len(self.offsets) - 1

    @property
    def n_total_neighbors(self) -> int:
        """Total number of neighbor relationships across all sources."""
        return len(self.indices)

    def truncate_per_source(self, max_count: int | None = None) -> "Adjacency":
        """Limit each source to at most max_count neighbors.

        This is useful for capping the number of candidates in spatial queries
        (e.g., BVH candidate cells) to prevent memory explosion.

        Args:
            max_count: Maximum neighbors per source. If None (default),
                returns self unchanged (no limit applied).

        Returns:
            New Adjacency with at most max_count neighbors per source.
            If max_count is None, returns self.

        Example:
            >>> adj = Adjacency(
            ...     offsets=torch.tensor([0, 5, 8, 10]),
            ...     indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ... )
            >>> adj.to_list()
            [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]]
            >>> adj.truncate_per_source(2).to_list()
            [[0, 1], [5, 6], [8, 9]]
        """
        if max_count is None:
            return self

        device = self.offsets.device

        ### Compute counts per source
        counts = self.offsets[1:] - self.offsets[:-1]

        ### Clamp counts to max_count
        clamped_counts = torch.clamp(counts, max=max_count)

        ### Build new offsets from clamped counts
        new_offsets = torch.zeros_like(self.offsets)
        new_offsets[1:] = torch.cumsum(clamped_counts, dim=0)

        ### Build mask for which indices to keep
        # For each index position, determine which source it belongs to
        # and its position within that source
        n_indices = len(self.indices)
        if n_indices == 0:
            return Adjacency(offsets=new_offsets, indices=self.indices)

        positions = torch.arange(n_indices, device=device)

        # Find source ID for each position using searchsorted
        # offsets[1:] gives the exclusive end of each source's range
        source_ids = torch.searchsorted(self.offsets[1:], positions, right=False)

        # Compute position within source: position - offsets[source_id]
        within_source_pos = positions - self.offsets[source_ids]

        # Keep only positions where within_source_pos < max_count
        keep_mask = within_source_pos < max_count

        return Adjacency(
            offsets=new_offsets,
            indices=self.indices[keep_mask],
        )


def build_adjacency_from_pairs(
    source_indices: torch.Tensor,  # shape: (n_pairs,)
    target_indices: torch.Tensor,  # shape: (n_pairs,)
    n_sources: int,
) -> Adjacency:
    """Build offset-index adjacency from (source, target) pairs.

    This utility consolidates the common pattern of constructing an Adjacency object
    from a list of directed edges (source → target pairs).

    Algorithm:
        1. Sort pairs by source index (then by target for consistency)
        2. Use bincount to count neighbors per source
        3. Use cumsum to compute offsets
        4. Return Adjacency with sorted neighbor lists

    Args:
        source_indices: Source entity indices, shape (n_pairs,)
        target_indices: Target entity (neighbor) indices, shape (n_pairs,)
        n_sources: Total number of source entities (may exceed max(source_indices))

    Returns:
        Adjacency object where adjacency.to_list()[i] contains all targets
        connected from source i. Sources with no outgoing edges have empty lists.

    Example:
        >>> # Create adjacency: 0→[1,2], 1→[3], 2→[], 3→[0]
        >>> sources = torch.tensor([0, 0, 1, 3])
        >>> targets = torch.tensor([1, 2, 3, 0])
        >>> adj = build_adjacency_from_pairs(sources, targets, n_sources=4)
        >>> adj.to_list()
        [[1, 2], [3], [], [0]]
    """
    device = source_indices.device

    ### Handle empty pairs
    if len(source_indices) == 0:
        return Adjacency(
            offsets=torch.zeros(n_sources + 1, dtype=torch.int64, device=device),
            indices=torch.zeros(0, dtype=torch.int64, device=device),
        )

    ### Sort by (source, target) for grouping
    # Use lexicographic sort: sort by source first, then by target
    # Multiply source by (max_target + 1) to ensure source dominates in sort order
    max_target = target_indices.max().item() if len(target_indices) > 0 else 0
    sort_keys = source_indices * (max_target + 2) + target_indices
    sort_indices = torch.argsort(sort_keys)

    sorted_sources = source_indices[sort_indices]
    sorted_targets = target_indices[sort_indices]

    ### Compute offsets for each source
    # offsets[i] marks the start of source i's neighbor list
    offsets = torch.zeros(n_sources + 1, dtype=torch.int64, device=device)

    # Count occurrences of each source index
    source_counts = torch.bincount(sorted_sources, minlength=n_sources)

    # Cumulative sum to get offsets
    offsets[1:] = torch.cumsum(source_counts, dim=0)

    return Adjacency(
        offsets=offsets,
        indices=sorted_targets,
    )
