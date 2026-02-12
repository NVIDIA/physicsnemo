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

    Examples
    --------
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

        Returns
        -------
        list[list[int]]
            Ragged list where result[i] contains all neighbors of source i.
            Empty sublists represent sources with no neighbors.

        Examples
        --------
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

    @property
    def counts(self) -> torch.Tensor:
        """Number of neighbors for each source element.

        Returns
        -------
        torch.Tensor
            Shape (n_sources,), dtype int64. counts[i] is the number of
            neighbors for source i.

        Example
        -------
        >>> adj = Adjacency(
        ...     offsets=torch.tensor([0, 3, 3, 5]),
        ...     indices=torch.tensor([1, 2, 0, 4, 3]),
        ... )
        >>> adj.counts.tolist()
        [3, 0, 2]
        """
        return self.offsets[1:] - self.offsets[:-1]

    def expand_to_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand offset-indices encoding to (source_idx, target_idx) pairs.

        This is the inverse of build_adjacency_from_pairs. It produces a pair
        of tensors where (source_indices[i], target_indices[i]) represents the
        i-th edge in the adjacency.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (source_indices, target_indices), both shape (n_total_neighbors,).
            source_indices[i] is the source entity for the i-th pair.
            target_indices[i] is the target entity for the i-th pair.

        Examples
        --------
            >>> adj = Adjacency(
            ...     offsets=torch.tensor([0, 2, 4, 5]),
            ...     indices=torch.tensor([10, 11, 20, 21, 30]),
            ... )
            >>> sources, targets = adj.expand_to_pairs()
            >>> sources.tolist()
            [0, 0, 1, 1, 2]
            >>> targets.tolist()
            [10, 11, 20, 21, 30]
        """
        device = self.offsets.device

        ### Handle empty adjacency
        if self.n_total_neighbors == 0:
            return (
                torch.tensor([], dtype=torch.int64, device=device),
                self.indices,
            )

        ### For each position in indices, find which source it belongs to
        # offsets[i] <= position < offsets[i+1] means position belongs to source i
        # searchsorted(offsets, position, right=True) - 1 gives source index
        positions = torch.arange(
            self.n_total_neighbors, dtype=torch.int64, device=device
        )
        source_indices = torch.searchsorted(self.offsets, positions, right=True) - 1

        return source_indices, self.indices

    def truncate_per_source(self, max_count: int | None = None) -> "Adjacency":
        """Limit each source to at most max_count neighbors.

        This is useful for capping the number of candidates in spatial queries
        (e.g., BVH candidate cells) to prevent memory explosion.

        Parameters
        ----------
        max_count : int | None, optional
            Maximum neighbors per source. If None (default),
            returns self unchanged (no limit applied).

        Returns
        -------
        Adjacency
            New Adjacency with at most max_count neighbors per source.
            If max_count is None, returns self.

        Examples
        --------
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
        if self.n_total_neighbors == 0:
            return Adjacency(offsets=new_offsets, indices=self.indices)

        ### Use expand_to_pairs to get source ID for each position
        source_ids, _ = self.expand_to_pairs()

        # Compute position within source: position - offsets[source_id]
        positions = torch.arange(self.n_total_neighbors, device=device)
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

    Parameters
    ----------
    source_indices : torch.Tensor
        Source entity indices, shape (n_pairs,)
    target_indices : torch.Tensor
        Target entity (neighbor) indices, shape (n_pairs,)
    n_sources : int
        Total number of source entities (may exceed max(source_indices))

    Returns
    -------
    Adjacency
        Adjacency object where adjacency.to_list()[i] contains all targets
        connected from source i. Sources with no outgoing edges have empty lists.

    Examples
    --------
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

    ### Lexicographic sort by (source, target) using two stable argsorts.
    # This avoids the int64 overflow that occurs with the composite-key
    # approach (source * max_target + target) when indices exceed ~3 × 10^9.
    sort_by_target = torch.argsort(target_indices, stable=True)
    sort_indices = sort_by_target[
        torch.argsort(source_indices[sort_by_target], stable=True)
    ]

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
