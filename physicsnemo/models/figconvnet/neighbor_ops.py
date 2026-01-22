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

r"""Neighbor search operations for FIGConvNet.

This module provides efficient neighbor search implementations used for
point-grid convolution operations in the FIGConvNet architecture.

The main functions are:

- :func:`neighbor_radius_search`: Find neighbors within a radius
- :func:`batched_neighbor_radius_search`: Batched radius search
- :func:`neighbor_knn_search`: Find k-nearest neighbors
- :func:`batched_neighbor_knn_search`: Batched KNN search

The main classes are:

- :class:`NeighborSearchReturn`: Container for neighbor search results
"""

# ruff: noqa: S101,F722
from typing import Literal, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from physicsnemo.models.figconvnet.warp_neighbor_search import (
    batched_radius_search_warp,
    radius_search_warp,
)


class NeighborSearchReturn:
    r"""Container for neighbor search results.

    This class wraps the output of neighbor search operations, providing
    a consistent interface for accessing neighbor indices and row splits
    (CSR format offsets).

    Parameters
    ----------
    *args : Union[Tuple[Tensor, Tensor], NeighborSearchReturn]
        Either two tensors (neighbors_index, neighbors_row_splits) or
        another NeighborSearchReturn object to copy from.

    Attributes
    ----------
    neighbors_index : torch.Tensor
        Flat tensor of neighbor indices of shape :math:`(N_{total},)` where
        :math:`N_{total}` is the total number of neighbor pairs.
    neighbors_row_splits : torch.Tensor
        CSR format offsets of shape :math:`(M + 1,)` where :math:`M` is the
        number of query points. ``neighbors_row_splits[i]`` gives the start
        index in ``neighbors_index`` for query point ``i``.

    Examples
    --------
    >>> import torch
    >>> # 3 query points with 2, 3, 1 neighbors respectively
    >>> indices = torch.tensor([0, 1, 2, 3, 4, 5])
    >>> row_splits = torch.tensor([0, 2, 5, 6])
    >>> result = NeighborSearchReturn(indices, row_splits)
    >>> result.neighbors_index
    tensor([0, 1, 2, 3, 4, 5])

    Note
    ----
    The CSR (Compressed Sparse Row) format is efficient for variable-length
    neighbor lists, as it avoids padding overhead.
    """

    # N is the total number of neighbors for all M queries
    _neighbors_index: Int[Tensor, "N"]  # noqa: F821
    # M is the number of queries
    _neighbors_row_splits: Int[Tensor, "M + 1"]  # noqa: F821

    def __init__(self, *args):
        """Initialize NeighborSearchReturn.

        Parameters
        ----------
        *args : tuple
            Either (neighbors_index, neighbors_row_splits) tensors or
            a single NeighborSearchReturn object.

        Raises
        ------
        ValueError
            If not initialized with 1 or 2 arguments.
        """
        # If there are two args, assume they are neighbors_index and neighbors_row_splits
        # If there is one arg, assume it is a NeighborSearchReturnType
        if len(args) == 2:
            self._neighbors_index = args[0].long()
            self._neighbors_row_splits = args[1].long()
        elif len(args) == 1:
            self._neighbors_index = args[0].neighbors_index.long()
            self._neighbors_row_splits = args[0].neighbors_row_splits.long()
        else:
            raise ValueError(
                "NeighborSearchReturn must be initialized with 1 or 2 arguments"
            )

    @property
    def neighbors_index(self) -> Int[Tensor, "N"]:  # noqa: F821
        r"""Get the neighbor indices tensor.

        Returns
        -------
        torch.Tensor
            Flat tensor of neighbor indices.
        """
        return self._neighbors_index

    @property
    def neighbors_row_splits(self) -> Int[Tensor, "M + 1"]:  # noqa: F821
        r"""Get the row splits tensor.

        Returns
        -------
        torch.Tensor
            CSR format offsets for each query point.
        """
        return self._neighbors_row_splits

    def to(self, device: Union[str, int, torch.device]) -> "NeighborSearchReturn":
        r"""Move tensors to specified device.

        Parameters
        ----------
        device : Union[str, int, torch.device]
            Target device.

        Returns
        -------
        NeighborSearchReturn
            Self, with tensors on target device.
        """
        self._neighbors_index = self._neighbors_index.to(device)
        self._neighbors_row_splits = self._neighbors_row_splits.to(device)
        return self


def neighbor_radius_search(
    inp_positions: Float[Tensor, "N 3"],
    out_positions: Float[Tensor, "M 3"],
    radius: float,
    search_method: Literal["warp"] = "warp",
) -> NeighborSearchReturn:
    r"""Find neighbors within a radius for each query point.

    For each point in ``out_positions``, finds all points in ``inp_positions``
    that are within the specified radius.

    Parameters
    ----------
    inp_positions : torch.Tensor
        Reference point positions of shape :math:`(N, 3)`.
    out_positions : torch.Tensor
        Query point positions of shape :math:`(M, 3)`.
    radius : float
        Search radius.
    search_method : Literal["warp"], optional, default="warp"
        Backend for neighbor search. Currently only "warp" is supported.

    Returns
    -------
    NeighborSearchReturn
        Container with neighbor indices and row splits.

    Raises
    ------
    ValueError
        If an unsupported search method is specified.

    Examples
    --------
    >>> import torch
    >>> inp = torch.rand(1000, 3).cuda()
    >>> out = torch.rand(100, 3).cuda()
    >>> neighbors = neighbor_radius_search(inp, out, radius=0.1)
    >>> neighbors.neighbors_index.shape  # Variable based on data
    torch.Size([...])

    Note
    ----
    This function sets the CUDA device based on input tensor location,
    which is critical for multi-GPU setups.
    """
    # Critical for multi GPU
    if inp_positions.is_cuda:
        torch.cuda.set_device(inp_positions.device)
    if inp_positions.device != out_positions.device:
        raise ValueError(
            f"Device mismatch: inp_positions on {inp_positions.device}, "
            f"out_positions on {out_positions.device}"
        )

    if search_method == "warp":
        neighbor_index, neighbor_distance, neighbor_split = radius_search_warp(
            inp_positions, out_positions, radius
        )
    else:
        raise ValueError(f"search_method {search_method} not supported.")

    neighbors = NeighborSearchReturn(neighbor_index, neighbor_split)
    return neighbors


@torch.no_grad()
def batched_neighbor_radius_search(
    inp_positions: Float[Tensor, "B N 3"],
    out_positions: Float[Tensor, "B M 3"],
    radius: float,
    search_method: Literal["warp"] = "warp",
) -> NeighborSearchReturn:
    r"""Batched radius neighbor search.

    Performs radius search for each batch element, with results concatenated
    and indices offset appropriately.

    Parameters
    ----------
    inp_positions : torch.Tensor
        Reference point positions of shape :math:`(B, N, 3)`.
    out_positions : torch.Tensor
        Query point positions of shape :math:`(B, M, 3)`.
    radius : float
        Search radius.
    search_method : Literal["warp"], optional, default="warp"
        Backend for neighbor search.

    Returns
    -------
    NeighborSearchReturn
        Container with neighbor indices (offset by batch) and row splits.

    Raises
    ------
    AssertionError
        If batch sizes don't match.
    ValueError
        If an unsupported search method is specified.

    Examples
    --------
    >>> import torch
    >>> inp = torch.rand(4, 1000, 3).cuda()
    >>> out = torch.rand(4, 100, 3).cuda()
    >>> neighbors = batched_neighbor_radius_search(inp, out, radius=0.1)
    """
    if inp_positions.shape[0] != out_positions.shape[0]:
        raise ValueError(
            f"Batch size mismatch, {inp_positions.shape[0]} != {out_positions.shape[0]}"
        )

    if search_method == "warp":
        neighbor_index, neighbor_dist, neighbor_offset = batched_radius_search_warp(
            inp_positions, out_positions, radius
        )
    else:
        raise ValueError(f"search_method {search_method} not supported.")

    return NeighborSearchReturn(neighbor_index, neighbor_offset)


@torch.no_grad()
def _knn_search(
    ref_positions: Float[Tensor, "N 3"],
    query_positions: Float[Tensor, "M 3"],
    k: int,
) -> Int[Tensor, "M K"]:
    r"""Perform k-nearest neighbor search using distance computation.

    Parameters
    ----------
    ref_positions : torch.Tensor
        Reference point positions of shape :math:`(N, 3)`.
    query_positions : torch.Tensor
        Query point positions of shape :math:`(M, 3)`.
    k : int
        Number of nearest neighbors to find.

    Returns
    -------
    torch.Tensor
        Neighbor indices of shape :math:`(M, K)`.

    Raises
    ------
    AssertionError
        If k is invalid or tensors are on different devices.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= ref_positions.shape[0]:
        raise ValueError(
            f"k ({k}) must be less than number of reference points ({ref_positions.shape[0]})"
        )
    if ref_positions.device != query_positions.device:
        raise ValueError(
            f"Device mismatch: ref_positions on {ref_positions.device}, "
            f"query_positions on {query_positions.device}"
        )

    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)

    # Use topk to get the top k indices from distances
    dists = torch.cdist(query_positions, ref_positions)
    _, neighbors_index = torch.topk(dists, k, dim=1, largest=False)

    return neighbors_index


@torch.no_grad()
def _chunked_knn_search(
    ref_positions: Float[Tensor, "N 3"],
    query_positions: Float[Tensor, "M 3"],
    k: int,
    chunk_size: int = 4096,
) -> Int[Tensor, "M K"]:
    r"""Perform chunked k-nearest neighbor search for memory efficiency.

    Divides query points into chunks to avoid memory issues with large
    distance matrices.

    Parameters
    ----------
    ref_positions : torch.Tensor
        Reference point positions of shape :math:`(N, 3)`.
    query_positions : torch.Tensor
        Query point positions of shape :math:`(M, 3)`.
    k : int
        Number of nearest neighbors to find.
    chunk_size : int, optional, default=4096
        Maximum number of query points to process at once.

    Returns
    -------
    torch.Tensor
        Neighbor indices of shape :math:`(M, K)`.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= ref_positions.shape[0]:
        raise ValueError(
            f"k ({k}) must be less than number of reference points ({ref_positions.shape[0]})"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    neighbors_index = []
    for i in range(0, query_positions.shape[0], chunk_size):
        chunk_out_positions = query_positions[i : i + chunk_size]
        chunk_neighbors_index = _knn_search(ref_positions, chunk_out_positions, k)
        neighbors_index.append(chunk_neighbors_index)

    return torch.concatenate(neighbors_index, dim=0)


@torch.no_grad()
def neighbor_knn_search(
    ref_positions: Float[Tensor, "N 3"],
    query_positions: Float[Tensor, "M 3"],
    k: int,
    search_method: Literal["chunk"] = "chunk",
    chunk_size: int = 32768,  # 2^15
) -> Int[Tensor, "M K"]:
    r"""Find k-nearest neighbors for each query point.

    For each point in ``query_positions``, finds the k closest points in
    ``ref_positions``.

    Parameters
    ----------
    ref_positions : torch.Tensor
        Reference point positions of shape :math:`(N, 3)`.
    query_positions : torch.Tensor
        Query point positions of shape :math:`(M, 3)`.
    k : int
        Number of nearest neighbors to find. Must be in range ``(0, N)``.
    search_method : Literal["chunk"], optional, default="chunk"
        Search implementation. "chunk" uses memory-efficient chunked search.
    chunk_size : int, optional, default=32768
        Chunk size for memory-efficient search.

    Returns
    -------
    torch.Tensor
        Neighbor indices of shape :math:`(M, K)`.

    Raises
    ------
    ValueError
        If k is out of valid range.
    ValueError
        If an unsupported search method is specified.

    Examples
    --------
    >>> import torch
    >>> ref = torch.rand(1000, 3).cuda()
    >>> query = torch.rand(100, 3).cuda()
    >>> neighbors = neighbor_knn_search(ref, query, k=16)
    >>> neighbors.shape
    torch.Size([100, 16])
    """
    if not (0 < k < ref_positions.shape[0]):
        raise ValueError(f"k ({k}) must be in range (0, {ref_positions.shape[0]})")
    if search_method not in ["chunk"]:
        raise ValueError(f"search_method must be 'chunk', got '{search_method}'")

    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    if ref_positions.device != query_positions.device:
        raise ValueError(
            f"Device mismatch: ref_positions on {ref_positions.device}, "
            f"query_positions on {query_positions.device}"
        )

    if search_method == "chunk":
        if query_positions.shape[0] < chunk_size:
            neighbors_index = _knn_search(ref_positions, query_positions, k)
        else:
            neighbors_index = _chunked_knn_search(
                ref_positions, query_positions, k, chunk_size=chunk_size
            )
    else:
        raise ValueError(f"search_method {search_method} not supported.")

    return neighbors_index


@torch.no_grad()
def batched_neighbor_knn_search(
    ref_positions: Float[Tensor, "B N 3"],
    query_positions: Float[Tensor, "B M 3"],
    k: int,
    search_method: Literal["chunk"] = "chunk",
    chunk_size: int = 4096,
) -> Int[Tensor, "B M K"]:
    r"""Batched k-nearest neighbor search.

    Performs KNN search for each batch element independently, with indices
    offset to reference the flattened reference point array.

    Parameters
    ----------
    ref_positions : torch.Tensor
        Reference point positions of shape :math:`(B, N, 3)`.
    query_positions : torch.Tensor
        Query point positions of shape :math:`(B, M, 3)`.
    k : int
        Number of nearest neighbors to find.
    search_method : Literal["chunk"], optional, default="chunk"
        Search implementation.
    chunk_size : int, optional, default=4096
        Chunk size for memory-efficient search.

    Returns
    -------
    torch.Tensor
        Neighbor indices of shape :math:`(B, M, K)` with indices offset
        by ``b * N`` for batch element ``b``.

    Raises
    ------
    ValueError
        If batch sizes don't match.

    Examples
    --------
    >>> import torch
    >>> ref = torch.rand(4, 1000, 3).cuda()
    >>> query = torch.rand(4, 100, 3).cuda()
    >>> neighbors = batched_neighbor_knn_search(ref, query, k=16)
    >>> neighbors.shape
    torch.Size([4, 100, 16])

    Note
    ----
    The returned indices are offset so that they can be used to index
    a flattened ``(B * N, ...)`` tensor directly.
    """
    if ref_positions.shape[0] != query_positions.shape[0]:
        raise ValueError(
            f"Batch size mismatch, {ref_positions.shape[0]} != {query_positions.shape[0]}"
        )

    neighbors = []
    index_offset = 0

    for i in range(ref_positions.shape[0]):
        neighbor_index = neighbor_knn_search(
            ref_positions[i], query_positions[i], k, search_method, chunk_size
        )
        # Offset indices for flattened indexing
        neighbors.append(neighbor_index + index_offset)
        index_offset += ref_positions.shape[1]

    return torch.stack(neighbors, dim=0)
