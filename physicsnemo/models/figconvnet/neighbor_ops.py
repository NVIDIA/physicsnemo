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
    """
    Wrapper for the output of a neighbor search operation.
    """

    # N is the total number of neighbors for all M queries
    _neighbors_index: Int[Tensor, "N"]  # noqa: F821
    # M is the number of queries
    _neighbors_row_splits: Int[Tensor, "M + 1"]  # noqa: F821

    def __init__(self, *args):
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
    def neighbors_index(self):
        return self._neighbors_index

    @property
    def neighbors_row_splits(self):
        return self._neighbors_row_splits

    def to(self, device: Union[str, int, torch.device]):
        self._neighbors_index.to(device)
        self._neighbors_row_splits.to(device)
        return self


def neighbor_radius_search(
    inp_positions: Float[Tensor, "N 3"],
    out_positions: Float[Tensor, "M 3"],
    radius: float,
    search_method: Literal["warp"] = "warp",
) -> NeighborSearchReturn:
    """
    inp_positions: [N,3]
    out_positions: [M,3]
    radius: float
    search_method: Literal["warp", "open3d"]
    """
    # Critical for multi GPU
    if inp_positions.is_cuda:
        torch.cuda.set_device(inp_positions.device)
    assert inp_positions.device == out_positions.device
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
    """
    inp_positions: [B,N,3]
    out_positions: [B,M,3]
    radius: float
    search_method: Literal["warp", "open3d"]
    """
    assert (
        inp_positions.shape[0] == out_positions.shape[0]
    ), f"Batch size mismatch, {inp_positions.shape[0]} != {out_positions.shape[0]}"

    if search_method == "warp":
        neighbor_index, neighbor_dist, neighbor_offset = batched_radius_search_warp(
            inp_positions, out_positions, radius
        )
    else:
        raise ValueError(f"search_method {search_method} not supported.")

    return NeighborSearchReturn(neighbor_index, neighbor_offset)


@torch.no_grad()
def _knn_search(
    ref_positions: Int[Tensor, "N 3"],
    query_positions: Int[Tensor, "M 3"],
    k: int,
) -> Int[Tensor, "M K"]:
    """Perform knn search using the open3d backend."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert ref_positions.device == query_positions.device
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    # Use topk to get the top k indices from distances
    dists = torch.cdist(query_positions, ref_positions)
    _, neighbors_index = torch.topk(dists, k, dim=1, largest=False)
    return neighbors_index


@torch.no_grad()
def _chunked_knn_search(
    ref_positions: Int[Tensor, "N 3"],
    query_positions: Int[Tensor, "M 3"],
    k: int,
    chunk_size: int = 4096,
):
    """Divide the out_positions into chunks and perform knn search."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert chunk_size > 0
    neighbors_index = []
    for i in range(0, query_positions.shape[0], chunk_size):
        chunk_out_positions = query_positions[i : i + chunk_size]
        chunk_neighbors_index = _knn_search(ref_positions, chunk_out_positions, k)
        neighbors_index.append(chunk_neighbors_index)
    return torch.concatenate(neighbors_index, dim=0)


@torch.no_grad()
def neighbor_knn_search(
    ref_positions: Int[Tensor, "N 3"],
    query_positions: Int[Tensor, "M 3"],
    k: int,
    search_method: Literal["chunk"] = "chunk",
    chunk_size: int = 32768,  # 2^15
) -> Int[Tensor, "M K"]:
    """
    ref_positions: [N,3]
    query_positions: [M,3]
    k: int
    """
    assert 0 < k < ref_positions.shape[0]
    assert search_method in ["chunk"]
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    assert ref_positions.device == query_positions.device
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
    ref_positions: Int[Tensor, "B N 3"],
    query_positions: Int[Tensor, "B M 3"],
    k: int,
    search_method: Literal["chunk"] = "chunk",
    chunk_size: int = 4096,
) -> Int[Tensor, "B M K"]:
    """
    ref_positions: [B,N,3]
    query_positions: [B,M,3]
    k: int
    """
    assert (
        ref_positions.shape[0] == query_positions.shape[0]
    ), f"Batch size mismatch, {ref_positions.shape[0]} != {query_positions.shape[0]}"
    neighbors = []
    index_offset = 0
    for i in range(ref_positions.shape[0]):
        neighbor_index = neighbor_knn_search(
            ref_positions[i], query_positions[i], k, search_method, chunk_size
        )
        neighbors.append(neighbor_index + index_offset)
        index_offset += ref_positions.shape[1]
    return torch.stack(neighbors, dim=0)
