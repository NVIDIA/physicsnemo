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

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from physicsnemo.distributed import (
    DistributedManager,
    all_gather_v,
    gather_v,
    indexed_all_to_all_v,
    scatter_v,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphPartition:
    """
    Class acting as an "utility" structure to hold all relevant buffers and variables
    to define a graph partition and facilitate exchange of necessary buffers for
    message passing on a distributed graph.

    A global graph is assumed to be defined through a global CSC structure
    defining edges between source nodes and destination nodes which are assumed
    to be numbered indexed by contiguous IDs. Hence, features associated to both
    nodes and edges can be represented through dense feature tables globally.
    When partitioning graph and features, we distribute destination nodes and all
    their incoming edges on all ranks within the partition group based on a specified
    mapping. Based on this scheme, there will a be a difference between
    partitioned source nodes (partitioned features) and local source node
    IDs which refer to the node IDs within the local graph defined by the
    destination nodes on each rank. To allow message passing, communication
    primitives have to ensure to gather all corresponding features for all
    local source nodes based on the applied partitioning scheme. This also
    leads to the distinction of local source node IDs and remote source node
    IDs on each rank where the latter simply refers to the local row ID within
    the dense partitioning of node features and the former indicates the source
    of a message for each edge within each local graph.

    Parameters
    ----------
    partition_size : int
        size of partition
    partition_rank : int
        local rank of this partition w.r.t. group of partitions
    device : torch.device
        device handle for buffers within this partition rank
    """

    partition_size: int
    partition_rank: int
    device: torch.device
    # flag to indicate using adj matrix 1-D row-decomp
    matrix_decomp: bool = False

    # data structures defining partition
    # set in after initialization or during execution
    # of desired partition scheme

    # local CSR offsets defining local graph on each `partition_rank`
    local_offsets: Optional[torch.Tensor] = None
    # local CSR indices defining local graph on each `partition_rank`
    local_indices: Optional[torch.Tensor] = None
    # number of source nodes in local graph on each `partition_rank`
    num_local_src_nodes: int = -1
    # number of destination nodes in local graph on each `partition_rank`
    num_local_dst_nodes: int = -1
    # number of edges in local graph on each `partition_rank`
    num_local_indices: int = -1
    # mapping from global to local ID space (source node IDs)
    map_partitioned_src_ids_to_global: Optional[torch.Tensor] = None
    map_concatenated_local_src_ids_to_global: Optional[torch.Tensor] = None
    # mapping from local to global ID space (destination node IDs)
    map_partitioned_dst_ids_to_global: Optional[torch.Tensor] = None
    map_concatenated_local_dst_ids_to_global: Optional[torch.Tensor] = None
    # mapping from local to global ID space (edge IDs)
    map_partitioned_edge_ids_to_global: Optional[torch.Tensor] = None
    map_concatenated_local_edge_ids_to_global: Optional[torch.Tensor] = None
    # reverse mappings
    map_global_src_ids_to_concatenated_local: Optional[torch.Tensor] = None
    map_global_dst_ids_to_concatenated_local: Optional[torch.Tensor] = None
    map_global_edge_ids_to_concatenated_local: Optional[torch.Tensor] = None

    # utility lists and sizes required for exchange of messages
    # between graph partitions through distributed communication primitives

    # number of IDs each rank potentially sends to all other ranks
    sizes: Optional[List[List[int]]] = None
    # local indices of IDs current rank sends to all other ranks
    scatter_indices: Optional[List[torch.Tensor]] = None
    # number of global source nodes for each `partition_rank`
    num_src_nodes_in_each_partition: Optional[List[int]] = None
    # number of global destination nodes for each `partition_rank`
    num_dst_nodes_in_each_partition: Optional[List[int]] = None
    # number of global indices for each `partition_rank`
    num_indices_in_each_partition: Optional[List[int]] = None

    def __post_init__(self):
        # after partition_size has been set in __init__
        if self.partition_size <= 0:
            raise ValueError(f"Expected partition_size > 0, got {self.partition_size}")
        if not (0 <= self.partition_rank < self.partition_size):
            raise ValueError(
                f"Expected 0 <= partition_rank < {self.partition_size}, got {self.partiton_rank}"
            )

        if self.sizes is None:
            self.sizes = [
                [None for _ in range(self.partition_size)]
                for _ in range(self.partition_size)
            ]

        if self.scatter_indices is None:
            self.scatter_indices = [None] * self.partition_size
        if self.num_src_nodes_in_each_partition is None:
            self.num_src_nodes_in_each_partition = [None] * self.partition_size
        if self.num_dst_nodes_in_each_partition is None:
            self.num_dst_nodes_in_each_partition = [None] * self.partition_size
        if self.num_indices_in_each_partition is None:
            self.num_indices_in_each_partition = [None] * self.partition_size

    def to(self, *args, **kwargs):
        # move all tensors
        for attr in dir(self):
            attr_val = getattr(self, attr)
            if isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(*args, **kwargs))

        # handle scatter_indices separately
        self.scatter_indices = [idx.to(*args, **kwargs) for idx in self.scatter_indices]

        return self


def partition_graph_with_id_mapping(
    global_offsets: torch.Tensor,
    global_indices: torch.Tensor,
    mapping_src_ids_to_ranks: torch.Tensor,
    mapping_dst_ids_to_ranks: torch.Tensor,
    partition_size: int,
    partition_rank: int,
    device: torch.device,
) -> GraphPartition:
    """
    Utility function which partitions a global graph given as CSC structure.
    It partitions both the global ID spaces for source nodes and destination nodes
    based on the corresponding mappings as well as the graph structure and edge IDs.
    For more details on partitioning in general see `GraphPartition`.
    The function performs the partitioning based on a global graph in CPU
    memory for each rank independently. It could be rewritten to e.g. only
    do it one rank and exchange the partitions or to an algorithm that also
    assumes an already distributed global graph, however, we expect global
    graphs to fit in CPU memory. After the partitioning, we can get rid off
    the larger one in CPU memory, only keep the local graphs on each GPU, and
    avoid tedious gather/scatter routines for exchanging partitions in the process.
    Note: It is up to the user to ensure that the provided mapping is valid. In particular,
    we expect each rank to receive a non-empty partition of node IDs.

    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    mapping_src_ids_to_ranks: torch.Tensor
        maps each global ID from every source node to its partition rank
    mapping_dst_ids_to_ranks: torch.Tensor
        maps each global ID from every destination node to its partition rank
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    """

    # initialize graph partition
    graph_partition = GraphPartition(
        partition_size=partition_size, partition_rank=partition_rank, device=device
    )

    # --------------------------------------------------------------
    # initialize temporary variables used in computing the partition

    # global IDs of in each partition
    dst_nodes_in_each_partition = [None] * partition_size
    src_nodes_in_each_partition = [None] * partition_size
    num_dst_nodes_in_each_partition = [None] * partition_size
    num_src_nodes_in_each_partition = [None] * partition_size

    dtype = global_indices.dtype
    input_device = global_indices.device

    graph_partition.map_concatenated_local_src_ids_to_global = torch.empty_like(
        mapping_src_ids_to_ranks
    )
    graph_partition.map_concatenated_local_dst_ids_to_global = torch.empty_like(
        mapping_dst_ids_to_ranks
    )
    graph_partition.map_concatenated_local_edge_ids_to_global = torch.empty_like(
        global_indices
    )
    graph_partition.map_global_src_ids_to_concatenated_local = torch.empty_like(
        mapping_src_ids_to_ranks
    )
    graph_partition.map_global_dst_ids_to_concatenated_local = torch.empty_like(
        mapping_dst_ids_to_ranks
    )
    graph_partition.map_global_edge_ids_to_concatenated_local = torch.empty_like(
        global_indices
    )
    _map_global_src_ids_to_local = torch.empty_like(mapping_src_ids_to_ranks)

    # temporarily track cum-sum of nodes per partition for "concatenated_local_ids"
    _src_id_offset = 0
    _dst_id_offset = 0
    _edge_id_offset = 0

    for rank in range(partition_size):
        dst_nodes_in_each_partition[rank] = torch.nonzero(
            mapping_dst_ids_to_ranks == rank
        ).view(-1)
        src_nodes_in_each_partition[rank] = torch.nonzero(
            mapping_src_ids_to_ranks == rank
        ).view(-1)
        num_nodes = dst_nodes_in_each_partition[rank].numel()
        if num_nodes == 0:
            raise RuntimeError(
                f"Aborting partitioning, rank {rank} has 0 destination nodes to work on."
            )
        num_dst_nodes_in_each_partition[rank] = num_nodes

        num_nodes = src_nodes_in_each_partition[rank].numel()
        num_src_nodes_in_each_partition[rank] = num_nodes
        if num_nodes == 0:
            raise RuntimeError(
                f"Aborting partitioning, rank {rank} has 0 source nodes to work on."
            )

        # create mapping of global node IDs to/from "concatenated local" IDs
        ids = src_nodes_in_each_partition[rank]
        mapped_ids = torch.arange(
            start=_src_id_offset,
            end=_src_id_offset + ids.numel(),
            dtype=dtype,
            device=input_device,
        )
        _map_global_src_ids_to_local[ids] = mapped_ids - _src_id_offset
        graph_partition.map_global_src_ids_to_concatenated_local[ids] = mapped_ids
        graph_partition.map_concatenated_local_src_ids_to_global[mapped_ids] = ids
        _src_id_offset += ids.numel()

        ids = dst_nodes_in_each_partition[rank]
        mapped_ids = torch.arange(
            start=_dst_id_offset,
            end=_dst_id_offset + ids.numel(),
            dtype=dtype,
            device=input_device,
        )
        graph_partition.map_global_dst_ids_to_concatenated_local[ids] = mapped_ids
        graph_partition.map_concatenated_local_dst_ids_to_global[mapped_ids] = ids
        _dst_id_offset += ids.numel()

    graph_partition.num_src_nodes_in_each_partition = num_src_nodes_in_each_partition
    graph_partition.num_dst_nodes_in_each_partition = num_dst_nodes_in_each_partition

    # create local graph structures
    for rank in range(partition_size):
        offset_start = global_offsets[dst_nodes_in_each_partition[rank]].view(-1)
        offset_end = global_offsets[dst_nodes_in_each_partition[rank] + 1].view(-1)
        degree = offset_end - offset_start
        local_offsets = degree.view(-1).cumsum(dim=0)
        local_offsets = torch.cat(
            [
                torch.Tensor([0]).to(
                    dtype=dtype,
                    device=input_device,
                ),
                local_offsets,
            ]
        )

        partitioned_edge_ids = torch.cat(
            [
                torch.arange(
                    start=offset_start[i],
                    end=offset_end[i],
                    dtype=dtype,
                    device=input_device,
                )
                for i in range(len(offset_start))
            ]
        )

        ids = partitioned_edge_ids
        mapped_ids = torch.arange(
            _edge_id_offset,
            _edge_id_offset + ids.numel(),
            device=ids.device,
            dtype=ids.dtype,
        )
        graph_partition.map_global_edge_ids_to_concatenated_local[ids] = mapped_ids
        graph_partition.map_concatenated_local_edge_ids_to_global[mapped_ids] = ids
        _edge_id_offset += ids.numel()

        partitioned_src_ids = torch.cat(
            [
                global_indices[offset_start[i] : offset_end[i]].clone()
                for i in range(len(offset_start))
            ]
        )

        global_src_ids_on_rank, inverse_mapping = partitioned_src_ids.unique(
            sorted=True, return_inverse=True
        )
        remote_local_src_ids_on_rank = _map_global_src_ids_to_local[
            global_src_ids_on_rank
        ]

        _map_global_src_ids_to_local_graph = torch.zeros_like(mapping_src_ids_to_ranks)
        _num_local_indices = 0
        for rank_offset in range(partition_size):
            mask = mapping_src_ids_to_ranks[global_src_ids_on_rank] == rank_offset
            if partition_rank == rank_offset:
                # indices to send to this rank from this rank
                graph_partition.scatter_indices[rank] = (
                    remote_local_src_ids_on_rank[mask]
                    .detach()
                    .clone()
                    .to(dtype=torch.int64)
                )
            numel_mask = mask.sum().item()
            graph_partition.sizes[rank_offset][rank] = numel_mask

            tmp_ids = torch.arange(
                _num_local_indices,
                _num_local_indices + numel_mask,
                device=input_device,
                dtype=dtype,
            )
            _num_local_indices += numel_mask
            tmp_map = global_src_ids_on_rank[mask]
            _map_global_src_ids_to_local_graph[tmp_map] = tmp_ids

        local_indices = _map_global_src_ids_to_local_graph[partitioned_src_ids]
        graph_partition.num_indices_in_each_partition[rank] = local_indices.size(0)

        if rank == partition_rank:
            # local graph
            graph_partition.local_offsets = local_offsets
            graph_partition.local_indices = local_indices
            graph_partition.num_local_indices = graph_partition.local_indices.size(0)
            graph_partition.num_local_dst_nodes = num_dst_nodes_in_each_partition[rank]
            graph_partition.num_local_src_nodes = global_src_ids_on_rank.size(0)

            # partition-local mappings (local IDs to global)
            graph_partition.map_partitioned_src_ids_to_global = (
                src_nodes_in_each_partition[rank]
            )
            graph_partition.map_partitioned_dst_ids_to_global = (
                dst_nodes_in_each_partition[rank]
            )
            graph_partition.map_partitioned_edge_ids_to_global = partitioned_edge_ids

    for r in range(graph_partition.partition_size):
        err_msg = "error in graph partition: list containing sizes of exchanged indices does not match the tensor of indices to be exchanged"
        if (
            graph_partition.sizes[graph_partition.partition_rank][r]
            != graph_partition.scatter_indices[r].numel()
        ):
            raise AssertionError(err_msg)

    graph_partition = graph_partition.to(device=device)

    return graph_partition


def partition_graph_with_matrix_decomposition(
    global_offsets: torch.Tensor,
    global_indices: torch.Tensor,
    num_nodes: int,
    partition_book: torch.Tensor,
    partition_size: int,
    partition_rank: int,
    device: torch.device,
) -> GraphPartition:
    """
    Utility function which partitions a global graph given as CSC structure based on its adjacency
    matirx using 1-D row-wise decomposition. This approach ensures a 1D uniform distribution of nodes
    and their associated 1-hop incoming edges. By treating source and destination nodes equivalently
    during partitioning, this approach assumes the graph is not bipartite.
    This decomposition also ensures that the graph convolution (spMM) remains local by maintaining a copy of
    the local incoming edge features and the local node outputs from the graph convolution.
    The memory complexity of this approach is O[(N/P + E/P)*hid_dim*L], where N/E are the number of nodes/edges.
    The transformation from local node storage to local edge storage is achieved using nccl `alltoall`.

    Key differences from the existing graph partition scheme (partition_graph_with_id_mapping):
    (1) This function partitions the global node ID space uniformly, without distinguishing
    between source and destination nodes (i.e., matrix row ordering or column ordering). Both
    src/dst or row/col nodes are indexed consistently within the adjacency matrix.
    (2) Each local graph (sub-matrix) can be defined/constructed by just node/edge offsets from
    global graph.
    (3) The partitioning is performed on a global graph stored in CPU memory, and then each device
    (rank) constructs its local graph independently from the global csc matrix.

    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    num_nodes : int
        number of nodes in the global graph
    partition_book : torch.Tensor
        the boundaries of 1-D row-decomp of adj. matrix for all ranks
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    """

    # initialize graph partition
    graph_partition = GraphPartition(
        partition_size=partition_size, partition_rank=partition_rank, device=device
    )
    dtype = global_indices.dtype
    # --------------------------------------------------------------
    # First partition the global row ptrs (dst nodes) to local row ptrs
    num_edges = global_indices.size(0)
    node_offset = partition_book[partition_rank]
    num_local_nodes = (
        partition_book[partition_rank + 1] - partition_book[partition_rank]
    )
    edge_partition_offset = global_offsets[node_offset]
    if node_offset + num_local_nodes > num_nodes:
        raise ValueError("Invalid node offset and number of local nodes")

    local_offsets = global_offsets[node_offset : node_offset + num_local_nodes + 1].to(
        device=device, non_blocking=True
    )
    graph_partition.local_offsets = local_offsets - edge_partition_offset
    graph_partition.num_local_dst_nodes = num_local_nodes

    # Scan through all partitions and compress the source nodes (edges) for each partition
    # to fill the local send/recv buffers for all-to-all communications
    partition_book = partition_book.to(device=device)
    for to_partition in range(partition_size):
        local_indices = global_indices[
            global_offsets[partition_book[to_partition]] : global_offsets[
                partition_book[to_partition + 1]
            ]
        ].to(device=device, non_blocking=True)
        # compress the columns (src nodes or local_indices) for each partition and record mapping (inverse_indices)
        global_src_node_at_partition, inverse_indices = local_indices.unique(
            sorted=True, return_inverse=True
        )
        global_src_node_at_partition_rank = (
            torch.bucketize(global_src_node_at_partition, partition_book, right=True)
            - 1
        )
        src_node_indices = torch.nonzero(
            global_src_node_at_partition_rank == partition_rank, as_tuple=False
        ).squeeze(1)
        # fill local send buffer for alltoalls (scatter selected nodes to_partition rank)
        graph_partition.scatter_indices[to_partition] = (
            global_src_node_at_partition[src_node_indices] - node_offset
        )
        # fill the numbers of indices (edges), dst nodes and src nodes for each partition
        graph_partition.num_indices_in_each_partition[to_partition] = (
            local_indices.size(0)
        )
        graph_partition.num_dst_nodes_in_each_partition[to_partition] = (
            partition_book[to_partition + 1] - partition_book[to_partition]
        )
        graph_partition.num_src_nodes_in_each_partition[to_partition] = (
            global_src_node_at_partition.size(0)
        )

        if to_partition == partition_rank:
            graph_partition.local_indices = inverse_indices
            graph_partition.num_local_indices = graph_partition.local_indices.size(0)
            graph_partition.num_local_src_nodes = global_src_node_at_partition.size(0)
            # map from local (compressed) column indices [0, ..., num_local_src_nodes] to their global node IDs
            graph_partition.map_partitioned_src_ids_to_global = (
                global_src_node_at_partition
            )

        for from_partition in range(partition_size):
            # fill all recv buffer sizes for alltoalls
            graph_partition.sizes[from_partition][to_partition] = torch.count_nonzero(
                global_src_node_at_partition_rank == from_partition
            )

    # trivial mappings due to 1D row-wise decomposition
    graph_partition.map_partitioned_dst_ids_to_global = torch.arange(
        node_offset, node_offset + num_local_nodes, dtype=dtype, device=device
    )
    graph_partition.map_partitioned_edge_ids_to_global = torch.arange(
        edge_partition_offset,
        edge_partition_offset + graph_partition.num_local_indices,
        dtype=dtype,
        device=device,
    )
    # trivial mappings due to 1D row-wise decomposition, with mem. cost O(E, N) at each dev; need to optimize
    graph_partition.map_concatenated_local_src_ids_to_global = torch.arange(
        num_nodes, dtype=dtype, device=device
    )
    graph_partition.map_concatenated_local_edge_ids_to_global = torch.arange(
        num_edges, dtype=dtype, device=device
    )
    graph_partition.map_concatenated_local_dst_ids_to_global = (
        graph_partition.map_concatenated_local_src_ids_to_global
    )
    graph_partition.map_global_src_ids_to_concatenated_local = (
        graph_partition.map_concatenated_local_src_ids_to_global
    )
    graph_partition.map_global_dst_ids_to_concatenated_local = (
        graph_partition.map_concatenated_local_src_ids_to_global
    )
    graph_partition.map_global_edge_ids_to_concatenated_local = (
        graph_partition.map_concatenated_local_edge_ids_to_global
    )
    graph_partition.matrix_decomp = True

    for r in range(graph_partition.partition_size):
        err_msg = "error in graph partition: list containing sizes of exchanged indices does not match the tensor of indices to be exchanged"
        if (
            graph_partition.sizes[graph_partition.partition_rank][r]
            != graph_partition.scatter_indices[r].numel()
        ):
            raise AssertionError(err_msg)
    graph_partition = graph_partition.to(device=device)
    return graph_partition


def partition_graph_nodewise(
    global_offsets: torch.Tensor,
    global_indices: torch.Tensor,
    partition_size: int,
    partition_rank: int,
    device: torch.device,
    matrix_decomp: bool = False,
) -> GraphPartition:
    """
    Utility function which partitions a global graph given as CSC structure naively
    by splitting both the IDs of source and destination nodes into chunks of equal
    size. For more details on partitioning in general see `GraphPartition`.
    The function performs the partitioning based on a global graph in CPU
    memory for each rank independently. It could be rewritten to e.g. only
    do it one rank and exchange the partitions or to an algorithm that also
    assumes an already distributed global graph, however, we expect global
    graphs to fit in CPU memory. After the partitioning, we can get rid off
    the larger one in CPU memory, only keep the local graphs on each GPU, and
    avoid tedious gather/scatter routines for exchanging partitions in the process.

    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    matrix_decomp : bool
        flag to enable matrix decomposition for partitioning
    """
    num_global_src_nodes = global_indices.max().item() + 1
    num_global_dst_nodes = global_offsets.size(0) - 1
    num_dst_nodes_per_partition = (
        num_global_dst_nodes + partition_size - 1
    ) // partition_size

    if matrix_decomp:
        if num_global_src_nodes != num_global_dst_nodes:
            raise ValueError(
                "Must be square adj. matrix (num_src=num_dst) for matrix decomposition"
            )
        partition_book = torch.arange(
            0,
            num_global_dst_nodes,
            num_dst_nodes_per_partition,
            dtype=global_indices.dtype,
        )
        partition_book = torch.cat(
            [
                partition_book,
                torch.tensor([num_global_dst_nodes], dtype=global_indices.dtype),
            ]
        )
        return partition_graph_with_matrix_decomposition(
            global_offsets,
            global_indices,
            num_global_dst_nodes,
            partition_book,
            partition_size,
            partition_rank,
            device,
        )

    num_src_nodes_per_partition = (
        num_global_src_nodes + partition_size - 1
    ) // partition_size

    mapping_dst_ids_to_ranks = (
        torch.arange(
            num_global_dst_nodes,
            dtype=global_offsets.dtype,
            device=global_offsets.device,
        )
        // num_dst_nodes_per_partition
    )
    mapping_src_ids_to_ranks = (
        torch.arange(
            num_global_src_nodes,
            dtype=global_offsets.dtype,
            device=global_offsets.device,
        )
        // num_src_nodes_per_partition
    )

    return partition_graph_with_id_mapping(
        global_offsets,
        global_indices,
        mapping_src_ids_to_ranks,
        mapping_dst_ids_to_ranks,
        partition_size,
        partition_rank,
        device,
    )


def partition_graph_by_coordinate_bbox(
    global_offsets: torch.Tensor,
    global_indices: torch.Tensor,
    src_coordinates: torch.Tensor,
    dst_coordinates: torch.Tensor,
    coordinate_separators_min: List[List[Optional[float]]],
    coordinate_separators_max: List[List[Optional[float]]],
    partition_size: int,
    partition_rank: int,
    device: torch.device,
) -> GraphPartition:
    """
    Utility function which partitions a global graph given as CSC structure.
    It partitions both the global ID spaces for source nodes and destination nodes
    based on their corresponding coordinates. Each partition will manage points which
    fulfill the boxconstraints specified by the specified coordinate separators. For each
    rank one is expected to specify the minimum and maximum coordinate value for each dimension.
    A partition the will manage all points for which ``min_val <= coord[d] < max_val`` holds. If one
    of the constraints is passed as `None`, it is assumed to be non-binding and the partition is defined
    by the corresponding half-space. Each rank maintains both a partition of the global source and
    destination nodes resulting from this subspace division.
    The function performs the partitioning based on a global graph in CPU
    memory for each rank independently. It could be rewritten to e.g. only
    do it one rank and exchange the partitions or to an algorithm that also
    assumes an already distributed global graph, however, we expect global
    graphs to fit in CPU memory. After the partitioning, we can get rid off
    the larger one in CPU memory, only keep the local graphs on each GPU, and
    avoid tedious gather/scatter routines for exchanging partitions in the process.
    Note: It is up to the user to ensure that the provided partition is valid.
    In particular, we expect each rank to receive a non-empty partition of node IDs.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.gnn_layers import partition_graph_by_coordinate_bbox
    >>> # simple graph with a degree of 2 per node
    >>> num_src_nodes = 8
    >>> num_dst_nodes = 4
    >>> offsets = torch.arange(num_dst_nodes + 1, dtype=torch.int64) * 2
    >>> indices = torch.arange(num_src_nodes, dtype=torch.int64)
    >>> # example with 2D coordinates
    >>> # assuming partitioning a 2D problem into the 4 quadrants
    >>> partition_size = 4
    >>> partition_rank = 0
    >>> coordinate_separators_min = [[0, 0], [None, 0], [None, None], [0, None]]
    >>> coordinate_separators_max = [[None, None], [0, None], [0, 0], [None, 0]]
    >>> device = "cuda:0" if torch.cuda.is_available() else "cpu"
    >>> # dummy coordinates
    >>> src_coordinates = torch.FloatTensor(
    ...     [
    ...         [-1.0, 1.0],
    ...         [1.0, 1.0],
    ...         [-1.0, -1.0],
    ...         [1.0, -1.0],
    ...         [-2.0, 2.0],
    ...         [2.0, 2.0],
    ...         [-2.0, -2.0],
    ...         [2.0, -2.0],
    ...     ]
    ... )
    >>> dst_coordinates = torch.FloatTensor(
    ...     [
    ...         [-1.0, 1.0],
    ...         [1.0, 1.0],
    ...         [-1.0, -1.0],
    ...         [1.0, -1.0],
    ...     ]
    ... )
    >>> # call partitioning routine
    >>> pg = partition_graph_by_coordinate_bbox(
    ...     offsets,
    ...     indices,
    ...     src_coordinates,
    ...     dst_coordinates,
    ...     coordinate_separators_min,
    ...     coordinate_separators_max,
    ...     partition_size,
    ...     partition_rank,
    ...     device,
    ... )
    >>> pg.local_offsets  # doctest: +ELLIPSIS
    tensor([0, 2]...)
    >>> pg.local_indices  # doctest: +ELLIPSIS
    tensor([0, 1]...)
    >>> pg.sizes
    [[0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]]
    >>>
    >>> # example with lat-long coordinates
    >>> # dummy coordinates
    >>> src_lat = torch.FloatTensor([-75, -60, -45, -30, 30, 45, 60, 75]).view(-1, 1)
    >>> dst_lat = torch.FloatTensor([-60, -30, 30, 30]).view(-1, 1)
    >>> src_long = torch.FloatTensor([-135, -135, 135, 135, -45, -45, 45, 45]).view(-1, 1)
    >>> dst_long = torch.FloatTensor([-135, 135, -45, 45]).view(-1, 1)
    >>> src_coordinates = torch.cat([src_lat, src_long], dim=1)
    >>> dst_coordinates = torch.cat([dst_lat, dst_long], dim=1)
    >>> # separate sphere at equator and 0 degree longitude into 4 parts
    >>> coordinate_separators_min = [
    ...     [-90, -180],
    ...     [-90, 0],
    ...     [0, -180],
    ...     [0, 0],
    ... ]
    >>> coordinate_separators_max = [
    ...     [0, 0],
    ...     [0, 180],
    ...     [90, 0],
    ...     [90, 180],
    ... ]
    >>> # call partitioning routine
    >>> partition_size = 4
    >>> partition_rank = 0
    >>> device = "cuda:0" if torch.cuda.is_available() else "cpu"
    >>> pg = partition_graph_by_coordinate_bbox(
    ...     offsets,
    ...     indices,
    ...     src_coordinates,
    ...     dst_coordinates,
    ...     coordinate_separators_min,
    ...     coordinate_separators_max,
    ...     partition_size,
    ...     partition_rank,
    ...     device,
    ... )
    >>> pg.local_offsets  # doctest: +ELLIPSIS
    tensor([0, 2]...)
    >>> pg.local_indices  # doctest: +ELLIPSIS
    tensor([0, 1]...)
    >>> pg.sizes
    [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]

    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    src_coordinates : torch.Tensor
        coordinates of each source node
    dst_coordinates : torch.Tensor
        coordinates of each destination node
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    """

    dim = src_coordinates.size(-1)
    if dst_coordinates.size(-1) != dim:
        raise ValueError()
    if len(coordinate_separators_min) != partition_size:
        a, b = len(coordinate_separators_min), partition_size
        error_msg = "Expected len(coordinate_separators_min) == partition_size"
        error_msg += f", but got {a} and {b} respectively"
        raise ValueError(error_msg)
    if len(coordinate_separators_max) != partition_size:
        a, b = len(coordinate_separators_max), partition_size
        error_msg = "Expected len(coordinate_separators_max) == partition_size"
        error_msg += f", but got {a} and {b} respectively"
        raise ValueError(error_msg)

    for rank in range(partition_size):
        if len(coordinate_separators_min[rank]) != dim:
            a, b = len(coordinate_separators_min[rank]), dim
            error_msg = f"Expected len(coordinate_separators_min[{rank}]) == dim"
            error_msg += f", but got {a} and {b} respectively"
        if len(coordinate_separators_max[rank]) != dim:
            a, b = len(coordinate_separators_max[rank]), dim
            error_msg = f"Expected len(coordinate_separators_max[{rank}]) == dim"
            error_msg += f", but got {a} and {b} respectively"

    num_global_src_nodes = global_indices.max().item() + 1
    num_global_dst_nodes = global_offsets.size(0) - 1

    mapping_dst_ids_to_ranks = torch.zeros(
        num_global_dst_nodes, dtype=global_offsets.dtype, device=global_offsets.device
    )
    mapping_src_ids_to_ranks = torch.zeros(
        num_global_src_nodes,
        dtype=global_offsets.dtype,
        device=global_offsets.device,
    )

    def _assign_ranks(mapping, coordinates):
        for p in range(partition_size):
            mask = torch.ones_like(mapping).to(dtype=torch.bool)
            for d in range(dim):
                min_val, max_val = (
                    coordinate_separators_min[p][d],
                    coordinate_separators_max[p][d],
                )
                if min_val is not None:
                    mask = mask & (coordinates[:, d] >= min_val)
                if max_val is not None:
                    mask = mask & (coordinates[:, d] < max_val)
            mapping[mask] = p

    _assign_ranks(mapping_src_ids_to_ranks, src_coordinates)
    _assign_ranks(mapping_dst_ids_to_ranks, dst_coordinates)

    return partition_graph_with_id_mapping(
        global_offsets,
        global_indices,
        mapping_src_ids_to_ranks,
        mapping_dst_ids_to_ranks,
        partition_size,
        partition_rank,
        device,
    )


class DistributedGraph:
    def __init__(
        self,
        global_offsets: torch.Tensor,
        global_indices: torch.Tensor,
        partition_size: int,
        graph_partition_group_name: str = None,
        graph_partition: Optional[GraphPartition] = None,
    ):  # pragma: no cover
        """
        Utility Class representing a distributed graph based on a given
        partitioning of a CSC graph structure. By default, a naive node-wise
        partitioning scheme is applied, see ``partition_graph_nodewise`` for
        details on that. This class then wraps necessary communication primitives
        to access all relevant feature buffers related to the graph.

        Parameters
        ----------
        global_offsets : torch.Tensor
            CSC offsets, can live on the CPU
        global_indices : torch.Tensor
            CSC indices, can live on the CPU
        partition_size : int
            Number of process groups across which graphs are distributed, expected to
            be larger than 1, i.e. an actual partition distributed among multiple ranks.
        partition_group_name : str, default=None
            Name of process group across which graphs are distributed. Passing no process
            group name leads to a parallelism across the default process group.
            Otherwise, the group size of a process group is expected to match partition_size.
        graph_partition : GraphPartition, optional
            Optional graph_partition, if passed as None, the naive
            node-wise partitioning scheme will be applied to global_offsets and global_indices,
            otherwise, these will be ignored and the passed partition will be used instead.
        """

        dist_manager = DistributedManager()
        self.device = dist_manager.device
        self.partition_rank = dist_manager.group_rank(name=graph_partition_group_name)
        self.partition_size = dist_manager.group_size(name=graph_partition_group_name)
        error_msg = f"Passed partition_size does not correspond to size of process_group, got {partition_size} and {self.partition_size} respectively."
        if self.partition_size != partition_size:
            raise AssertionError(error_msg)
        self.process_group = dist_manager.group(name=graph_partition_group_name)

        if graph_partition is None:
            # default partitioning scheme
            self.graph_partition = partition_graph_nodewise(
                global_offsets,
                global_indices,
                self.partition_size,
                self.partition_rank,
                self.device,
            )

        else:
            error_msg = f"Passed graph_partition.partition_size does not correspond to size of process_group, got {graph_partition.partition_size} and {self.partition_size} respectively."
            if graph_partition.partition_size != self.partition_size:
                raise AssertionError(error_msg)
            error_msg = f"Passed graph_partition.device does not correspond to device of this rank, got {graph_partition.device} and {self.device} respectively."
            if graph_partition.device != self.device:
                raise AssertionError(error_msg)
            self.graph_partition = graph_partition

        send_sizes = self.graph_partition.sizes[self.graph_partition.partition_rank]
        recv_sizes = [
            p[self.graph_partition.partition_rank] for p in self.graph_partition.sizes
        ]
        msg = f"GraphPartition(rank={self.graph_partition.partition_rank}, "
        msg += f"num_local_src_nodes={self.graph_partition.num_local_src_nodes}, "
        msg += f"num_local_dst_nodes={self.graph_partition.num_local_dst_nodes}, "
        msg += f"num_partitioned_src_nodes={self.graph_partition.num_src_nodes_in_each_partition[self.graph_partition.partition_rank]}, "
        msg += f"num_partitioned_dst_nodes={self.graph_partition.num_dst_nodes_in_each_partition[self.graph_partition.partition_rank]}, "
        msg += f"send_sizes={send_sizes}, recv_sizes={recv_sizes})"
        print(msg)

        dist.barrier(self.process_group)

    def get_src_node_features_in_partition(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks

        if self.graph_partition.matrix_decomp:
            logger.warning(
                "Matrix decomposition assumes one type of node feature partition, and the graph"
                "adjacency matrix is square with identical src/dst node domains. "
                "So, only `get_dst_node_features_in_partition` is used/needed to get src or dst"
                "node features within a partition."
            )
            return self.get_dst_node_features_in_partition(
                global_node_features,
                scatter_features=scatter_features,
                src_rank=src_rank,
            )
        if scatter_features:
            global_node_features = global_node_features[
                self.graph_partition.map_concatenated_local_src_ids_to_global
            ]
            return scatter_v(
                global_node_features,
                self.graph_partition.num_src_nodes_in_each_partition,
                dim=0,
                src=src_rank,
                group=self.process_group,
            )

        return global_node_features.to(device=self.device)[
            self.graph_partition.map_partitioned_src_ids_to_global, :
        ]

    def get_src_node_features_in_local_graph(
        self, partitioned_src_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        # main primitive to gather all necessary src features
        # which are required for a csc-based message passing step
        return indexed_all_to_all_v(
            partitioned_src_node_features,
            indices=self.graph_partition.scatter_indices,
            sizes=self.graph_partition.sizes,
            use_fp32=True,
            dim=0,
            group=self.process_group,
        )

    def get_dst_node_features_in_partition(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            global_node_features = global_node_features.to(device=self.device)[
                self.graph_partition.map_concatenated_local_dst_ids_to_global
            ]
            return scatter_v(
                global_node_features,
                self.graph_partition.num_dst_nodes_in_each_partition,
                dim=0,
                src=src_rank,
                group=self.process_group,
            )

        return global_node_features.to(device=self.device)[
            self.graph_partition.map_partitioned_dst_ids_to_global, :
        ]

    def get_dst_node_features_in_local_graph(
        self,
        partitioned_dst_node_features: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        return partitioned_dst_node_features

    def get_edge_features_in_partition(
        self,
        global_edge_features: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            global_edge_features = global_edge_features[
                self.graph_partition.map_concatenated_local_edge_ids_to_global
            ]
            return scatter_v(
                global_edge_features,
                self.graph_partition.num_indices_in_each_partition,
                dim=0,
                src=src_rank,
                group=self.process_group,
            )

        return global_edge_features.to(device=self.device)[
            self.graph_partition.map_partitioned_edge_ids_to_global, :
        ]

    def get_edge_features_in_local_graph(
        self, partitioned_edge_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        return partitioned_edge_features

    def get_global_src_node_features(
        self,
        partitioned_node_features: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        error_msg = f"Passed partitioned_node_features.device does not correspond to device of this rank, got {partitioned_node_features.device} and {self.device} respectively."
        if partitioned_node_features.device != self.device:
            raise AssertionError(error_msg)

        if self.graph_partition.matrix_decomp:
            logger.warning(
                "Matrix decomposition assumes one type of node feature partition, and the graph"
                "adjacency matrix is square with identical src/dst node domains. "
                "So, only `get_global_dst_node_features` is used/needed to get global src or dst"
                "node features."
            )
            return self.get_global_dst_node_features(
                partitioned_node_features,
                get_on_all_ranks=get_on_all_ranks,
                dst_rank=dst_rank,
            )

        if not get_on_all_ranks:
            global_node_feat = gather_v(
                partitioned_node_features,
                self.graph_partition.num_src_nodes_in_each_partition,
                dim=0,
                dst=dst_rank,
                group=self.process_group,
            )
            if self.graph_partition.partition_rank == dst_rank:
                global_node_feat = global_node_feat[
                    self.graph_partition.map_global_src_ids_to_concatenated_local
                ]

            return global_node_feat

        global_node_feat = all_gather_v(
            partitioned_node_features,
            self.graph_partition.num_src_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )
        global_node_feat = global_node_feat[
            self.graph_partition.map_global_src_ids_to_concatenated_local
        ]
        return global_node_feat

    def get_global_dst_node_features(
        self,
        partitioned_node_features: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        error_msg = f"Passed partitioned_node_features.device does not correspond to device of this rank, got {partitioned_node_features.device} and {self.device} respectively."
        if partitioned_node_features.device != self.device:
            raise AssertionError(error_msg)

        if not get_on_all_ranks:
            global_node_feat = gather_v(
                partitioned_node_features,
                self.graph_partition.num_dst_nodes_in_each_partition,
                dim=0,
                dst=dst_rank,
                group=self.process_group,
            )
            if self.graph_partition.partition_rank == dst_rank:
                global_node_feat = global_node_feat[
                    self.graph_partition.map_global_dst_ids_to_concatenated_local
                ]

            return global_node_feat

        global_node_feat = all_gather_v(
            partitioned_node_features,
            self.graph_partition.num_dst_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )
        global_node_feat = global_node_feat[
            self.graph_partition.map_global_dst_ids_to_concatenated_local
        ]
        return global_node_feat

    def get_global_edge_features(
        self,
        partitioned_edge_features: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        error_msg = f"Passed partitioned_edge_features.device does not correspond to device of this rank, got {partitioned_edge_features.device} and {self.device} respectively."
        if partitioned_edge_features.device != self.device:
            raise AssertionError(error_msg)

        if not get_on_all_ranks:
            global_edge_feat = gather_v(
                partitioned_edge_features,
                self.graph_partition.num_indices_in_each_partition,
                dim=0,
                dst=dst_rank,
                group=self.process_group,
            )
            if self.graph_partition.partition_rank == dst_rank:
                global_edge_feat = global_edge_feat[
                    self.graph_partition.map_global_edge_ids_to_concatenated_local
                ]
            return global_edge_feat

        global_edge_feat = all_gather_v(
            partitioned_edge_features,
            self.graph_partition.num_indices_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )
        global_edge_feat = global_edge_feat[
            self.graph_partition.map_global_edge_ids_to_concatenated_local
        ]
        return global_edge_feat
