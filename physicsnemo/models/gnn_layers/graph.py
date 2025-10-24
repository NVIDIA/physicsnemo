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

import warnings
from types import NoneType
from typing import Any, List, Optional, TypeAlias

import torch
from torch import Tensor

try:
    import dgl
    from dgl import DGLGraph
except ImportError:
    warnings.warn(
        "CuGraphCSC requires the DGL library. DGL library will soon be deprecated.",
        DeprecationWarning,
    )

    DGLGraph: TypeAlias = NoneType

try:
    from typing import Self
except ImportError:
    # for Python versions < 3.11
    from typing_extensions import Self

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.gnn_layers import (
    DistributedGraph,
    GraphPartition,
    partition_graph_by_coordinate_bbox,
)

try:
    from pylibcugraphops.pytorch import BipartiteCSC, StaticCSC

    USE_CUGRAPHOPS = True

except ImportError:
    StaticCSC = None
    BipartiteCSC = None
    USE_CUGRAPHOPS = False


class CuGraphCSC:
    """Constructs a CuGraphCSC object which is a generic graph object wrapping
    typical fields of the CSC representation. It is intended for easy handling
    of the dedicated graph structures required to call into the optimized cugraph-ops
    routines and is a convenience wrapper around a partioned graph in a distributed
    setting. In the latter case, a conversion to DGL compatible structures is possible.

    Parameters
    ----------
    offsets : Tensor
        The offsets tensor.
    indices : Tensor
        The indices tensor.
    num_src_nodes : int
        The number of source nodes.
    num_dst_nodes : int
        The number of destination nodes.
    ef_indices : Optional[Tensor], optional
        The edge feature indices tensor, by default None.
        These can be used if you want to keep edge-input originally
        indexed over COO-indices instead of permuting it such that they
        can be indexed by CSC-indices.
    reverse_graph_bwd : bool, optional
        Whether to reverse the graph for the backward pass, by default True
    cache_graph : bool, optional
        Whether to cache graph structures when wrapping offsets and indices
        to the corresponding cugraph-ops graph types. If graph change in each
        iteration, set to False, by default True.
    partition_size : int, default=1
        Number of process groups across which graph is distributed. If equal to 1,
        the model is run in a normal Single-GPU congiguration. For details on how
        the graph is partitioned, see ``DistributedGraph``.
    partition_group_name : str, default=None
        Name of process group across which graph is distributed. If partition_size
        is set to 1, the model is run in a normal Single-GPU configuration and the
        specification of a process group is not necessary. If partitition_size > 1,
        passing no process group name leads to a parallelism across the default
        process group. Otherwise, the group size of a process group is expected
        to match partition_size.
    """

    def __init__(
        self,
        offsets: Tensor,
        indices: Tensor,
        num_src_nodes: int,
        num_dst_nodes: int,
        ef_indices: Optional[Tensor] = None,
        reverse_graph_bwd: bool = True,
        cache_graph: bool = True,
        partition_size: Optional[int] = -1,
        partition_group_name: Optional[str] = None,
        graph_partition: Optional[GraphPartition] = None,
    ) -> None:
        self.offsets = offsets
        self.indices = indices
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.ef_indices = ef_indices
        self.reverse_graph_bwd = reverse_graph_bwd
        self.cache_graph = cache_graph

        # cugraph-ops structures
        self.bipartite_csc = None
        self.static_csc = None
        # dgl graph
        self.dgl_graph = None

        self.is_distributed = False
        self.dist_csc = None

        if partition_size <= 1:
            self.is_distributed = False
            return

        if self.ef_indices is not None:
            raise AssertionError(
                "DistributedGraph does not support mapping CSC-indices to COO-indices."
            )

        self.dist_graph = DistributedGraph(
            self.offsets,
            self.indices,
            partition_size,
            partition_group_name,
            graph_partition=graph_partition,
        )

        # overwrite graph information with local graph after distribution
        self.offsets = self.dist_graph.graph_partition.local_offsets
        self.indices = self.dist_graph.graph_partition.local_indices
        self.num_src_nodes = self.dist_graph.graph_partition.num_local_src_nodes
        self.num_dst_nodes = self.dist_graph.graph_partition.num_local_dst_nodes
        self.is_distributed = True

    @staticmethod
    def from_dgl(
        graph: DGLGraph,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        partition_by_bbox: bool = False,
        src_coordinates: Optional[torch.Tensor] = None,
        dst_coordinates: Optional[torch.Tensor] = None,
        coordinate_separators_min: Optional[List[List[Optional[float]]]] = None,
        coordinate_separators_max: Optional[List[List[Optional[float]]]] = None,
    ):  # pragma: no cover
        # DGL changed their APIs w.r.t. how sparse formats can be accessed
        # this here is done to support both versions
        if hasattr(graph, "adj_tensors"):
            offsets, indices, edge_perm = graph.adj_tensors("csc")
        elif hasattr(graph, "adj_sparse"):
            offsets, indices, edge_perm = graph.adj_sparse("csc")
        else:
            raise ValueError("Passed graph object doesn't support conversion to CSC.")

        n_src_nodes, n_dst_nodes = (graph.num_src_nodes(), graph.num_dst_nodes())

        graph_partition = None

        if partition_by_bbox and partition_size > 1:
            dist_manager = DistributedManager()
            partition_rank = dist_manager.group_rank(name=partition_group_name)

            graph_partition = partition_graph_by_coordinate_bbox(
                offsets.to(dtype=torch.int64),
                indices.to(dtype=torch.int64),
                src_coordinates=src_coordinates,
                dst_coordinates=dst_coordinates,
                coordinate_separators_min=coordinate_separators_min,
                coordinate_separators_max=coordinate_separators_max,
                partition_size=partition_size,
                partition_rank=partition_rank,
                device=dist_manager.device,
            )

        graph_csc = CuGraphCSC(
            offsets.to(dtype=torch.int64),
            indices.to(dtype=torch.int64),
            n_src_nodes,
            n_dst_nodes,
            partition_size=partition_size,
            partition_group_name=partition_group_name,
            graph_partition=graph_partition,
        )

        return graph_csc, edge_perm

    def get_src_node_features_in_partition(
        self,
        global_src_feat: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """
        Get local chunk of global source node features for each rank corresponding
        to its rank in the process group across which the graph is partitioned.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_src_node_features_in_partition(
                global_src_feat, scatter_features=scatter_features, src_rank=src_rank
            )
        return global_src_feat

    def get_src_node_features_in_local_graph(
        self, local_src_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Get all source node features on all ranks from all other ranks which are requires
        for the neighborhood definition in the local graph. ``local_src_feat`` here
        corresponds to the local chunk of the global source node features on each rank
        corresponding to its rank in the process group across which the graph is partitioned.
        After this primitive, any message passing routine should have all necessary tensors
        to work on the corresponding local graph according to the partition rank.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_src_node_features_in_local_graph(local_src_feat)
        return local_src_feat

    def get_dst_node_features_in_partition(
        self,
        global_dst_feat: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """
        Get local chunk of global destination node features for each rank corresponding
        to its rank in the process group across which the graph is partitioned.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_dst_node_features_in_partition(
                global_dst_feat, scatter_features=scatter_features, src_rank=src_rank
            )
        return global_dst_feat

    def get_edge_features_in_partition(
        self,
        global_efeat: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """
        Get local chunk of global edge features for each rank corresponding
        to its rank in the process group across which the graph is partitioned.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_edge_features_in_partition(
                global_efeat, scatter_features=scatter_features, src_rank=src_rank
            )
        return global_efeat

    def get_global_src_node_features(
        self,
        local_nfeat: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:
        """
        Based on local source node features on each rank corresponding
        to its rank in the process group across which the graph is partitioned,
        get the global node features either on all group ranks or on group rank 0.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_global_src_node_features(
                local_nfeat,
                get_on_all_ranks,
                dst_rank=dst_rank,
            )
        return local_nfeat

    def get_global_dst_node_features(
        self,
        local_nfeat: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:
        """
        Based on local destination node features on each rank corresponding
        to its rank in the process group across which the graph is partitioned,
        get the global node features either on all group ranks or on group rank 0.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_global_dst_node_features(
                local_nfeat,
                get_on_all_ranks,
                dst_rank=dst_rank,
            )
        return local_nfeat

    def get_global_edge_features(
        self,
        local_efeat: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:
        """
        Based on local edge features on each rank corresponding
        to its rank in the process group across which the graph is partitioned,
        get the global edge features either on all group ranks or on group rank 0.
        """
        if self.is_distributed:  # pragma: no cover
            return self.dist_graph.get_global_edge_features(
                local_efeat,
                get_on_all_ranks,
                dst_rank=dst_rank,
            )
        return local_efeat

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Moves the object to the specified device, dtype, or format and returns the
        updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        NodeBlockCUGO
            The updated object after moving to the specified device, dtype, or format.
        """
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype not in (
            None,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                f"Invalid dtype, expected torch.int32 or torch.int64, got {dtype}."
            )
        self.offsets = self.offsets.to(device=device, dtype=dtype)
        self.indices = self.indices.to(device=device, dtype=dtype)
        if self.ef_indices is not None:
            self.ef_indices = self.ef_indices.to(device=device, dtype=dtype)

        return self

    def to_bipartite_csc(self, dtype=None) -> BipartiteCSC:
        """Converts the graph to a bipartite CSC graph.

        Parameters
        ----------
        dtype : torch.dtype, optional
            The dtype of the graph, by default None

        Returns
        -------
        BipartiteCSC
            The bipartite CSC graph.
        """

        if not (USE_CUGRAPHOPS):
            raise RuntimeError(
                "Conversion failed, expected cugraph-ops to be installed."
            )
        if not self.offsets.is_cuda:
            raise RuntimeError("Expected the graph structures to reside on GPU.")

        if self.bipartite_csc is None or not self.cache_graph:
            # Occassionally, we have to watch out for the IdxT type
            # of offsets and indices. Technically, they are only relevant
            # for storing node and edge indices. However, they are also used
            # to index pointers in the underlying kernels (for now). This means
            # that depending on the data dimension, one has to rely on int64
            # for the indices despite int32 technically being enough to store the
            # graph. This will be improved in cugraph-ops-23.06. Until then, allow
            # the change of dtype.
            graph_offsets = self.offsets
            graph_indices = self.indices
            graph_ef_indices = self.ef_indices

            if dtype is not None:
                graph_offsets = self.offsets.to(dtype=dtype)
                graph_indices = self.indices.to(dtype=dtype)
                if self.ef_indices is not None:
                    graph_ef_indices = self.ef_indices.to(dtype=dtype)

            graph = BipartiteCSC(
                graph_offsets,
                graph_indices,
                self.num_src_nodes,
                graph_ef_indices,
                reverse_graph_bwd=self.reverse_graph_bwd,
            )
            self.bipartite_csc = graph

        return self.bipartite_csc

    def to_static_csc(self, dtype=None) -> StaticCSC:
        """Converts the graph to a static CSC graph.

        Parameters
        ----------
        dtype : torch.dtype, optional
            The dtype of the graph, by default None

        Returns
        -------
        StaticCSC
            The static CSC graph.
        """

        if not (USE_CUGRAPHOPS):
            raise RuntimeError(
                "Conversion failed, expected cugraph-ops to be installed."
            )
        if not self.offsets.is_cuda:
            raise RuntimeError("Expected the graph structures to reside on GPU.")

        if self.static_csc is None or not self.cache_graph:
            # Occassionally, we have to watch out for the IdxT type
            # of offsets and indices. Technically, they are only relevant
            # for storing node and edge indices. However, they are also used
            # to index pointers in the underlying kernels (for now). This means
            # that depending on the data dimension, one has to rely on int64
            # for the indices despite int32 technically being enough to store the
            # graph. This will be improved in cugraph-ops-23.06. Until then, allow
            # the change of dtype.
            graph_offsets = self.offsets
            graph_indices = self.indices
            graph_ef_indices = self.ef_indices

            if dtype is not None:
                graph_offsets = self.offsets.to(dtype=dtype)
                graph_indices = self.indices.to(dtype=dtype)
                if self.ef_indices is not None:
                    graph_ef_indices = self.ef_indices.to(dtype=dtype)

            graph = StaticCSC(
                graph_offsets,
                graph_indices,
                graph_ef_indices,
            )
            self.static_csc = graph

        return self.static_csc

    def to_dgl_graph(self) -> DGLGraph:  # pragma: no cover
        """Converts the graph to a DGLGraph.
        This can be useful if e.g. one wants to operate on a distributed
        graph in PhysicsNeMo which assumes a simple CSC structure, but
        has only implemented GNN primitives in a DGL backend.

        Returns
        -------
        DGLGraph
            The DGLGraph created from the given object in CSC format.
        """

        if self.dgl_graph is None or not self.cache_graph:
            if self.ef_indices is not None:
                raise AssertionError("ef_indices is not supported.")
            graph_offsets = self.offsets
            dst_degree = graph_offsets[1:] - graph_offsets[:-1]
            src_indices = self.indices
            dst_indices = torch.arange(
                0,
                graph_offsets.size(0) - 1,
                dtype=graph_offsets.dtype,
                device=graph_offsets.device,
            )
            dst_indices = torch.repeat_interleave(dst_indices, dst_degree, dim=0)

            # labels not important here
            self.dgl_graph = dgl.heterograph(
                {("src", "src2dst", "dst"): ("coo", (src_indices, dst_indices))},
                idtype=torch.int32,
            )

        return self.dgl_graph
