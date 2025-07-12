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
from typing import Any, Callable, Dict, Tuple, TypeAlias, Union

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data as PyGData

try:
    import dgl  # noqa: F401 for docs
    import dgl.function as fn
    from dgl import DGLGraph
except ImportError:
    warnings.warn(
        "Note: This only applies if you're using DGL.\n"
        "MeshGraphNet (DGL version) requires the DGL library.\n"
        "Install it with your preferred CUDA version from:\n"
        "https://www.dgl.ai/pages/start.html\n"
    )

    DGLGraph: TypeAlias = NoneType

try:
    import torch_scatter
except ImportError:
    warnings.warn(
        "MeshGraphNet will soon require PyTorch Geometric and torch_scatter.\n"
        "Install it from here:\n"
        "https://github.com/rusty1s/pytorch_scatter\n"
    )

from physicsnemo.models.gnn_layers import CuGraphCSC

GraphType: TypeAlias = PyGData | DGLGraph | CuGraphCSC


try:
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_bipartite_e2e,
        update_efeat_static_e2e,
    )

    USE_CUGRAPHOPS = True

except ImportError:
    update_efeat_bipartite_e2e = None
    update_efeat_static_e2e = None
    agg_concat_e2n = None
    USE_CUGRAPHOPS = False


def checkpoint_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Applies the identity function for checkpointing.

    This function serves as an identity function for use with model layers
    when checkpointing is not enabled. It simply forwards the input arguments
    to the specified layer and returns its output.

    Parameters
    ----------
    layer : Callable
        The model layer or function to apply to the input arguments.
    *args
        Positional arguments to be passed to the layer.
    **kwargs
        Keyword arguments to be passed to the layer.

    Returns
    -------
    Any
        The output of the specified layer after processing the input arguments.
    """
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool) -> Callable:
    """Sets checkpoint function.

    This function returns the appropriate checkpoint function based on the
    provided `do_checkpointing` flag. If `do_checkpointing` is True, the
    function returns the checkpoint function from PyTorch's
    `torch.utils.checkpoint`. Otherwise, it returns an identity function
    that simply passes the inputs through the given layer.

    Parameters
    ----------
    do_checkpointing : bool
        Whether to use checkpointing for gradient computation. Checkpointing
        can reduce memory usage during backpropagation at the cost of
        increased computation time.

    Returns
    -------
    Callable
        The selected checkpoint function to use for gradient computation.
    """
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


def concat_message_function(edges: Tensor) -> Dict[str, Tensor]:
    """Concatenates source node, destination node, and edge features.

    Parameters
    ----------
    edges : Tensor
        Edges.

    Returns
    -------
    Dict[Tensor]
        Concatenated source node, destination node, and edge features.
    """
    # concats src node , dst node, and edge features
    cat_feat = torch.cat((edges.data["x"], edges.src["x"], edges.dst["x"]), dim=1)
    return {"cat_feat": cat_feat}


@torch.jit.ignore()
def concat_efeat_dgl(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[torch.Tensor, torch.Tensor]],
    graph: DGLGraph,
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor, Tensor]
        Node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tuple):
        src_feat, dst_feat = nfeat
        with graph.local_scope():
            graph.srcdata["x"] = src_feat
            graph.dstdata["x"] = dst_feat
            graph.edata["x"] = efeat
            graph.apply_edges(concat_message_function)
            return graph.edata["cat_feat"]

    with graph.local_scope():
        graph.ndata["x"] = nfeat
        graph.edata["x"] = efeat
        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


def concat_efeat_pyg(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor, Tensor]],
    graph: PyGData,
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for PyG graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features.
    graph : PyGData
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    src_feat, dst_feat = nfeat if isinstance(nfeat, Tuple) else (nfeat, nfeat)
    src_idx, dst_idx = graph.edge_index.long()
    cat_feat = torch.cat((efeat, src_feat[src_idx], dst_feat[dst_idx]), dim=1)
    return cat_feat


def concat_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: GraphType,
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features.
    graph : GraphType
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tensor):
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                src_feat, dst_feat = nfeat, nfeat
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                efeat = concat_efeat_dgl(
                    efeat, (src_feat, dst_feat), graph.to_dgl_graph()
                )

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                    # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
                    bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                    dst_feat = nfeat
                    efeat = update_efeat_bipartite_e2e(
                        efeat, src_feat, dst_feat, bipartite_graph, "concat"
                    )

                else:
                    static_graph = graph.to_static_csc()
                    efeat = update_efeat_static_e2e(
                        efeat,
                        nfeat,
                        static_graph,
                        mode="concat",
                        use_source_emb=True,
                        use_target_emb=True,
                    )
        elif isinstance(graph, DGLGraph):
            efeat = concat_efeat_dgl(efeat, nfeat, graph)
        elif isinstance(graph, PyGData):
            efeat = concat_efeat_pyg(efeat, nfeat, graph)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    elif isinstance(nfeat, Tuple):
        src_feat, dst_feat = nfeat
        # update edge features through concatenating edge and node features
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                efeat = concat_efeat_dgl(
                    efeat, (src_feat, dst_feat), graph.to_dgl_graph()
                )

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)
                # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
                bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
                efeat = update_efeat_bipartite_e2e(
                    efeat, src_feat, dst_feat, bipartite_graph, "concat"
                )
        elif isinstance(graph, DGLGraph):
            efeat = concat_efeat_dgl(efeat, (src_feat, dst_feat), graph)
        elif isinstance(graph, PyGData):
            efeat = concat_efeat_pyg(efeat, (src_feat, dst_feat), graph)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    else:
        raise ValueError(f"Unsupported node feature type: {type(nfeat)}")

    return efeat


@torch.jit.script
def sum_edge_node_feat(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    src_idx : Tensor
        Source node indices.
    dst_idx : Tensor
        Destination node indices.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """

    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def sum_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: GraphType,
):
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features (static setting) or tuple of node features of
        source and destination nodes (bipartite setting).
    graph : GraphType
        The underlying graph.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """
    if isinstance(nfeat, Tensor):
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                src_feat, dst_feat = nfeat, nfeat
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                sum_efeat = sum_edge_node_feat(efeat, src_feat, dst_feat, src, dst)

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(nfeat)
                    dst_feat = nfeat
                    bipartite_graph = graph.to_bipartite_csc()
                    sum_efeat = update_efeat_bipartite_e2e(
                        efeat, src_feat, dst_feat, bipartite_graph, mode="sum"
                    )

                else:
                    static_graph = graph.to_static_csc()
                    sum_efeat = update_efeat_bipartite_e2e(
                        efeat, nfeat, static_graph, mode="sum"
                    )
        elif isinstance(graph, DGLGraph):
            src_feat, dst_feat = nfeat, nfeat
            src, dst = (item.long() for item in graph.edges())
            sum_efeat = sum_edge_node_feat(efeat, src_feat, dst_feat, src, dst)
        elif isinstance(graph, PyGData):
            src_feat, dst_feat = nfeat, nfeat
            src, dst = graph.edge_index.long()
            sum_efeat = sum_edge_node_feat(efeat, src_feat, dst_feat, src, dst)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    else:
        src_feat, dst_feat = nfeat
        if isinstance(graph, CuGraphCSC):
            if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                src, dst = (item.long() for item in graph.to_dgl_graph().edges())
                sum_efeat = sum_edge_node_feat(efeat, src_feat, dst_feat, src, dst)

            else:
                if graph.is_distributed:
                    src_feat = graph.get_src_node_features_in_local_graph(src_feat)

                bipartite_graph = graph.to_bipartite_csc()
                sum_efeat = update_efeat_bipartite_e2e(
                    efeat, src_feat, dst_feat, bipartite_graph, mode="sum"
                )
        elif isinstance(graph, DGLGraph):
            src, dst = (item.long() for item in graph.edges())
            sum_efeat = sum_edge_node_feat(efeat, src_feat, dst_feat, src, dst)
        elif isinstance(graph, PyGData):
            src, dst = graph.edge_index.long()
            sum_efeat = sum_edge_node_feat(efeat, src_feat, dst_feat, src, dst)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")

    return sum_efeat


@torch.jit.ignore()
def agg_concat_dgl(
    efeat: Tensor, dst_nfeat: Tensor, graph: DGLGraph, aggregation: str
) -> Tensor:
    """Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = efeat

        # aggregate edge features
        if aggregation == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        # concat dst-node & edge features
        cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), -1)
        return cat_feat


def agg_concat_pyg(
    efeat: Tensor,
    nfeat: Tensor,
    graph: PyGData,
    aggregation: str,
) -> Tensor:
    src, dst = graph.edge_index
    h_dest = torch_scatter.scatter(
        efeat, dst, dim=0, dim_size=nfeat.shape[0], reduce=aggregation
    )
    cat_feat = torch.cat((h_dest, nfeat), -1)
    return cat_feat


def aggregate_and_concat(
    efeat: Tensor,
    nfeat: Tensor,
    graph: GraphType,
    aggregation: str,
):
    """
    Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : GraphType
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """

    if isinstance(graph, CuGraphCSC):
        # in this case, we don't have to distinguish a distributed setting
        # or the defalt setting as both efeat and nfeat are already
        # gurantueed to be on the same rank on both cases due to our
        # partitioning scheme

        if graph.dgl_graph is not None or not USE_CUGRAPHOPS:
            cat_feat = agg_concat_dgl(efeat, nfeat, graph.to_dgl_graph(), aggregation)

        else:
            static_graph = graph.to_static_csc()
            cat_feat = agg_concat_e2n(nfeat, efeat, static_graph, aggregation)
    elif isinstance(graph, DGLGraph):
        cat_feat = agg_concat_dgl(efeat, nfeat, graph, aggregation)
    elif isinstance(graph, PyGData):
        cat_feat = agg_concat_pyg(efeat, nfeat, graph, aggregation)
    else:
        raise ValueError(f"Unsupported graph type: {type(graph)}")

    return cat_feat
