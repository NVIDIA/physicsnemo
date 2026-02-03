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

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Literal, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import get_activation
from physicsnemo.nn.module.gnn_layers.mesh_edge_block import MeshEdgeBlock
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.gnn_layers.mesh_node_block import MeshNodeBlock
from physicsnemo.nn.module.gnn_layers.utils import GraphType, set_checkpoint_fn
from physicsnemo.utils.profiling import profile


@dataclass
class MetaData(ModelMetaData):
    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class MeshGraphNet(Module):
    r"""MeshGraphNet network architecture.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features.
    input_dim_edges : int
        Number of edge features.
    output_dim : int
        Number of outputs.
    processor_size : int, optional, default=15
        Number of message passing blocks.
    mlp_activation_fn : str, optional, default="relu"
        Activation function to use.
    num_layers_node_processor : int, optional, default=2
        Number of MLP layers for processing nodes in each message passing block.
    num_layers_edge_processor : int, optional, default=2
        Number of MLP layers for processing edge features in each message passing block.
    hidden_dim_processor : int, optional, default=128
        Hidden layer size for the message passing blocks.
    hidden_dim_node_encoder : int, optional, default=128
        Hidden layer size for the node feature encoder.
    num_layers_node_encoder : Union[int, None], optional, default=2
        Number of MLP layers for the node feature encoder. If ``None`` is provided,
        the MLP collapses to an identity function, i.e. no node encoder.
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden layer size for the edge feature encoder.
    num_layers_edge_encoder : Union[int, None], optional, default=2
        Number of MLP layers for the edge feature encoder. If ``None`` is provided,
        the MLP collapses to an identity function, i.e. no edge encoder.
    hidden_dim_node_decoder : int, optional, default=128
        Hidden layer size for the node feature decoder.
    num_layers_node_decoder : Union[int, None], optional, default=2
        Number of MLP layers for the node feature decoder. If ``None`` is provided,
        the MLP collapses to an identity function, i.e. no decoder.
    aggregation : Literal["sum", "mean"], optional, default="sum"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.
    do_concat_trick : bool, optional, default=False
        Whether to replace concat+MLP with MLP+idx+sum.
    num_processor_checkpoint_segments : int, optional, default=0
        Number of processor segments for gradient checkpointing (0 disables checkpointing).
    checkpoint_offloading : bool, optional, default=False
        Whether to offload the checkpointing to the CPU.
    norm_type : Literal["LayerNorm", "TELayerNorm"], optional, default="LayerNorm"
        Normalization type. Allowed values are ``"LayerNorm"`` and ``"TELayerNorm"``.
        ``"TELayerNorm"`` refers to the Transformer Engine implementation of LayerNorm and
        requires NVIDIA Transformer Engine to be installed (optional dependency).

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}, D_{in}^{node})`.
    edge_features : torch.Tensor
        Input edge features of shape :math:`(N_{edges}, D_{in}^{edge})`.
    graph : :class:`~physicsnemo.nn.module.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
        Connectivity/topology only. Do not duplicate node or edge features on the graph;
        pass them via ``node_features`` and ``edge_features``. If present on
        the graph, they will be ignored by the model.
        ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
        ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
        The current :class:`~physicsnemo.nn.module.gnn_layers.graph_types.GraphType` resolves to
        PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
        :mod:`physicsnemo.nn.module.gnn_layers.graph_types` for the exact alias and requirements.

    Outputs
    -------
    torch.Tensor
        Output node features of shape :math:`(N_{nodes}, D_{out})`.

    Examples
    --------
    >>> # ``norm_type`` in MeshGraphNet is deprecated,
    >>> # TE will be automatically used if possible unless told otherwise.
    >>> # (You don't have to set this variable, it's faster to use TE!)
    >>> # Example of how to disable:
    >>> import os
    >>> os.environ['PHYSICSNEMO_FORCE_TE'] = 'False'
    >>>
    >>> model = physicsnemo.models.meshgraphnet.MeshGraphNet(
    ...         input_dim_nodes=4,
    ...         input_dim_edges=3,
    ...         output_dim=2,
    ...     )
    >>> from torch_geometric.data import Data
    >>> edge_index = torch.randint(0, 10, (2, 5))
    >>> graph = Data(edge_index=edge_index)
    >>> node_features = torch.randn(10, 4)
    >>> edge_features = torch.randn(5, 3)
    >>> output = model(node_features, edge_features, graph)
    >>> output.size()
    torch.Size([10, 2])

    Note
    ----
    Reference: `Learning Mesh-Based Simulation with Graph Networks <https://arxiv.org/pdf/2010.03409>`.

    See also :class:`~physicsnemo.nn.module.gnn_layers.mesh_graph_mlp.MeshGraphMLP`,
    :class:`~physicsnemo.nn.module.gnn_layers.mesh_edge_block.MeshEdgeBlock`,
    and :class:`~physicsnemo.nn.module.gnn_layers.mesh_node_block.MeshNodeBlock`.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
    ):
        super().__init__(meta=MetaData())

        # Store public constructor attributes used for validation and serialization
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim

        activation_fn = get_activation(mlp_activation_fn)

        if norm_type not in ["LayerNorm", "TELayerNorm"]:
            raise ValueError("Norm type should be either 'LayerNorm' or 'TELayerNorm'")

        if not torch.cuda.is_available() and norm_type == "TELayerNorm":
            warn("TELayerNorm is not supported on CPU. Switching to LayerNorm.")
            norm_type = "LayerNorm"

        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
        )

    @profile
    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        edge_features: Float[torch.Tensor, "num_edges input_dim_edges"],
        graph: GraphType,
        **kwargs,
    ) -> Float[torch.Tensor, "num_nodes output_dim"]:
        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_nodes}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
                )

        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(x)
        return x


class MeshGraphNetProcessor(Module):
    r"""MeshGraphNet processor block.

    Parameters
    ----------
    processor_size : int, optional, default=15
        Number of alternating edge/node update layers in the processor.
    input_dim_node : int, optional, default=128
        Dimensionality of per-node hidden features provided to the processor.
    input_dim_edge : int, optional, default=128
        Dimensionality of per-edge hidden features provided to the processor.
    num_layers_node : int, optional, default=2
        Number of MLP layers within each node update block.
    num_layers_edge : int, optional, default=2
        Number of MLP layers within each edge update block.
    aggregation : Literal["sum", "mean"], optional, default="sum"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.
    norm_type : Literal["LayerNorm", "TELayerNorm"], optional, default="LayerNorm"
        Normalization type. Allowed values are ``"LayerNorm"`` and ``"TELayerNorm"``.
        ``"TELayerNorm"`` uses the Transformer Engine LayerNorm and requires NVIDIA
        Transformer Engine to be installed.
    activation_fn : torch.nn.Module, optional, default=nn.ReLU()
        Activation function module used inside the MLPs.
    do_concat_trick : bool, optional, default=False
        Whether to replace concat+MLP with MLP+idx+sum.
    num_processor_checkpoint_segments : int, optional, default=0
        Number of checkpoint segments across processor layers (0 disables checkpointing).
    checkpoint_offloading : bool, optional, default=False
        Whether to offload checkpoint activations to CPU.

    Forward
    -------
    node_features : torch.Tensor
        Node features of shape :math:`(N_{nodes}, D_{node})`.
    edge_features : torch.Tensor
        Edge features of shape :math:`(N_{edges}, D_{edge})`.
    graph : :class:`~physicsnemo.nn.module.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
        Connectivity/topology only. Do not duplicate node or edge features on the graph;
        pass them via ``node_features`` and ``edge_features``. If present on
        the graph, they will be ignored by the model.
        ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
        ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
        The current :class:`~physicsnemo.nn.module.gnn_layers.graph_types.GraphType` resolves to
        PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
        :mod:`physicsnemo.nn.module.gnn_layers.graph_types` for the exact alias and requirements.

    Outputs
    -------
    torch.Tensor
        Updated node features of shape :math:`(N_{nodes}, D_{node})`.
    """

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
        activation_fn: nn.Module = nn.ReLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments
        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.checkpoint_offloading = (
            checkpoint_offloading if (num_processor_checkpoint_segments > 0) else False
        )

        edge_block_invars = (
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_edge,
            activation_fn,
            norm_type,
            do_concat_trick,
            False,
        )
        node_block_invars = (
            aggregation,
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_node,
            activation_fn,
            norm_type,
            False,
        )

        edge_blocks = [
            MeshEdgeBlock(*edge_block_invars) for _ in range(self.processor_size)
        ]
        node_blocks = [
            MeshNodeBlock(*node_block_invars) for _ in range(self.processor_size)
        ]
        layers = list(chain(*zip(edge_blocks, node_blocks)))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        self._set_checkpoint_segments(self.num_processor_checkpoint_segments)
        self._set_checkpoint_offload_ctx(self.checkpoint_offloading)

    def _set_checkpoint_offload_ctx(self, enabled: bool) -> None:
        r"""Set the context for CPU offloading of checkpoints.

        Parameters
        ----------
        enabled : bool
            If ``True``, offload checkpoint activations to CPU using
            :func:`torch.autograd.graph.save_on_cpu`.

        Returns
        -------
        None
        """
        if enabled:
            self.checkpoint_offload_ctx = torch.autograd.graph.save_on_cpu(
                pin_memory=True
            )
        else:
            self.checkpoint_offload_ctx = nullcontext()

    def _set_checkpoint_segments(self, checkpoint_segments: int) -> None:
        r"""Set the number of checkpoint segments.

        Parameters
        ----------
        checkpoint_segments : int
            Number of checkpoint segments. If greater than 0, the number of
            processor layers must be divisible by ``checkpoint_segments``.

        Raises
        ------
        ValueError
            If the number of processor layers is not a multiple of the number of
            checkpoint segments.

        Returns
        -------
        None
        """
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    @profile
    def _run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[[Tensor, Tensor, GraphType], Tuple[Tensor, Tensor]]:
        r"""Create a segment function for gradient checkpointing.

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment.
        segment_end : int
            Layer index as end of the segment (exclusive).

        Returns
        -------
        Callable[[torch.Tensor, torch.Tensor, GraphType], Tuple[torch.Tensor, torch.Tensor]]
            Custom forward function that updates edge and node features for the segment.
        """
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: GraphType,
        ) -> Tuple[Tensor, Tensor]:
            r"""Custom forward function for a processor segment.

            Parameters
            ----------
            node_features : torch.Tensor
                Node features of shape :math:`(N_{nodes}, D_{node})`.
            edge_features : torch.Tensor
                Edge features of shape :math:`(N_{edges}, D_{edge})`.
            graph : GraphType
                Graph container.

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor]
                Updated ``(edge_features, node_features)``.
            """
            for module in segment:
                edge_features, node_features = module(
                    edge_features, node_features, graph
                )
            return edge_features, node_features

        return custom_forward

    @profile
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
    ) -> Tensor:
        if not torch.compiler.is_compiling():
            if node_features.ndim != 2 or node_features.shape[1] != self.input_dim_node:
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_node}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if edge_features.ndim != 2 or edge_features.shape[1] != self.input_dim_edge:
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edge}) but got tensor of shape {tuple(edge_features.shape)}"
                )
        with self.checkpoint_offload_ctx:
            for segment_start, segment_end in self.checkpoint_segments:
                edge_features, node_features = self.checkpoint_fn(
                    self._run_function(segment_start, segment_end),
                    node_features,
                    edge_features,
                    graph,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

        return node_features
