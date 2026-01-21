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

from dataclasses import dataclass
from itertools import chain
from typing import Callable, Literal, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import get_activation
from physicsnemo.nn.gnn_layers.mesh_edge_block import HybridMeshEdgeBlock
from physicsnemo.nn.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.gnn_layers.mesh_node_block import HybridMeshNodeBlock
from physicsnemo.nn.gnn_layers.utils import GraphType
from physicsnemo.utils.profiling import profile

# Import the MeshGraphNet
from .meshgraphnet import MeshGraphNet, MeshGraphNetProcessor


@dataclass
class HybridMetaData(ModelMetaData):
    """Metadata for HybridMeshGraphNet"""

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


class HybridMeshGraphNet(MeshGraphNet):
    r"""Hybrid MeshGraphNet with separate mesh and world edge encoders.

    This class extends the vanilla MeshGraphNet to support hybrid functionality
    with separate encoders for mesh edges and world edges.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features.
    input_dim_edges : int
        Number of edge features (applies to both mesh and world edges).
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
        Number of MLP layers for the node feature encoder.
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden layer size for the edge feature encoders.
    num_layers_edge_encoder : Union[int, None], optional, default=2
        Number of MLP layers for the edge feature encoders.
    hidden_dim_node_decoder : int, optional, default=128
        Hidden layer size for the node feature decoder.
    num_layers_node_decoder : Union[int, None], optional, default=2
        Number of MLP layers for the node feature decoder.
    aggregation : Literal["sum", "mean"], optional, default="sum"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.
    do_concat_trick : bool, optional, default=False
        Whether to replace concat+MLP with MLP+idx+sum.
    num_processor_checkpoint_segments : int, optional, default=0
        Number of processor segments for gradient checkpointing (0 disables checkpointing).
    checkpoint_offloading : bool, optional, default=False
        Whether to offload checkpointing to CPU.
    recompute_activation : bool, optional, default=False
        Whether to recompute activations.
    norm_type : Literal["LayerNorm", "TELayerNorm"], optional, default="LayerNorm"
        Normalization type. Allowed values are ``"LayerNorm"`` and ``"TELayerNorm"``.
        ``"TELayerNorm"`` refers to the Transformer Engine implementation of LayerNorm and
        requires NVIDIA Transformer Engine to be installed (optional dependency).

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}, D_{in}^{node})`.
    mesh_edge_features : torch.Tensor
        Mesh edge features of shape :math:`(N_{mesh\_edges}, D_{in}^{edge})`.
    world_edge_features : torch.Tensor
        World edge features of shape :math:`(N_{world\_edges}, D_{in}^{edge})`.
    graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
        Connectivity/topology only. Do not duplicate node or edge features on the graph;
        pass them via ``node_features`` and ``edge_features``. If present on
        the graph, they will be ignored by the model.
        ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
        ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
        The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
        PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
        :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.


    Outputs
    -------
    torch.Tensor
        Output node features of shape :math:`(N_{nodes}, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
    >>>
    >>> # Create model on a consistent device
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model = HybridMeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=2).to(device)
    >>>
    >>> # Create connectivity: mesh edges and world edges, then combine
    >>> num_nodes, num_mesh_edges, num_world_edges = 10, 5, 5
    >>> mesh_src = torch.randint(0, num_nodes, (num_mesh_edges,))
    >>> mesh_dst = torch.randint(0, num_nodes, (num_mesh_edges,))
    >>> mesh_edge_index = torch.stack([mesh_src, mesh_dst], dim=0)  # (2, E_mesh)
    >>> world_src = torch.randint(0, num_nodes, (num_world_edges,))
    >>> world_dst = torch.randint(0, num_nodes, (num_world_edges,))
    >>> world_edge_index = torch.stack([world_src, world_dst], dim=0)  # (2, E_world)
    >>> edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)  # (2, E)
    >>> graph = Data(edge_index=edge_index, num_nodes=num_nodes).to(device)
    >>>
    >>> # Features: pass separately for mesh and world edges, and for nodes
    >>> node_features = torch.randn(num_nodes, 4, device=device)
    >>> mesh_edge_features = torch.randn(num_mesh_edges, 3, device=device)
    >>> world_edge_features = torch.randn(num_world_edges, 3, device=device)
    >>>
    >>> # Forward
    >>> out = model(node_features, mesh_edge_features, world_edge_features, graph)
    >>> out.size()
    torch.Size([10, 2])

    Note
    ----
    The HybridMeshGraphNet requires separate feature tensors for mesh edges and world edges,
    allowing for different processing pipelines for different edge types.

    Note
    ----
    Reference:
    - `Learning Mesh-Based Simulation with Graph Networks <https://arxiv.org/pdf/2010.03409>`.
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
        # Initialize the parent class
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder,
            hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
            recompute_activation=recompute_activation,
            norm_type=norm_type,
        )

        if do_concat_trick:
            raise NotImplementedError(
                "Concat trick is not supported for HybridMeshGraphNet yet."
            )

        if recompute_activation:
            raise NotImplementedError(
                "Recompute activation is not supported for HybridMeshGraphNet yet."
            )

        # Override metadata
        self.meta = HybridMetaData()

        # Get activation function for the new encoder
        activation_fn = get_activation(mlp_activation_fn)

        # Convert single edge_encoder to mesh_edge_encoder
        self.mesh_edge_encoder = self.edge_encoder
        del self.edge_encoder

        # Add world_edge_encoder
        self.world_edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # Replace processor with hybrid version
        self.processor = HybridMeshGraphNetProcessor(
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
        mesh_edge_features: Float[torch.Tensor, "num_mesh_edges input_dim_edges"],
        world_edge_features: Float[torch.Tensor, "num_world_edges input_dim_edges"],
        graph: GraphType,
        **kwargs,
    ) -> Float[torch.Tensor, "num_nodes output_dim"]:
        r"""Forward pass for hybrid MeshGraphNet.

        Parameters
        ----------
        node_features : torch.Tensor
            Input node features of shape :math:`(N_{nodes}, D_{in}^{node})`.
        mesh_edge_features : torch.Tensor
            Mesh edge features of shape :math:`(N_{mesh\_edges}, D_{in}^{edge})`.
        world_edge_features : torch.Tensor
            World edge features of shape :math:`(N_{world\_edges}, D_{in}^{edge})`.
        graph : GraphType
            Graph container.

        Returns
        -------
        torch.Tensor
            Output node features of shape :math:`(N_{nodes}, D_{out})`.
        """
        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_nodes}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if (
                mesh_edge_features.ndim != 2
                or mesh_edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_mesh_edges, {self.input_dim_edges}) but got tensor of shape {tuple(mesh_edge_features.shape)}"
                )
            if (
                world_edge_features.ndim != 2
                or world_edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_world_edges, {self.input_dim_edges}) but got tensor of shape {tuple(world_edge_features.shape)}"
                )
        mesh_edge_features = self.mesh_edge_encoder(mesh_edge_features)
        world_edge_features = self.world_edge_encoder(world_edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(
            node_features, mesh_edge_features, world_edge_features, graph
        )
        x = self.node_decoder(x)
        return x


class HybridMeshGraphNetProcessor(MeshGraphNetProcessor):
    r"""Hybrid MeshGraphNet processor that handles both mesh and world edges.

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
        Message aggregation method used in node update blocks. Allowed values are ``"sum"`` and ``"mean"``.
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
    mesh_edge_features : torch.Tensor
        Mesh edge features of shape :math:`(N_{mesh\_edges}, D_{edge})`.
    world_edge_features : torch.Tensor
        World edge features of shape :math:`(N_{world\_edges}, D_{edge})`.
    graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
        Connectivity/topology only. Do not duplicate node or edge features on the graph;
        pass them via ``node_features`` and ``edge_features``. If present on
        the graph, they will be ignored by the model.
        ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
        ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
        The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
        PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
        :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.

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
        super().__init__(
            processor_size=processor_size,
            input_dim_node=input_dim_node,
            input_dim_edge=input_dim_edge,
            num_layers_node=num_layers_node,
            num_layers_edge=num_layers_edge,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
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
            HybridMeshEdgeBlock(*edge_block_invars) for _ in range(self.processor_size)
        ]
        node_blocks = [
            HybridMeshNodeBlock(*node_block_invars) for _ in range(self.processor_size)
        ]
        layers = list(chain(*zip(edge_blocks, node_blocks)))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)

    @profile
    def _run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[[Tensor, Tensor, Tensor, GraphType], Tuple[Tensor, Tensor, Tensor]]:
        r"""Create a segment function for gradient checkpointing (hybrid).

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment.
        segment_end : int
            Layer index as end of the segment (exclusive).

        Returns
        -------
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, GraphType], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            Custom forward function that updates mesh-edge, world-edge, and node features.
        """
        segment = self.processor_layers[segment_start:segment_end]

        def _custom_forward(
            node_features: Tensor,
            mesh_edge_features: Tensor,
            world_edge_features: Tensor,
            graph: GraphType,
        ) -> Tuple[Tensor, Tensor, Tensor]:
            r"""Custom forward function for a hybrid processor segment.

            Parameters
            ----------
            node_features : torch.Tensor
                Node features of shape :math:`(N_{nodes}, D_{node})`.
            mesh_edge_features : torch.Tensor
                Mesh edge features of shape :math:`(N_{mesh\_edges}, D_{edge})`.
            world_edge_features : torch.Tensor
                World edge features of shape :math:`(N_{world\_edges}, D_{edge})`.
            graph : GraphType
                Graph container.

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                Updated ``(mesh_edge_features, world_edge_features, node_features)``.
            """
            for module in segment:
                mesh_edge_features, world_edge_features, node_features = module(
                    mesh_edge_features, world_edge_features, node_features, graph
                )
            return mesh_edge_features, world_edge_features, node_features

        return _custom_forward

    @profile
    def forward(
        self,
        node_features: Tensor,
        mesh_edge_features: Tensor,
        world_edge_features: Tensor,
        graph: GraphType,
    ) -> Tensor:
        if not torch.compiler.is_compiling():
            if node_features.ndim != 2 or node_features.shape[1] != self.input_dim_node:
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_node}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if (
                mesh_edge_features.ndim != 2
                or mesh_edge_features.shape[1] != self.input_dim_edge
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_mesh_edges, {self.input_dim_edge}) but got tensor of shape {tuple(mesh_edge_features.shape)}"
                )
            if (
                world_edge_features.ndim != 2
                or world_edge_features.shape[1] != self.input_dim_edge
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_world_edges, {self.input_dim_edge}) but got tensor of shape {tuple(world_edge_features.shape)}"
                )
        with self.checkpoint_offload_ctx:
            for segment_start, segment_end in self.checkpoint_segments:
                mesh_edge_features, world_edge_features, node_features = (
                    self.checkpoint_fn(
                        self._run_function(segment_start, segment_end),
                        node_features,
                        mesh_edge_features,
                        world_edge_features,
                        graph,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )

        return node_features
