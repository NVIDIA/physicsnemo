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
from typing import Iterable, Literal, Optional

import torch
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.nn.module.gnn_layers.bsms import BistrideGraphMessagePassing
from physicsnemo.nn.module.gnn_layers.utils import GraphType


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


class BiStrideMeshGraphNet(MeshGraphNet):
    r"""Bi-stride MeshGraphNet network architecture.

    Bi-stride MGN augments vanilla MGN with a U-Net-like multi-scale message
    passing that alternates between coarsening and refining the mesh. This
    improves modeling of long-range interactions.

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
    num_mesh_levels : int, optional, default=2
        Number of mesh levels used by the bi-stride U-Net (multi-scale) processor.
    bistride_pos_dim : int, optional, default=3
        Dimensionality of node positions stored in ``graph.pos`` (required by bi-stride).
    num_layers_bistride : int, optional, default=2
        Number of layers within each bi-stride message passing block.
    bistride_unet_levels : int, optional, default=1
        Number of times to apply the bi-stride U-Net (depth of repeat).
    hidden_dim_processor : int, optional, default=128
        Hidden layer size for the message passing blocks.
    hidden_dim_node_encoder : int, optional, default=128
        Hidden layer size for the node feature encoder.
    num_layers_node_encoder : Union[int, None], optional, default=2
        Number of MLP layers for the node feature encoder. If ``None`` is provided, the MLP
        collapses to an identity function (no node encoder).
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden layer size for the edge feature encoder.
    num_layers_edge_encoder : Union[int, None], optional, default=2
        Number of MLP layers for the edge feature encoder. If ``None`` is provided, the MLP
        collapses to an identity function (no edge encoder).
    hidden_dim_node_decoder : int, optional, default=128
        Hidden layer size for the node feature decoder.
    num_layers_node_decoder : Union[int, None], optional, default=2
        Number of MLP layers for the node feature decoder. If ``None`` is provided, the MLP
        collapses to an identity function (no decoder).
    aggregation : Literal["sum", "mean"], optional, default="sum"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.
    do_concat_trick : bool, optional, default=False
        Whether to replace concat+MLP with MLP+idx+sum.
    num_processor_checkpoint_segments : int, optional, default=0
        Number of processor segments for gradient checkpointing (checkpointing disabled if 0).
        The number of segments should be a factor of :math:`2\times\text{processor\_size}`.
        For example, if ``processor_size`` is 15, then ``num_processor_checkpoint_segments`` can be 10
        since it's a factor of :math:`15 \times 2 = 30`. Start with fewer segments if memory is tight,
        as each segment affects training speed.
    recompute_activation : bool, optional, default=False
        Whether to recompute activations during backward to reduce memory usage.

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
        Requires ``graph.pos`` with shape :math:`(N_{nodes}, \text{bistride\_pos\_dim})` for bi-stride.
    ms_edges : Iterable[torch.Tensor], optional
        Multi-scale edge lists; each is typically an integer tensor of shape :math:`(2, E_l)`.
    ms_ids : Iterable[torch.Tensor], optional
        Multi-scale node id tensors per level; typically shape :math:`(N_l,)`.

    Outputs
    -------
    torch.Tensor
        Output node features of shape :math:`(N_{nodes}, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet
    >>>
    >>> # Choose a device and create the model on it
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>>
    >>> # Create a simple graph
    >>> num_nodes = 8
    >>> src = torch.arange(num_nodes, device=device)
    >>> dst = (src + 1) % num_nodes
    >>> edge_index = torch.stack([src, dst], dim=0)  # (2, E)
    >>> graph = Data(edge_index=edge_index, num_nodes=num_nodes).to(device)
    >>> graph.pos = torch.randn(num_nodes, 3, device=device)  # position needed by bi-stride
    >>>
    >>> # Features
    >>> node_features = torch.randn(num_nodes, 10, device=device)
    >>> edge_features = torch.randn(edge_index.shape[1], 4, device=device)
    >>>
    >>> # Multi-scale inputs (one level for simplicity)
    >>> ms_edges = [edge_index, edge_index]  # list of (2, E_l) tensors
    >>> ms_ids = [torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)]  # list of (N_l,) tensors
    >>>
    >>> # Model
    >>> model = BiStrideMeshGraphNet(
    ...     input_dim_nodes=10,
    ...     input_dim_edges=4,
    ...     output_dim=4,
    ...     processor_size=2,
    ...     hidden_dim_processor=32,
    ...     hidden_dim_node_encoder=16,
    ...     hidden_dim_edge_encoder=16,
    ...     num_layers_bistride=1,
    ...     num_mesh_levels=1,
    ... ).to(device)
    >>>
    >>> out = model(node_features, edge_features, graph, ms_edges, ms_ids)
    >>> out.size()
    torch.Size([8, 4])

    Note
    ----
    Reference: `Efficient Learning of Mesh-Based Physical Simulation with
    Bi-Stride Multi-Scale Graph Neural Network <https://arxiv.org/pdf/2210.02573>`.
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
        num_mesh_levels: int = 2,
        bistride_pos_dim: int = 3,
        num_layers_bistride: int = 2,
        bistride_unet_levels: int = 1,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Optional[int] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Optional[int] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Optional[int] = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        recompute_activation: bool = False,
    ):
        super().__init__(
            input_dim_nodes,
            input_dim_edges,
            output_dim,
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
            recompute_activation=recompute_activation,
        )
        self.meta = MetaData()

        self.bistride_unet_levels = bistride_unet_levels

        self.bistride_processor = BistrideGraphMessagePassing(
            unet_depth=num_mesh_levels,
            latent_dim=hidden_dim_processor,
            hidden_layer=num_layers_bistride,
            pos_dim=bistride_pos_dim,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        ms_edges: Iterable[Tensor] = (),
        ms_ids: Iterable[Tensor] = (),
        **kwargs,
    ) -> Tensor:
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

        node_pos = graph.pos
        ms_edges = [es.to(node_pos.device).squeeze(0) for es in ms_edges]
        ms_ids = [ids.squeeze(0) for ids in ms_ids]
        for _ in range(self.bistride_unet_levels):
            x = self.bistride_processor(x, ms_ids, ms_edges, node_pos)
        x = self.node_decoder(x)
        return x
