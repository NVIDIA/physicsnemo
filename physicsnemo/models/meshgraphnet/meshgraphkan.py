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
from typing import Literal, Union

import torch
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn.gnn_layers.graph_types import GraphType

# Import the Kolmogorov–Arnold Network layer.
from physicsnemo.nn.kan_layers import KolmogorovArnoldNetwork

from .meshgraphnet import MeshGraphNet


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


class MeshGraphKAN(MeshGraphNet):
    r"""MeshGraphKAN with a Kolmogorov–Arnold Network (KAN) node encoder.

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
    Activation function for non-KAN components.
    num_layers_node_processor : int, optional, default=2
        Number of MLP layers for processing nodes in each message passing block.
    num_layers_edge_processor : int, optional, default=2
        Number of MLP layers for processing edge features in each message passing block.
    hidden_dim_processor : int, optional, default=128
        Hidden layer size for the message passing blocks.
    hidden_dim_node_encoder : int, optional, default=128
        Output dimension for the KAN node encoder.
    num_layers_node_encoder : Union[int, None], optional, default=2
        Ignored for the KAN node encoder.
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden layer size for the edge feature encoder.
    num_layers_edge_encoder : Union[int, None], optional, default=2
        Number of MLP layers for the edge feature encoder.
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
        Whether to offload checkpointing to the CPU.
    recompute_activation : bool, optional, default=False
        Whether to recompute activations during backward for memory savings.
    num_harmonics : int, optional, default=5
        Number of Fourier harmonics used in the KAN node encoder.

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}, D_{in}^{node})`.
    edge_features : torch.Tensor
        Input edge features of shape :math:`(N_{edges}, D_{in}^{edge})`.
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
    >>> # ``norm_type`` in MeshGraphNet layers is deprecated,
    >>> # TE will be automatically used if possible unless told otherwise.
    >>> # (You don't have to set this variable, it's faster to use TE!)
    >>> # Example of how to disable:
    >>> import os
    >>> os.environ['PHYSICSNEMO_FORCE_TE'] = 'False'
    >>>
    >>> model = MeshGraphKAN(
    ...     input_dim_nodes=4,
    ...     input_dim_edges=3,
    ...     output_dim=2,
    ... )
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
    References:
    - `Learning Mesh-Based Simulation with Graph Networks <https://arxiv.org/pdf/2010.03409>`.
    - `KAN: Kolmogorov–Arnold Networks <https://arxiv.org/pdf/2404.19756>`.
    - `Interpretable physics-informed graph neural networks for
        flood forecasting <https://onlinelibrary.wiley.com/doi/pdf/10.1111/mice.13484>`.
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
        num_layers_node_encoder: Union[int, None] = 2,  # Ignored for KAN.
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        num_harmonics: int = 5,
    ):
        # Build the standard MGN components
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
            norm_type="LayerNorm",
        )

        # Override metadata to KAN-specific
        self.meta = MetaData()

        # Replace the standard MLP node encoder with the KAN layer.
        self.node_encoder = KolmogorovArnoldNetwork(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            num_harmonics=num_harmonics,
            add_bias=True,
        )

    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        edge_features: Float[torch.Tensor, "num_edges input_dim_edges"],
        graph: Union[GraphType, list[GraphType]],
        **kwargs,
    ) -> Float[torch.Tensor, "num_nodes output_dim"]:
        # Reuse MeshGraphNet.forward (encodes edges, encodes nodes with KAN, runs processor, decodes)
        return super().forward(node_features, edge_features, graph, **kwargs)
