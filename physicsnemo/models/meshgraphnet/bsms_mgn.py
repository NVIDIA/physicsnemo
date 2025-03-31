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

from dataclasses import dataclass
from typing import Iterable, List, Optional

from dgl import DGLGraph
from torch import Tensor

from physicsnemo.models.gnn_layers.bsms import BistrideGraphMessagePassing
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.meta import ModelMetaData


@dataclass
class MetaData(ModelMetaData):
    name: str = "BiStrideMeshGraphNet"
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
    """Bi-stride MeshGraphNet network architecture

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features
    output_dim : int
        Number of outputs
    processor_size : int, optional
        Number of message passing blocks, by default 15
    mlp_activation_fn : Union[str, List[str]], optional
        Activation function to use, by default 'relu'
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : Union[int, None], optional
        Number of MLP layers for the node feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no node encoder
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : Union[int, None], optional
        Number of MLP layers for the edge feature encoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no edge encoder
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : Union[int, None], optional
        Number of MLP layers for the node feature decoder, by default 2.
        If None is provided, the MLP will collapse to a Identity function, i.e. no decoder
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0 (checkpointing disabled).
        The number of segments should be a factor of 2 * `processor_size`, for example, if
        `processor_size` is 15, then `num_processor_checkpoint_segments` can be 10 since it's
        a factor of 15 * 2 = 30. It is recommended to start with a smaller number of segments
        until the model fits into memory since each segment will affect model training speed.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: str | List[str] = "relu",
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
        aggregation: str = "sum",
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
        graph: DGLGraph,
        ms_edges: Iterable[Tensor] = (),
        ms_ids: Iterable[Tensor] = (),
        **kwargs,
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)

        node_pos = graph.ndata["pos"]
        ms_edges = [es.to(node_pos.device).squeeze(0) for es in ms_edges]
        ms_ids = [ids.squeeze(0) for ids in ms_ids]
        for _ in range(self.bistride_unet_levels):
            x = self.bistride_processor(x, ms_ids, ms_edges, node_pos)
        x = self.node_decoder(x)
        return x
