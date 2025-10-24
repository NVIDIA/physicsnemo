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
from typing import TypeAlias

try:
    from dgl import DGLGraph
except ImportError:
    warnings.warn(
        "Note: This only applies if you're using DGL.\n"
        "MeshGraphNet (DGL version) requires the DGL library.\n"
        "Install it with your preferred CUDA version from:\n"
        "https://www.dgl.ai/pages/start.html\n"
    )

    DGLGraph: TypeAlias = NoneType

import torch
import torch_cluster
import torch_geometric as pyg
import torch_scatter

from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet


class Mesh_Reduced(torch.nn.Module):
    """PbGMR-GMUS architecture.

    A mesh-reduced architecture that combines encoding and decoding processors
    for physics prediction in reduced mesh space.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features.
    input_dim_edges : int
        Number of edge features.
    output_decode_dim : int
        Number of decoding outputs (per node).
    output_encode_dim : int, optional
        Number of encoding outputs (per pivotal position), by default 3.
    processor_size : int, optional
        Number of message passing blocks, by default 15.
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2.
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2.
    hidden_dim_processor : int, optional
        Hidden layer size for the message passing blocks, by default 128.
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128.
    num_layers_node_encoder : int, optional
        Number of MLP layers for the node feature encoder, by default 2.
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128.
    num_layers_edge_encoder : int, optional
        Number of MLP layers for the edge feature encoder, by default 2.
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128.
    num_layers_node_decoder : int, optional
        Number of MLP layers for the node feature decoder, by default 2.
    k : int, optional
        Number of nodes considered for per pivotal position, by default 3.
    aggregation : str, optional
        Message aggregation type, by default "mean".

    Notes
    -----
    Reference: Han, Xu, et al. "Predicting physics in mesh-reduced space with temporal attention."
    arXiv preprint arXiv:2201.09113 (2022).
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_decode_dim: int,
        output_encode_dim: int = 3,
        processor_size: int = 15,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        k: int = 3,
        aggregation: str = "mean",
    ):
        super(Mesh_Reduced, self).__init__()
        self.knn_encoder_already = False
        self.knn_decoder_already = False
        self.encoder_processor = MeshGraphNet(
            input_dim_nodes,
            input_dim_edges,
            output_encode_dim,
            processor_size,
            "relu",
            num_layers_node_processor,
            num_layers_edge_processor,
            hidden_dim_processor,
            hidden_dim_node_encoder,
            num_layers_node_encoder,
            hidden_dim_edge_encoder,
            num_layers_edge_encoder,
            hidden_dim_node_decoder,
            num_layers_node_decoder,
            aggregation,
        )
        self.decoder_processor = MeshGraphNet(
            output_encode_dim,
            input_dim_edges,
            output_decode_dim,
            processor_size,
            "relu",
            num_layers_node_processor,
            num_layers_edge_processor,
            hidden_dim_processor,
            hidden_dim_node_encoder,
            num_layers_node_encoder,
            hidden_dim_edge_encoder,
            num_layers_edge_encoder,
            hidden_dim_node_decoder,
            num_layers_node_decoder,
            aggregation,
        )
        self.k = k
        self.PivotalNorm = torch.nn.LayerNorm(output_encode_dim)

    def knn_interpolate(
        self,
        x: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        batch_x: torch.Tensor = None,
        batch_y: torch.Tensor = None,
        k: int = 3,
        num_workers: int = 1,
    ):
        """Perform k-nearest neighbor interpolation.

        Parameters
        ----------
        x : torch.Tensor
            Input features to interpolate.
        pos_x : torch.Tensor
            Source positions.
        pos_y : torch.Tensor
            Target positions.
        batch_x : torch.Tensor, optional
            Batch indices for source positions, by default None.
        batch_y : torch.Tensor, optional
            Batch indices for target positions, by default None.
        k : int, optional
            Number of nearest neighbors to consider, by default 3.
        num_workers : int, optional
            Number of workers for parallel processing, by default 1.

        Returns
        -------
        torch.Tensor
            Interpolated features.
        torch.Tensor
            Source indices.
        torch.Tensor
            Target indices.
        torch.Tensor
            Interpolation weights.
        """
        with torch.no_grad():
            assign_index = torch_cluster.knn(
                pos_x,
                pos_y,
                k,
                batch_x=batch_x,
                batch_y=batch_y,
                num_workers=num_workers,
            )
            y_idx, x_idx = assign_index[0], assign_index[1]
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y = torch_scatter.scatter(
            x[x_idx] * weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
        )
        y = y / torch_scatter.scatter(
            weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
        )

        return y.float(), x_idx, y_idx, weights

    def encode(self, x, edge_features, graph, position_mesh, position_pivotal):
        """Encode mesh features to pivotal space.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_features : torch.Tensor
            Edge features.
        graph : Union[DGLGraph, pyg.data.Data]
            Input graph.
        position_mesh : torch.Tensor
            Mesh positions.
        position_pivotal : torch.Tensor
            Pivotal positions.

        Returns
        -------
        torch.Tensor
            Encoded features in pivotal space.
        """
        x = self.encoder_processor(x, edge_features, graph)
        x = self.PivotalNorm(x)
        nodes_index = torch.arange(graph.batch_size).to(x.device)
        if isinstance(graph, DGLGraph):
            batch_mesh = nodes_index.repeat_interleave(graph.batch_num_nodes())
        elif isinstance(graph, pyg.data.Data):
            batch_mesh = graph.batch
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
        position_mesh_batch = position_mesh.repeat(graph.batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(graph.batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * graph.batch_size).to(x.device)
        )

        x, _, _, _ = self.knn_interpolate(
            x=x,
            pos_x=position_mesh_batch,
            pos_y=position_pivotal_batch,
            batch_x=batch_mesh,
            batch_y=batch_pivotal,
        )

        return x

    def decode(self, x, edge_features, graph, position_mesh, position_pivotal):
        """Decode pivotal features back to mesh space.

        Parameters
        ----------
        x : torch.Tensor
            Input features in pivotal space.
        edge_features : torch.Tensor
            Edge features.
        graph : Union[DGLGraph, pyg.data.Data]
            Input graph.
        position_mesh : torch.Tensor
            Mesh positions.
        position_pivotal : torch.Tensor
            Pivotal positions.

        Returns
        -------
        torch.Tensor
            Decoded features in mesh space.
        """
        nodes_index = torch.arange(graph.batch_size).to(x.device)
        if isinstance(graph, DGLGraph):
            batch_mesh = nodes_index.repeat_interleave(graph.batch_num_nodes())
        elif isinstance(graph, pyg.data.Data):
            batch_mesh = graph.batch
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
        position_mesh_batch = position_mesh.repeat(graph.batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(graph.batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * graph.batch_size).to(x.device)
        )

        x, _, _, _ = self.knn_interpolate(
            x=x,
            pos_x=position_pivotal_batch,
            pos_y=position_mesh_batch,
            batch_x=batch_pivotal,
            batch_y=batch_mesh,
        )

        x = self.decoder_processor(x, edge_features, graph)
        return x
