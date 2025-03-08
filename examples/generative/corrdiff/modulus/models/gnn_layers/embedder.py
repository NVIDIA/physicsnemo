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

from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .mesh_graph_mlp import MeshGraphMLP


class GraphCastEncoderEmbedder(nn.Module):
    """GraphCast feature embedder for gird node features, multimesh node features,
    grid2mesh edge features, and multimesh edge features.

    Parameters
    ----------
    input_dim_grid_nodes : int, optional
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim : int, optional
        Dimensionality of the embedded features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        # MLP for grid node embedding
        self.grid_node_mlp = MeshGraphMLP(
            input_dim=input_dim_grid_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for mesh node embedding
        self.mesh_node_mlp = MeshGraphMLP(
            input_dim=input_dim_mesh_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for mesh edge embedding
        self.mesh_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for grid2mesh edge embedding
        self.grid2mesh_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        g2m_efeat: Tensor,
        mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Input node feature embedding
        grid_nfeat = self.grid_node_mlp(grid_nfeat)
        mesh_nfeat = self.mesh_node_mlp(mesh_nfeat)
        # Input edge feature embedding
        g2m_efeat = self.grid2mesh_edge_mlp(g2m_efeat)
        mesh_efeat = self.mesh_edge_mlp(mesh_efeat)
        return grid_nfeat, mesh_nfeat, g2m_efeat, mesh_efeat


class GraphCastDecoderEmbedder(nn.Module):
    """GraphCast feature embedder for mesh2grid edge features

    Parameters
    ----------
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim : int, optional
        Dimensionality of the embedded features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        # MLP for mesh2grid edge embedding
        self.mesh2grid_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        m2g_efeat: Tensor,
    ) -> Tensor:
        m2g_efeat = self.mesh2grid_edge_mlp(m2g_efeat)
        return m2g_efeat
