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

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.utils.profiling import profile

from .mesh_graph_mlp import (
    MeshGraphEdgeMLPConcat,
    MeshGraphEdgeMLPSum,
    MeshGraphHeteroEdgeMLPConcat,
)
from .utils import GraphType


class MeshEdgeBlock(nn.Module):
    """Edge block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.

    Parameters
    ----------
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat

        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    @profile
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: GraphType,
    ) -> Tensor:
        efeat_new = self.edge_mlp(efeat, nfeat, graph)
        efeat_new = efeat_new + efeat
        return efeat_new, nfeat


class HybridMeshEdgeBlock(nn.Module):
    """Hybrid Edge block that extends MeshEdgeBlock to support both mesh and world edge features

    This class extends the MeshEdgeBlock to support hybrid functionality
    with separate processing for mesh edges and world edges.

    Parameters
    ----------
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        Hidden layer size, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_concat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.edge_mlp = MeshGraphHeteroEdgeMLPConcat(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    @profile
    def forward(
        self,
        mesh_efeat: Tensor,
        world_efeat: Tensor,
        nfeat: Tensor,
        graph: GraphType,
    ) -> Tensor:
        """Forward pass for hybrid edge processing

        Parameters
        ----------
        mesh_efeat : Tensor
            Mesh edge features
        world_efeat : Tensor
            World edge features
        nfeat : Tensor
            Node features
        graph : GraphType
            Graph structure

        Returns
        -------
        Tensor
            Updated mesh edge features, world edge features, and node features
        """
        # Process mesh edges
        mesh_efeat_new, world_efeat_new = self.edge_mlp(
            mesh_efeat, world_efeat, nfeat, graph
        )
        mesh_efeat_new = mesh_efeat_new + mesh_efeat
        world_efeat_new = world_efeat_new + world_efeat
        return mesh_efeat_new, world_efeat_new, nfeat
