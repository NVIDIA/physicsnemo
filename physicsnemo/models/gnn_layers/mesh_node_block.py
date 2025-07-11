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

import torch
import torch.nn as nn
from torch import Tensor

from .mesh_graph_mlp import MeshGraphMLP
from .utils import GraphType, aggregate_and_concat


class MeshNodeBlock(nn.Module):
    """Node block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.

    Parameters
    ----------
    aggregation : str, optional
        Aggregation method (sum, mean) , by default "sum"
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the node features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
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
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: GraphType,
    ) -> Tuple[Tensor, Tensor]:
        # update edge features
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_mlp(cat_feat) + nfeat
        return efeat, nfeat_new
