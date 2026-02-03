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

import torch
from torch import Tensor

import physicsnemo.models.meshgraphnet.meshgraphnet as mgn

from physicsnemo.core import ModelMetaData
from physicsnemo.nn.module.gnn_layers.utils import GraphType
from physicsnemo.nn import get_activation


@dataclass
class MetaData(ModelMetaData):
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


class AeroGraphNet(mgn.MeshGraphNet):
    """A variant of MeshGraphNet model that also predicts a drag coefficient.

    This model is based on a standard PhysicsNeMo `MeshGraphNet` model
    with additional output, C_d (drag coefficient).
    """

    def __init__(
        self,
        *args,
        hidden_dim_processor: int = 128,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int | None = 2,
        mlp_activation_fn: str | list[str] = "relu",
        recompute_activation: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            mlp_activation_fn=mlp_activation_fn,
            recompute_activation=recompute_activation,
            **kwargs,
        )
        # Update meta.
        self.meta = MetaData()

        self.c_d_decoder = mgn.MeshGraphMLP(
            hidden_dim_processor,
            output_dim=1,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=get_activation(mlp_activation_fn),
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        **kwargs,
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        c_d = torch.relu(self.c_d_decoder(x.mean(dim=0)))
        x = self.node_decoder(x)
        return {"graph": x, "c_d": c_d}
