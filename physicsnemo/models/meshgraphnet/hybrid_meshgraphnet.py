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

import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import torch.nn as nn
from torch import Tensor

try:
    import dgl  # noqa: F401 for docs

    warnings.warn(
        "DGL version of MeshGraphNet will soon be deprecated. "
        "Please use PyG version instead.",
        DeprecationWarning,
    )
except ImportError:
    warnings.warn(
        "Note: This only applies if you're using DGL.\n"
        "MeshGraphNet (DGL version) requires the DGL library.\n"
        "Install it with your preferred CUDA version from:\n"
        "https://www.dgl.ai/pages/start.html\n"
    )

try:
    import torch_scatter  # noqa: F401
except ImportError:
    warnings.warn(
        "MeshGraphNet will soon require PyTorch Geometric and torch_scatter.\n"
        "Install it from here:\n"
        "https://github.com/rusty1s/pytorch_scatter\n"
    )

from itertools import chain

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.gnn_layers.mesh_edge_block import HybridMeshEdgeBlock
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.models.gnn_layers.mesh_node_block import HybridMeshNodeBlock
from physicsnemo.models.gnn_layers.utils import GraphType
from physicsnemo.models.layers import get_activation
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.utils.profiling import profile

# Import the MeshGraphNet
from .meshgraphnet import MeshGraphNet, MeshGraphNetProcessor


@dataclass
class HybridMetaData(ModelMetaData):
    """Metadata for HybridMeshGraphNet"""

    name: str = "HybridMeshGraphNet"
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
    """Hybrid MeshGraphNet with separate mesh and world edge encoders

    This class extends the vanilla MeshGraphNet to support hybrid functionality
    with separate encoders for mesh edges and world edges.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features (applies to both mesh and world edges)
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
        Number of MLP layers for the node feature encoder, by default 2
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoders, by default 128
    num_layers_edge_encoder : Union[int, None], optional
        Number of MLP layers for the edge feature encoders, by default 2
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : Union[int, None], optional
        Number of MLP layers for the node feature decoder, by default 2
    aggregation: str, optional
        Message aggregation type, by default "sum"
    do_concat_trick: bool, optional
        Whether to replace concat+MLP with MLP+idx+sum, by default False
    num_processor_checkpoint_segments: int, optional
        Number of processor segments for gradient checkpointing, by default 0
    checkpoint_offloading: bool, optional
        Whether to offload checkpointing to CPU, by default False
    recompute_activation: bool, optional
        Whether to recompute activations, by default False
    norm_type: str, optional
        Normalization type, by default "LayerNorm"

    Example
    -------
    .. code-block:: python

        # Create model
        model = HybridMeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=2)

        # Forward pass requires:
        # - node_features: (num_nodes, 4)
        # - mesh_edge_features: (num_mesh_edges, 3)
        # - world_edge_features: (num_world_edges, 3)
        # - graph: DGL graph containing both edge types

    Note
    ----
    The HybridMeshGraphNet requires separate feature tensors for mesh edges and world edges,
    allowing for different processing pipelines for different edge types.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: Union[str, List[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        norm_type="LayerNorm",
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
        node_features: Tensor,
        mesh_edge_features: Tensor,
        world_edge_features: Tensor,
        graph: GraphType,
        **kwargs,
    ) -> Tensor:
        """Forward pass for hybrid MeshGraphNet"""
        mesh_edge_features = self.mesh_edge_encoder(mesh_edge_features)
        world_edge_features = self.world_edge_encoder(world_edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(
            node_features, mesh_edge_features, world_edge_features, graph
        )
        x = self.node_decoder(x)
        return x


class HybridMeshGraphNetProcessor(MeshGraphNetProcessor):
    """Hybrid MeshGraphNet processor that extends the original to handle both mesh and world edges"""

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
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
    def run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[[Tensor, Tensor, Tensor, GraphType], Tuple[Tensor, Tensor, Tensor]]:
        """Custom forward for gradient checkpointing - overridden for hybrid functionality"""
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            mesh_edge_features: Tensor,
            world_edge_features: Tensor,
            graph: GraphType,
        ) -> Tuple[Tensor, Tensor, Tensor]:
            """Custom forward function"""
            for module in segment:
                mesh_edge_features, world_edge_features, node_features = module(
                    mesh_edge_features, world_edge_features, node_features, graph
                )
            return mesh_edge_features, world_edge_features, node_features

        return custom_forward

    @profile
    def forward(
        self,
        node_features: Tensor,
        mesh_edge_features: Tensor,
        world_edge_features: Tensor,
        graph: GraphType,
    ) -> Tensor:
        """Forward pass overridden for hybrid functionality"""
        with self.checkpoint_offload_ctx:
            for segment_start, segment_end in self.checkpoint_segments:
                mesh_edge_features, world_edge_features, node_features = (
                    self.checkpoint_fn(
                        self.run_function(segment_start, segment_end),
                        node_features,
                        mesh_edge_features,
                        world_edge_features,
                        graph,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )

        return node_features
