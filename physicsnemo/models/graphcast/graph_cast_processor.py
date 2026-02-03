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

import importlib

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.nn.module.gnn_layers.mesh_edge_block import MeshEdgeBlock
from physicsnemo.nn.module.gnn_layers.mesh_node_block import MeshNodeBlock
from physicsnemo.nn.module.gnn_layers.utils import GraphType, set_checkpoint_fn

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)


class GraphCastProcessor(nn.Module):
    """Processor block used in GraphCast operating on a latent space
    represented by hierarchy of icosahedral meshes.

    Parameters
    ----------
    aggregation : str, optional
        message passing aggregation method ("sum", "mean"), by default "sum"
    processor_layers : int, optional
        number of processor layers, by default 16
    input_dim_nodes : int, optional
        input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        input dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    do_conat_trick: : bool, default=False
        whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()

        edge_block_invars = (
            input_dim_nodes,
            input_dim_edges,
            input_dim_edges,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            do_concat_trick,
            recompute_activation,
        )
        node_block_invars = (
            aggregation,
            input_dim_nodes,
            input_dim_edges,
            input_dim_nodes,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            recompute_activation,
        )

        layers = []
        for _ in range(processor_layers):
            layers.append(MeshEdgeBlock(*edge_block_invars))
            layers.append(MeshNodeBlock(*node_block_invars))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        # per default, no checkpointing
        # one segment for compatability
        self.checkpoint_segments = [(0, self.num_processor_layers)]
        self.checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        """
        Set the number of checkpoint segments

        Parameters
        ----------
        checkpoint_segments : int
            number of checkpoint segments

        Raises
        ------
        ValueError
            if the number of processor layers is not a multiple of the number of
            checkpoint segments
        """
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))

            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(self, segment_start: int, segment_end: int):
        """Custom forward for gradient checkpointing

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment
        segment_end : int
            Layer index as end of the segment

        Returns
        -------
        function
            Custom forward function
        """
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(efeat, nfeat, graph):
            """Custom forward function"""
            for module in segment:
                efeat, nfeat = module(efeat, nfeat, graph)
            return efeat, nfeat

        return custom_forward

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: GraphType,
    ) -> Tensor:
        for segment_start, segment_end in self.checkpoint_segments:
            efeat, nfeat = self.checkpoint_fn(
                self.run_function(segment_start, segment_end),
                efeat,
                nfeat,
                graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        return efeat, nfeat


if TE_AVAILABLE:
    te = importlib.import_module("transformer_engine")

    class GraphCastProcessorGraphTransformer(nn.Module):
        """Processor block used in GenCast operating on a latent space
        represented by hierarchy of icosahedral meshes.

        Parameters
        ----------
        attn_mask : torch.Tensor
            Attention mask to be applied within the transformer layers.
        processor_layers : int, optional (default=16)
            Number of processing layers.
        input_dim_nodes : int, optional (default=512)
            Dimension of the input features for each node.
        hidden_dim : int, optional (default=512)
            Dimension of the hidden features within the transformer layers.
        """

        def __init__(
            self,
            attention_mask: torch.Tensor,
            num_attention_heads: int = 4,
            processor_layers: int = 16,
            input_dim_nodes: int = 512,
            hidden_dim: int = 512,
        ):
            super().__init__()
            self.num_attention_heads = num_attention_heads
            self.hidden_dim = hidden_dim
            self.attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
            self.register_buffer("mask", self.attention_mask, persistent=False)

            layers = [
                te.pytorch.TransformerLayer(
                    hidden_size=input_dim_nodes,
                    ffn_hidden_size=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    layer_number=i + 1,
                    fuse_qkv_params=False,
                )
                for i in range(processor_layers)
            ]
            self.processor_layers = nn.ModuleList(layers)

        def forward(
            self,
            nfeat: Tensor,
        ) -> Tensor:
            nfeat = nfeat.unsqueeze(1)
            # TODO make sure reshaping the last dim to (h, d) is done automatically in the transformer layer
            for module in self.processor_layers:
                nfeat = module(
                    nfeat,
                    attention_mask=self.mask,
                    self_attn_mask_type="arbitrary",
                )

            return torch.squeeze(nfeat, 1)

else:

    class GraphCastProcessorGraphTransformer(nn.Module):
        """Dummy class for when transformer engine is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GraphCastProcessorGraphTransformer: transformer engine is not installed, can not be used as a backend for a graph transformer.\n"
                "Please install transformer engine with: pip install transformer-engine"
            )
