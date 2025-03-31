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

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

try:
    from typing import Self
except ImportError:
    # for Python versions < 3.11
    from typing_extensions import Self

from physicsnemo.models.gnn_layers.embedder import (
    GraphCastDecoderEmbedder,
    GraphCastEncoderEmbedder,
)
from physicsnemo.models.gnn_layers.mesh_graph_decoder import MeshGraphDecoder
from physicsnemo.models.gnn_layers.mesh_graph_encoder import MeshGraphEncoder
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.models.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from physicsnemo.models.layers import get_activation
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from physicsnemo.utils.graphcast.graph import Graph

from .graph_cast_processor import (
    GraphCastProcessor,
    GraphCastProcessorGraphTransformer,
)

logger = logging.getLogger(__name__)


def get_lat_lon_partition_separators(partition_size: int):
    """Utility Function to get separation intervals for lat-lon
    grid for partition_sizes of interest.

    Parameters
    ----------
    partition_size : int
        size of graph partition
    """

    def _divide(num_lat_chunks: int, num_lon_chunks: int):
        # divide lat-lon grid into equally-sizes chunks along both latitude and longitude
        if (num_lon_chunks * num_lat_chunks) != partition_size:
            raise ValueError(
                "Can't divide lat-lon grid into grid {num_lat_chunks} x {num_lon_chunks} chunks for partition_size={partition_size}."
            )
        # divide latitutude into num_lat_chunks of size 180 / num_lat_chunks
        # divide longitude into chunks of size 360 / (partition_size / num_lat_chunks)
        lat_bin_width = 180.0 / num_lat_chunks
        lon_bin_width = 360.0 / num_lon_chunks

        lat_ranges = []
        lon_ranges = []

        for p_lat in range(num_lat_chunks):
            for p_lon in range(num_lon_chunks):
                lat_ranges += [
                    (lat_bin_width * p_lat - 90.0, lat_bin_width * (p_lat + 1) - 90.0)
                ]
                lon_ranges += [
                    (lon_bin_width * p_lon - 180.0, lon_bin_width * (p_lon + 1) - 180.0)
                ]

        lat_ranges[-1] = (lat_ranges[-1][0], None)
        lon_ranges[-1] = (lon_ranges[-1][0], None)

        return lat_ranges, lon_ranges

    # use two closest factors of partition_size
    lat_chunks, lon_chunks, i = 1, partition_size, 0
    while lat_chunks < lon_chunks:
        i += 1
        if partition_size % i == 0:
            lat_chunks = i
            lon_chunks = partition_size // lat_chunks

    lat_ranges, lon_ranges = _divide(lat_chunks, lon_chunks)

    # mainly for debugging
    if (lat_ranges is None) or (lon_ranges is None):
        raise ValueError("unexpected error, abort")

    min_seps = []
    max_seps = []

    for i in range(partition_size):
        lat = lat_ranges[i]
        lon = lon_ranges[i]
        min_seps.append([lat[0], lon[0]])
        max_seps.append([lat[1], lon[1]])

    return min_seps, max_seps


@dataclass
class MetaData(ModelMetaData):
    name: str = "GraphCastNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class GraphCastNet(Module):
    """GraphCast network architecture

    Parameters
    ----------
    multimesh_level: int, optional
        Level of the latent mesh, by default 6
    multimesh: bool, optional
        If the latent mesh is a multimesh, by default True
        If True, the latent mesh includes the nodes corresponding
        to the specified `mesh_level`and incorporates the edges from
        all mesh levels ranging from level 0 up to and including `mesh_level`.
    input_res: Tuple[int, int]
        Input resolution of the latitude-longitude grid
    input_dim_grid_nodes : int, optional
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim_grid_nodes : int, optional
        Final output dimensionality of the edge features, by default 227
    processor_type: str, optional
        The type of processor used in this model. Available options are
        'MessagePassing', and 'GraphTransformer', which correspond to the
        processors in GraphCast and GenCast, respectively.
        By default 'MessagePassing'.
    khop_neighbors: int, optional
        Number of khop neighbors used in the GraphTransformer.
        This option is ignored if 'MessagePassing' processor is used.
        By default 0.
    processor_layers : int, optional
        Number of processor layers, by default 16
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    activation_fn : str, optional
        Type of activation function, by default "silu"
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    use_cugraphops_encoder : bool, default=False
        Flag to select cugraphops kernels in encoder
    use_cugraphops_processor : bool, default=False
        Flag to select cugraphops kernels in the processor
    use_cugraphops_decoder : bool, default=False
        Flag to select cugraphops kernels in the decoder
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    partition_size : int, default=1
        Number of process groups across which graphs are distributed. If equal to 1,
        the model is run in a normal Single-GPU configuration.
    partition_group_name : str, default=None
        Name of process group across which graphs are distributed. If partition_size
        is set to 1, the model is run in a normal Single-GPU configuration and the
        specification of a process group is not necessary. If partitition_size > 1,
        passing no process group name leads to a parallelism across the default
        process group. Otherwise, the group size of a process group is expected
        to match partition_size.
    use_lat_lon_partitioning : bool, default=False
        flag to specify whether all graphs (grid-to-mesh, mesh, mesh-to-grid)
        are partitioned based on lat-lon-coordinates of nodes or based on IDs.
    expect_partitioned_input : bool, default=False
        Flag indicating whether the model expects the input to be already
        partitioned. This can be helpful e.g. in multi-step rollouts to avoid
        aggregating the output just to distribute it in the next step again.
    global_features_on_rank_0 : bool, default=False
        Flag indicating whether the model expects the input to be present
        in its "global" form only on group_rank 0. During the input preparation phase,
        the model will take care of scattering the input accordingly onto all ranks
        of the process group across which the graph is partitioned. Note that only either
        this flag or expect_partitioned_input can be set at a time.
    produce_aggregated_output : bool, default=True
        Flag indicating whether the model produces the aggregated output on each
        rank of the procress group across which the graph is distributed or
        whether the output is kept distributed. This can be helpful e.g.
        in multi-step rollouts to avoid aggregating the output just to distribute
        it in the next step again.
    produce_aggregated_output_on_all_ranks : bool, default=True
        Flag indicating - if produce_aggregated_output is True - whether the model
        produces the aggregated output on each rank of the process group across
        which the group is distributed or only on group_rank 0. This can be helpful
        for computing the loss using global targets only on a single rank which can
        avoid either having to distribute the computation of a loss function.

    Note
    ----
    Based on these papers:
    - "GraphCast: Learning skillful medium-range global weather forecasting"
        https://arxiv.org/abs/2212.12794
    - "Forecasting Global Weather with Graph Neural Networks"
        https://arxiv.org/abs/2202.07575
    - "Learning Mesh-Based Simulation with Graph Networks"
        https://arxiv.org/abs/2010.03409
    - "MultiScale MeshGraphNets"
        https://arxiv.org/abs/2210.00612
    - "GenCast: Diffusion-based ensemble forecasting for medium-range weather"
        https://arxiv.org/abs/2312.15796
    """

    def __init__(
        self,
        mesh_level: Optional[int] = 6,
        multimesh_level: Optional[int] = None,
        multimesh: bool = True,
        input_res: tuple = (721, 1440),
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 227,
        processor_type: str = "MessagePassing",
        khop_neighbors: int = 32,
        num_attention_heads: int = 4,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        use_cugraphops_encoder: bool = False,
        use_cugraphops_processor: bool = False,
        use_cugraphops_decoder: bool = False,
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        use_lat_lon_partitioning: bool = False,
        expect_partitioned_input: bool = False,
        global_features_on_rank_0: bool = False,
        produce_aggregated_output: bool = True,
        produce_aggregated_output_on_all_ranks: bool = True,
    ):
        super().__init__(meta=MetaData())

        # 'multimesh_level' deprecation handling
        if multimesh_level is not None:
            warnings.warn(
                "'multimesh_level' is deprecated and will be removed in a future version. Use 'mesh_level' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mesh_level = multimesh_level

        self.processor_type = processor_type
        if self.processor_type == "MessagePassing":
            khop_neighbors = 0
        self.is_distributed = False
        if partition_size > 1:
            self.is_distributed = True
        self.expect_partitioned_input = expect_partitioned_input
        self.global_features_on_rank_0 = global_features_on_rank_0
        self.produce_aggregated_output = produce_aggregated_output
        self.produce_aggregated_output_on_all_ranks = (
            produce_aggregated_output_on_all_ranks
        )
        self.partition_group_name = partition_group_name

        # create the lat_lon_grid
        self.latitudes = torch.linspace(-90, 90, steps=input_res[0])
        self.longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
        self.lat_lon_grid = torch.stack(
            torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1
        )

        # Set activation function
        activation_fn = get_activation(activation_fn)

        # construct the graph
        self.graph = Graph(self.lat_lon_grid, mesh_level, multimesh, khop_neighbors)

        self.mesh_graph, self.attn_mask = self.graph.create_mesh_graph(verbose=False)
        self.g2m_graph = self.graph.create_g2m_graph(verbose=False)
        self.m2g_graph = self.graph.create_m2g_graph(verbose=False)

        self.g2m_edata = self.g2m_graph.edata["x"]
        self.m2g_edata = self.m2g_graph.edata["x"]
        self.mesh_ndata = self.mesh_graph.ndata["x"]
        if self.processor_type == "MessagePassing":
            self.mesh_edata = self.mesh_graph.edata["x"]
        elif self.processor_type == "GraphTransformer":
            # Dummy tensor to avoid breaking the API
            self.mesh_edata = torch.zeros((1, input_dim_edges))
        else:
            raise ValueError(f"Invalid processor type {processor_type}")

        if use_cugraphops_encoder or self.is_distributed:
            kwargs = {}
            if use_lat_lon_partitioning:
                min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
                kwargs = {
                    "src_coordinates": self.g2m_graph.srcdata["lat_lon"],
                    "dst_coordinates": self.g2m_graph.dstdata["lat_lon"],
                    "coordinate_separators_min": min_seps,
                    "coordinate_separators_max": max_seps,
                }
            self.g2m_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.g2m_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
                partition_by_bbox=use_lat_lon_partitioning,
                **kwargs,
            )
            self.g2m_edata = self.g2m_edata[edge_perm]

            if self.is_distributed:
                self.g2m_edata = self.g2m_graph.get_edge_features_in_partition(
                    self.g2m_edata
                )

        if use_cugraphops_decoder or self.is_distributed:
            kwargs = {}
            if use_lat_lon_partitioning:
                min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
                kwargs = {
                    "src_coordinates": self.m2g_graph.srcdata["lat_lon"],
                    "dst_coordinates": self.m2g_graph.dstdata["lat_lon"],
                    "coordinate_separators_min": min_seps,
                    "coordinate_separators_max": max_seps,
                }

            self.m2g_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.m2g_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
                partition_by_bbox=use_lat_lon_partitioning,
                **kwargs,
            )
            self.m2g_edata = self.m2g_edata[edge_perm]

            if self.is_distributed:
                self.m2g_edata = self.m2g_graph.get_edge_features_in_partition(
                    self.m2g_edata
                )

        if use_cugraphops_processor or self.is_distributed:
            kwargs = {}
            if use_lat_lon_partitioning:
                min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
                kwargs = {
                    "src_coordinates": self.mesh_graph.ndata["lat_lon"],
                    "dst_coordinates": self.mesh_graph.ndata["lat_lon"],
                    "coordinate_separators_min": min_seps,
                    "coordinate_separators_max": max_seps,
                }

            self.mesh_graph, edge_perm = CuGraphCSC.from_dgl(
                graph=self.mesh_graph,
                partition_size=partition_size,
                partition_group_name=partition_group_name,
                partition_by_bbox=use_lat_lon_partitioning,
                **kwargs,
            )
            self.mesh_edata = self.mesh_edata[edge_perm]
            if self.is_distributed:
                self.mesh_edata = self.mesh_graph.get_edge_features_in_partition(
                    self.mesh_edata
                )
                self.mesh_ndata = self.mesh_graph.get_dst_node_features_in_partition(
                    self.mesh_ndata
                )

        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.input_res = input_res

        # by default: don't checkpoint at all
        self.model_checkpoint_fn = set_checkpoint_fn(False)
        self.encoder_checkpoint_fn = set_checkpoint_fn(False)
        self.decoder_checkpoint_fn = set_checkpoint_fn(False)

        # initial feature embedder
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.decoder_embedder = GraphCastDecoderEmbedder(
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # icosahedron processor
        if processor_layers <= 2:
            raise ValueError("Expected at least 3 processor layers")
        if processor_type == "MessagePassing":
            self.processor_encoder = GraphCastProcessor(
                aggregation=aggregation,
                processor_layers=1,
                input_dim_nodes=hidden_dim,
                input_dim_edges=hidden_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
            )
            self.processor = GraphCastProcessor(
                aggregation=aggregation,
                processor_layers=processor_layers - 2,
                input_dim_nodes=hidden_dim,
                input_dim_edges=hidden_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
            )
            self.processor_decoder = GraphCastProcessor(
                aggregation=aggregation,
                processor_layers=1,
                input_dim_nodes=hidden_dim,
                input_dim_edges=hidden_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                do_concat_trick=do_concat_trick,
                recompute_activation=recompute_activation,
            )
        else:
            self.processor_encoder = torch.nn.Identity()
            self.processor = GraphCastProcessorGraphTransformer(
                attention_mask=self.attn_mask,
                num_attention_heads=num_attention_heads,
                processor_layers=processor_layers,
                input_dim_nodes=hidden_dim,
                hidden_dim=hidden_dim,
            )
            self.processor_decoder = torch.nn.Identity()

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def set_checkpoint_model(self, checkpoint_flag: bool):
        """Sets checkpoint function for the entire model.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. In this case, all the other gradient checkpoitings
        will be disabled. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        # force a single checkpoint for the whole model
        self.model_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)
        if checkpoint_flag:
            self.processor.set_checkpoint_segments(-1)
            self.encoder_checkpoint_fn = set_checkpoint_fn(False)
            self.decoder_checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_processor(self, checkpoint_segments: int):
        """Sets checkpoint function for the processor excluding the first and last
        layers.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_segments` flag. If `checkpoint_segments` is positive,
        the function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`, with number of checkpointing segments equal to
        `checkpoint_segments`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_segments : int
            Number of checkpointing segments for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.processor.set_checkpoint_segments(checkpoint_segments)

    def set_checkpoint_encoder(self, checkpoint_flag: bool):
        """Sets checkpoint function for the embedder, encoder, and the first of
        the processor.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.encoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def set_checkpoint_decoder(self, checkpoint_flag: bool):
        """Sets checkpoint function for the last layer of the processor, the decoder,
        and the final MLP.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.decoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def encoder_forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
        """Forward method for the embedder, encoder, and the first of the processor.

        Parameters
        ----------
        grid_nfeat : Tensor
            Node features for the latitude-longitude grid.

        Returns
        -------
        mesh_efeat_processed: Tensor
            Processed edge features for the multimesh.
        mesh_nfeat_processed: Tensor
            Processed node features for the multimesh.
        grid_nfeat_encoded: Tensor
            Encoded node features for the latitude-longitude grid.
        """

        # embedd graph features
        (
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            g2m_efeat_embedded,
            mesh_efeat_embedded,
        ) = self.encoder_embedder(
            grid_nfeat,
            self.mesh_ndata,
            self.g2m_edata,
            self.mesh_edata,
        )

        # encode lat/lon to multimesh
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded,
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            self.g2m_graph,
        )

        # process multimesh graph
        if self.processor_type == "MessagePassing":
            mesh_efeat_processed, mesh_nfeat_processed = self.processor_encoder(
                mesh_efeat_embedded,
                mesh_nfeat_encoded,
                self.mesh_graph,
            )
        else:
            mesh_nfeat_processed = self.processor_encoder(
                mesh_nfeat_encoded,
            )
            mesh_efeat_processed = None
        return mesh_efeat_processed, mesh_nfeat_processed, grid_nfeat_encoded

    def decoder_forward(
        self,
        mesh_efeat_processed: Tensor,
        mesh_nfeat_processed: Tensor,
        grid_nfeat_encoded: Tensor,
    ) -> Tensor:
        """Forward method for the last layer of the processor, the decoder,
        and the final MLP.

        Parameters
        ----------
        mesh_efeat_processed : Tensor
            Multimesh edge features processed by the processor.
        mesh_nfeat_processed : Tensor
            Multi-mesh node features processed by the processor.
        grid_nfeat_encoded : Tensor
            The encoded node features for the latitude-longitude grid.

        Returns
        -------
        grid_nfeat_finale: Tensor
            The final node features for the latitude-longitude grid.
        """

        # process multimesh graph
        if self.processor_type == "MessagePassing":
            _, mesh_nfeat_processed = self.processor_decoder(
                mesh_efeat_processed,
                mesh_nfeat_processed,
                self.mesh_graph,
            )
        else:
            mesh_nfeat_processed = self.processor_decoder(
                mesh_nfeat_processed,
            )

        m2g_efeat_embedded = self.decoder_embedder(self.m2g_edata)

        # decode multimesh to lat/lon
        grid_nfeat_decoded = self.decoder(
            m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph
        )

        # map to the target output dimension
        grid_nfeat_finale = self.finale(
            grid_nfeat_decoded,
        )

        return grid_nfeat_finale

    def custom_forward(self, grid_nfeat: Tensor) -> Tensor:
        """GraphCast forward method with support for gradient checkpointing.

        Parameters
        ----------
        grid_nfeat : Tensor
            Node features of the latitude-longitude graph.

        Returns
        -------
        grid_nfeat_finale: Tensor
            Predicted node features of the latitude-longitude graph.
        """
        (
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
        ) = self.encoder_checkpoint_fn(
            self.encoder_forward,
            grid_nfeat,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        # checkpoint of processor done in processor itself
        if self.processor_type == "MessagePassing":
            mesh_efeat_processed, mesh_nfeat_processed = self.processor(
                mesh_efeat_processed,
                mesh_nfeat_processed,
                self.mesh_graph,
            )
        else:
            mesh_nfeat_processed = self.processor(
                mesh_nfeat_processed,
            )
            mesh_efeat_processed = None

        grid_nfeat_finale = self.decoder_checkpoint_fn(
            self.decoder_forward,
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        return grid_nfeat_finale

    def forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
        invar = self.prepare_input(
            grid_nfeat, self.expect_partitioned_input, self.global_features_on_rank_0
        )
        outvar = self.model_checkpoint_fn(
            self.custom_forward,
            invar,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        outvar = self.prepare_output(
            outvar,
            self.produce_aggregated_output,
            self.produce_aggregated_output_on_all_ranks,
        )
        return outvar

    def prepare_input(
        self,
        invar: Tensor,
        expect_partitioned_input: bool,
        global_features_on_rank_0: bool,
    ) -> Tensor:
        """Prepares the input to the model in the required shape.

        Parameters
        ----------
        invar : Tensor
            Input in the shape [N, C, H, W].

        expect_partitioned_input : bool
            flag indicating whether input is partioned according to graph partitioning scheme

        global_features_on_rank_0 : bool
            Flag indicating whether input is in its "global" form only on group_rank 0 which
            requires a scatter operation beforehand. Note that only either this flag or
            expect_partitioned_input can be set at a time.

        Returns
        -------
        Tensor
            Reshaped input.
        """
        if global_features_on_rank_0 and expect_partitioned_input:
            raise ValueError(
                "global_features_on_rank_0 and expect_partitioned_input cannot be set at the same time."
            )

        if not self.is_distributed:
            if invar.size(0) != 1:
                raise ValueError("GraphCast does not support batch size > 1")
            invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)

        else:
            # is_distributed
            if not expect_partitioned_input:
                # global_features_on_rank_0
                if invar.size(0) != 1:
                    raise ValueError("GraphCast does not support batch size > 1")

                invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)

                # scatter global features
                invar = self.g2m_graph.get_src_node_features_in_partition(
                    invar,
                    scatter_features=global_features_on_rank_0,
                )

        return invar

    def prepare_output(
        self,
        outvar: Tensor,
        produce_aggregated_output: bool,
        produce_aggregated_output_on_all_ranks: bool = True,
    ) -> Tensor:
        """Prepares the output of the model in the shape [N, C, H, W].

        Parameters
        ----------
        outvar : Tensor
            Output of the final MLP of the model.

        produce_aggregated_output : bool
            flag indicating whether output is gathered onto each rank
            or kept distributed

        produce_aggregated_output_on_all_ranks : bool
            flag indicating whether output is gatherered on each rank
            or only gathered at group_rank 0, True by default and
            only valid if produce_aggregated_output is set.

        Returns
        -------
        Tensor
            The reshaped output of the model.
        """
        if produce_aggregated_output or not self.is_distributed:
            # default case: output of shape [N, C, H, W]
            if self.is_distributed:
                outvar = self.m2g_graph.get_global_dst_node_features(
                    outvar,
                    get_on_all_ranks=produce_aggregated_output_on_all_ranks,
                )

            outvar = outvar.permute(1, 0)
            outvar = outvar.view(self.output_dim_grid_nodes, *self.input_res)
            outvar = torch.unsqueeze(outvar, dim=0)

        return outvar

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Moves the object to the specified device, dtype, or format.
        This method moves the object and its underlying graph and graph features to
        the specified device, dtype, or format, and returns the updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        GraphCastNet
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super(GraphCastNet, self).to(*args, **kwargs)

        self.g2m_edata = self.g2m_edata.to(*args, **kwargs)
        self.m2g_edata = self.m2g_edata.to(*args, **kwargs)
        self.mesh_ndata = self.mesh_ndata.to(*args, **kwargs)
        self.mesh_edata = self.mesh_edata.to(*args, **kwargs)

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.g2m_graph = self.g2m_graph.to(device)
        self.mesh_graph = self.mesh_graph.to(device)
        self.m2g_graph = self.m2g_graph.to(device)

        return self
