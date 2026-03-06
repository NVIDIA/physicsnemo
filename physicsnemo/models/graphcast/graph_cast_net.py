# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
from dataclasses import dataclass
from typing import Any, Literal, Optional, Self, Tuple

import torch
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.graphcast.utils.graph import Graph
from physicsnemo.nn import get_activation
from physicsnemo.nn.module.gnn_layers.embedder import (
    GraphCastDecoderEmbedder,
    GraphCastEncoderEmbedder,
)
from physicsnemo.nn.module.gnn_layers.mesh_graph_decoder import MeshGraphDecoder
from physicsnemo.nn.module.gnn_layers.mesh_graph_encoder import MeshGraphEncoder
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.gnn_layers.utils import set_checkpoint_fn

from .graph_cast_processor import (
    GraphCastProcessor,
    GraphCastProcessorGraphTransformer,
)

logger = logging.getLogger(__name__)


def get_lat_lon_partition_separators(
    partition_size: int,
) -> Tuple[list[list[float | None]], list[list[float | None]]]:
    r"""
    Compute separation intervals for lat-lon grid partitioning.

    Parameters
    ----------
    partition_size : int
        Size of the graph partition.

    Returns
    -------
    Tuple[list[list[float | None]], list[list[float | None]]]
        The ``(min_seps, max_seps)`` coordinate separators for each partition.
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
    r"""
    GraphCast network architecture for global weather forecasting on an icosahedral mesh graph.

    Parameters
    ----------
    mesh_level : int, optional, default=6
        Level of the latent mesh used to build the graph.
    multimesh : bool, optional, default=True
        If ``True``, the latent mesh includes nodes from all mesh levels up to
        and including ``mesh_level``.
    input_res : Tuple[int, int], optional, default=(721, 1440)
        Resolution of the latitude-longitude grid ``(H, W)``.
    input_dim_grid_nodes : int, optional, default=474
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional, default=3
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional, default=4
        Input dimensionality of the edge features, by default 4
    output_dim_grid_nodes : int, optional, default=227
        Output dimensionality of the grid node features, by default 227
    processor_type : Literal["MessagePassing", "GraphTransformer"], optional, default="MessagePassing"
        Processor type used for the latent mesh. ``"GraphTransformer"`` uses
        :class:`~physicsnemo.models.graphcast.graph_cast_processor.GraphCastProcessorGraphTransformer`.
    khop_neighbors : int, optional, default=32
        Number of k-hop neighbors used in the graph transformer processor. Ignored
        when ``processor_type="MessagePassing"``. Defaults to 32
    num_attention_heads : int, optional, default=4
        Number of attention heads for the graph transformer processor. Defaults to 4
    processor_layers : int, optional, default=16
        Number of processor layers. Defaults to 16
    hidden_layers : int, optional, default=1
        Number of hidden layers in MLP blocks. Defaults to 1
    hidden_dim : int, optional, default=512
        Hidden dimension for node and edge embeddings. Defaults to 512
    aggregation : Literal["sum", "mean"], optional, default="sum"
        Message passing aggregation method. Defaults to "sum"
    activation_fn : str, optional, default="silu"
        Activation function name passed to :func:`~physicsnemo.nn.get_activation`. Defaults to "silu"
    norm_type : Literal["TELayerNorm", "LayerNorm"], optional, default="LayerNorm"
        Normalization type. ``"TELayerNorm"`` is recommended when supported. Defaults to "LayerNorm"
    use_cugraphops_encoder : bool, optional, default=False
        Deprecated flag for cugraphops encoder kernels (not supported). Defaults to False
    use_cugraphops_processor : bool, optional, default=False
        Deprecated flag for cugraphops processor kernels (not supported). Defaults to False
    use_cugraphops_decoder : bool, optional, default=False
        Deprecated flag for cugraphops decoder kernels (not supported). Defaults to False
    do_concat_trick : bool, optional, default=False
        Whether to replace concat+MLP with MLP+index+sum. Defaults to False
    recompute_activation : bool, optional, default=False
        Whether to recompute activations during backward to save memory. Defaults to False
    partition_size : int, optional, default=1
        Number of process groups across which graphs are distributed. If ``1``,
        the model runs in single-GPU mode. Defaults to 1
    partition_group_name : str | None, optional, default=None
        Name of the process group across which graphs are distributed. Defaults to None
    use_lat_lon_partitioning : bool, optional, default=False
        If ``True``, graph partitions are based on lat-lon coordinates instead of IDs. Defaults to False
    expect_partitioned_input : bool, optional, default=False
        If ``True``, the input is already partitioned. Defaults to False
    global_features_on_rank_0 : bool, optional, default=False
        If ``True``, global input features are only provided on rank 0 and are scattered. Defaults to False
    produce_aggregated_output : bool, optional, default=True
        Whether to gather outputs to a global tensor. Defaults to True
    produce_aggregated_output_on_all_ranks : bool, optional, default=True
        If ``produce_aggregated_output`` is ``True``, gather on all ranks or only rank 0. Defaults to True
    graph_backend : Literal["dgl", "pyg"], optional, default="pyg"
        Legacy argument to select the backend used to build the graphs.
        Defaults to "pyg"; "dgl" option is deprecated.

    Forward
    -------
    grid_nfeat : torch.Tensor
        Input grid features of shape :math:`(B, C_{in}, H, W)` where
        :math:`B=1`, :math:`C_{in} =` ``input_dim_grid_nodes``, and
        :math:`(H, W) =` ``input_res``.

    Outputs
    -------
    torch.Tensor
        Output grid features of shape :math:`(B, C_{out}, H, W)` where
        :math:`C_{out} =` ``output_dim_grid_nodes``.

    Notes
    -----
    This implementation follows the GraphCast and GenCast architectures; see
    `GraphCast <https://arxiv.org/abs/2212.12794>`_ and
    `GenCast <https://arxiv.org/abs/2312.15796>`_. The graph transformer processor
    requires ``transformer-engine`` to be installed.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model = GraphCastNet(
    ...     mesh_level=1,
    ...     input_res=(10, 20),
    ...     input_dim_grid_nodes=2,
    ...     input_dim_mesh_nodes=3,
    ...     input_dim_edges=4,
    ...     output_dim_grid_nodes=2,
    ...     processor_layers=3,
    ...     hidden_dim=4,
    ...     do_concat_trick=True,
    ... ).to(device)
    >>> x = torch.randn(1, 2, 10, 20, device=device)
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 2, 10, 20])
    """

    def __init__(
        self,
        mesh_level: Optional[int] = 6,
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
        graph_backend: Literal["dgl", "pyg"] = "pyg",
    ):
        super().__init__(meta=MetaData())

        # Disable cugraphops paths:
        if use_cugraphops_encoder or use_cugraphops_processor or use_cugraphops_decoder:
            raise ImportError(
                "cugraphops is deprecated and not supported for GraphCastNet."
            )

        self.processor_type = processor_type
        if self.processor_type == "MessagePassing":
            khop_neighbors = 0
        self.is_distributed = False
        if partition_size > 1:
            self.is_distributed = True
            if graph_backend == "pyg":
                raise NotImplementedError(
                    "Distributed mode (partition_size > 1) is not supported with PyG backend. "
                    "Distributed functionality was only available with the deprecated DGL/cugraphops backend."
                )
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
        self.graph = Graph(
            self.lat_lon_grid,
            mesh_level,
            multimesh,
            khop_neighbors,
            backend=graph_backend,
        )

        self.mesh_graph, self.attn_mask = self.graph.create_mesh_graph(verbose=False)
        self.g2m_graph = self.graph.create_g2m_graph(verbose=False)
        self.m2g_graph = self.graph.create_m2g_graph(verbose=False)
        self.graph_backend = graph_backend

        # Handle data access based on backend

        if graph_backend == "pyg":
            self.g2m_edata = self.g2m_graph.edge_attr
            self.m2g_edata = self.m2g_graph.edge_attr
            self.mesh_ndata = self.mesh_graph.x
            if self.processor_type == "MessagePassing":
                self.mesh_edata = self.mesh_graph.edge_attr
            elif self.processor_type == "GraphTransformer":
                # Dummy tensor to avoid breaking the API
                self.mesh_edata = torch.zeros((1, input_dim_edges))
            else:
                raise ValueError(f"Invalid processor type {processor_type}")
        else:
            raise ValueError(f"Unsupported graph backend: {graph_backend}")

        # This is all deprecated and to-be-removed.

        # if use_cugraphops_encoder or self.is_distributed:
        #     kwargs = {}
        #     if use_lat_lon_partitioning:
        #         min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
        #         kwargs = {
        #             "src_coordinates": self.g2m_graph.srcdata["lat_lon"],
        #             "dst_coordinates": self.g2m_graph.dstdata["lat_lon"],
        #             "coordinate_separators_min": min_seps,
        #             "coordinate_separators_max": max_seps,
        #         }
        #     self.g2m_graph, edge_perm = CuGraphCSC.from_dgl(
        #         graph=self.g2m_graph,
        #         partition_size=partition_size,
        #         partition_group_name=partition_group_name,
        #         partition_by_bbox=use_lat_lon_partitioning,
        #         **kwargs,
        #     )
        #     self.g2m_edata = self.g2m_edata[edge_perm]

        #     if self.is_distributed:
        #         self.g2m_edata = self.g2m_graph.get_edge_features_in_partition(
        #             self.g2m_edata
        #         )

        # if use_cugraphops_decoder or self.is_distributed:
        #     kwargs = {}
        #     if use_lat_lon_partitioning:
        #         min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
        #         kwargs = {
        #             "src_coordinates": self.m2g_graph.srcdata["lat_lon"],
        #             "dst_coordinates": self.m2g_graph.dstdata["lat_lon"],
        #             "coordinate_separators_min": min_seps,
        #             "coordinate_separators_max": max_seps,
        #         }

        #     self.m2g_graph, edge_perm = CuGraphCSC.from_dgl(
        #         graph=self.m2g_graph,
        #         partition_size=partition_size,
        #         partition_group_name=partition_group_name,
        #         partition_by_bbox=use_lat_lon_partitioning,
        #         **kwargs,
        #     )
        #     self.m2g_edata = self.m2g_edata[edge_perm]

        #     if self.is_distributed:
        #         self.m2g_edata = self.m2g_graph.get_edge_features_in_partition(
        #             self.m2g_edata
        #         )

        # if use_cugraphops_processor or self.is_distributed:
        #     kwargs = {}
        #     if use_lat_lon_partitioning:
        #         min_seps, max_seps = get_lat_lon_partition_separators(partition_size)
        #         kwargs = {
        #             "src_coordinates": self.mesh_graph.ndata["lat_lon"],
        #             "dst_coordinates": self.mesh_graph.ndata["lat_lon"],
        #             "coordinate_separators_min": min_seps,
        #             "coordinate_separators_max": max_seps,
        #         }

        #     self.mesh_graph, edge_perm = CuGraphCSC.from_dgl(
        #         graph=self.mesh_graph,
        #         partition_size=partition_size,
        #         partition_group_name=partition_group_name,
        #         partition_by_bbox=use_lat_lon_partitioning,
        #         **kwargs,
        #     )
        #     self.mesh_edata = self.mesh_edata[edge_perm]
        #     if self.is_distributed:
        #         self.mesh_edata = self.mesh_graph.get_edge_features_in_partition(
        #             self.mesh_edata
        #         )
        #         self.mesh_ndata = self.mesh_graph.get_dst_node_features_in_partition(
        #             self.mesh_ndata
        #         )

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

    def _validate_grid_input(
        self,
        grid_nfeat: (
            Float[torch.Tensor, "batch grid_features height width"]
            | Float[torch.Tensor, "grid_nodes grid_features"]
        ),
        expect_partitioned_input: bool,
    ) -> None:
        if torch.compiler.is_compiling():
            return

        if expect_partitioned_input and self.is_distributed:
            if grid_nfeat.ndim != 2 or grid_nfeat.shape[1] != self.input_dim_grid_nodes:
                raise ValueError(
                    "Expected tensor of shape (N_grid, C_in) but got tensor of shape "
                    f"{tuple(grid_nfeat.shape)}"
                )
            return

        if grid_nfeat.ndim != 4:
            raise ValueError(
                "Expected tensor of shape (B, C_in, H, W) but got tensor of shape "
                f"{tuple(grid_nfeat.shape)}"
            )

        expected_shape = (
            grid_nfeat.shape[0],
            self.input_dim_grid_nodes,
            *self.input_res,
        )
        if tuple(grid_nfeat.shape) != expected_shape:
            raise ValueError(
                "Expected tensor of shape "
                f"{expected_shape} but got tensor of shape {tuple(grid_nfeat.shape)}"
            )

    def _validate_grid_features(
        self, grid_nfeat: Float[torch.Tensor, "grid_nodes grid_features"]
    ) -> None:
        if torch.compiler.is_compiling():
            return

        if grid_nfeat.ndim != 2 or grid_nfeat.shape[1] != self.input_dim_grid_nodes:
            raise ValueError(
                "Expected tensor of shape (N_grid, C_in) but got tensor of shape "
                f"{tuple(grid_nfeat.shape)}"
            )

    def _validate_grid_outputs(
        self, outvar: Float[torch.Tensor, "grid_nodes out_features"]
    ) -> None:
        if torch.compiler.is_compiling():
            return

        if outvar.ndim != 2 or outvar.shape[1] != self.output_dim_grid_nodes:
            raise ValueError(
                "Expected tensor of shape (N_grid, C_out) but got tensor of shape "
                f"{tuple(outvar.shape)}"
            )

    def set_checkpoint_model(self, checkpoint_flag: bool):
        r"""
        Set checkpointing for the entire model.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to enable checkpointing using ``torch.utils.checkpoint``.

        Returns
        -------
        None
            This method updates internal checkpointing settings in-place.
        """
        # force a single checkpoint for the whole model
        self.model_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)
        if checkpoint_flag:
            self.processor.set_checkpoint_segments(-1)
            self.encoder_checkpoint_fn = set_checkpoint_fn(False)
            self.decoder_checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_processor(self, checkpoint_segments: int):
        r"""
        Set checkpointing for the processor interior layers.

        Parameters
        ----------
        checkpoint_segments : int
            Number of checkpoint segments. A positive value enables checkpointing.

        Returns
        -------
        None
            This method updates processor checkpointing settings in-place.
        """
        self.processor.set_checkpoint_segments(checkpoint_segments)

    def set_checkpoint_encoder(self, checkpoint_flag: bool):
        r"""
        Set checkpointing for the encoder path.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to enable checkpointing for the encoder path.

        Returns
        -------
        None
            This method updates encoder checkpointing settings in-place.
        """
        self.encoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def set_checkpoint_decoder(self, checkpoint_flag: bool):
        r"""
        Set checkpointing for the decoder path.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to enable checkpointing for the decoder path.

        Returns
        -------
        None
            This method updates decoder checkpointing settings in-place.
        """
        self.decoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def encoder_forward(
        self,
        grid_nfeat: Float[torch.Tensor, "grid_nodes grid_features"],
    ) -> Tuple[
        Optional[Float[torch.Tensor, "mesh_edges hidden_dim"]],
        Float[torch.Tensor, "mesh_nodes hidden_dim"],
        Float[torch.Tensor, "grid_nodes hidden_dim"],
    ]:
        r"""
        Run the embedder, encoder, and the first processor stage.

        Parameters
        ----------
        grid_nfeat : torch.Tensor
            Grid node features of shape :math:`(N_{grid}, C_{in})`.

        Returns
        -------
        mesh_efeat_processed : torch.Tensor | None
            Processed mesh edge features of shape :math:`(N_{mesh\_edge}, C_{hid})`,
            or ``None`` when using the graph transformer processor.
        mesh_nfeat_processed : torch.Tensor
            Processed mesh node features of shape :math:`(N_{mesh}, C_{hid})`.
        grid_nfeat_encoded : torch.Tensor
            Encoded grid node features of shape :math:`(N_{grid}, C_{hid})`.
        """
        self._validate_grid_features(grid_nfeat)

        # Embed grid/mesh/edge features.
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
        # (N_grid, C_hid), (N_mesh, C_hid), (N_g2m_edge, C_hid), (N_mesh_edge, C_hid)

        # Encode grid features to the multimesh.
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded,
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            self.g2m_graph,
        )
        # (N_grid, C_hid), (N_mesh, C_hid)

        # Process latent mesh features.
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
        mesh_efeat_processed: Optional[Float[torch.Tensor, "mesh_edges hidden_dim"]],
        mesh_nfeat_processed: Float[torch.Tensor, "mesh_nodes hidden_dim"],
        grid_nfeat_encoded: Float[torch.Tensor, "grid_nodes hidden_dim"],
    ) -> Float[torch.Tensor, "grid_nodes out_features"]:
        r"""
        Run the final processor stage, decoder, and output MLP.

        Parameters
        ----------
        mesh_efeat_processed : torch.Tensor | None
            Processed mesh edge features of shape :math:`(N_{mesh\_edge}, C_{hid})`,
            or ``None`` when using the graph transformer processor.
        mesh_nfeat_processed : torch.Tensor
            Processed mesh node features of shape :math:`(N_{mesh}, C_{hid})`.
        grid_nfeat_encoded : torch.Tensor
            Encoded grid node features of shape :math:`(N_{grid}, C_{hid})`.

        Returns
        -------
        torch.Tensor
            Final grid node features of shape :math:`(N_{grid}, C_{out})`.
        """
        if not torch.compiler.is_compiling():
            if mesh_nfeat_processed.ndim != 2:
                raise ValueError(
                    "Expected tensor of shape (N_mesh, C_hid) but got tensor of shape "
                    f"{tuple(mesh_nfeat_processed.shape)}"
                )
            if grid_nfeat_encoded.ndim != 2:
                raise ValueError(
                    "Expected tensor of shape (N_grid, C_hid) but got tensor of shape "
                    f"{tuple(grid_nfeat_encoded.shape)}"
                )
            if self.processor_type == "MessagePassing":
                if mesh_efeat_processed is None or mesh_efeat_processed.ndim != 2:
                    shape = (
                        None
                        if mesh_efeat_processed is None
                        else tuple(mesh_efeat_processed.shape)
                    )
                    raise ValueError(
                        "Expected tensor of shape (N_mesh_edge, C_hid) but got tensor of "
                        f"shape {shape}"
                    )

        # Process latent mesh features.
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
        # (N_m2g_edge, C_hid)

        # Decode multimesh features back to the grid.
        grid_nfeat_decoded = self.decoder(
            m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph
        )
        # (N_grid, C_hid)

        # Map hidden features to output channels.
        grid_nfeat_finale = self.finale(
            grid_nfeat_decoded,
        )
        # (N_grid, C_out)

        return grid_nfeat_finale

    def custom_forward(
        self,
        grid_nfeat: Float[torch.Tensor, "grid_nodes grid_features"],
    ) -> Float[torch.Tensor, "grid_nodes out_features"]:
        r"""
        GraphCast forward method with gradient checkpointing support.

        Parameters
        ----------
        grid_nfeat : torch.Tensor
            Grid node features of shape :math:`(N_{grid}, C_{in})`.

        Returns
        -------
        torch.Tensor
            Output grid node features of shape :math:`(N_{grid}, C_{out})`.
        """
        self._validate_grid_features(grid_nfeat)
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

        # Process latent mesh features (checkpointing handled internally).
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
        grid_nfeat: Float[torch.Tensor, "batch grid_features height width"],
    ) -> Float[torch.Tensor, "batch out_features height width"]:
        r"""
        Run the GraphCast forward pass.

        Parameters
        ----------
        grid_nfeat : torch.Tensor
            Input grid features of shape :math:`(B, C_{in}, H, W)` with
            :math:`B=1` when ``expect_partitioned_input`` is ``False``.

        Returns
        -------
        torch.Tensor
            Output grid features of shape :math:`(B, C_{out}, H, W)`.
        """
        self._validate_grid_input(grid_nfeat, self.expect_partitioned_input)
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
        invar: (
            Float[torch.Tensor, "batch grid_features height width"]
            | Float[torch.Tensor, "grid_nodes grid_features"]
        ),
        expect_partitioned_input: bool,
        global_features_on_rank_0: bool,
    ) -> Float[torch.Tensor, "grid_nodes grid_features"]:
        r"""
        Prepare model input in the required grid-node layout.

        Parameters
        ----------
        invar : torch.Tensor
            Input grid features of shape :math:`(B, C_{in}, H, W)` or partitioned
            features of shape :math:`(N_{grid}, C_{in})`.
        expect_partitioned_input : bool
            Whether ``invar`` is already partitioned.
        global_features_on_rank_0 : bool
            Whether global features are only provided on rank 0 and should be scattered.

        Returns
        -------
        torch.Tensor
            Grid-node features of shape :math:`(N_{grid}, C_{in})`.
        """
        self._validate_grid_input(invar, expect_partitioned_input)
        if global_features_on_rank_0 and expect_partitioned_input:
            raise ValueError(
                "global_features_on_rank_0 and expect_partitioned_input cannot be set at the same time."
            )

        if not self.is_distributed:
            if invar.size(0) != 1:
                raise ValueError(
                    "GraphCast does not support batch size > 1. Expected tensor of shape (1, C_in, H, W) but got tensor of shape "
                    f"{tuple(invar.shape)}"
                )
            # Flatten grid and place features on the last axis. (N_grid, C_in)
            invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)

        else:
            # is_distributed
            if not expect_partitioned_input:
                # global_features_on_rank_0
                if invar.size(0) != 1:
                    raise ValueError(
                        "GraphCast does not support batch size > 1. Expected tensor of shape (1, C_in, H, W) but got tensor of shape "
                        f"{tuple(invar.shape)}"
                    )

                # Flatten global grid features for distribution. (N_grid, C_in)
                invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)

                # scatter global features
                if not hasattr(self.g2m_graph, "get_src_node_features_in_partition"):
                    raise NotImplementedError(
                        f"Distributed mode is not supported with {self.graph_backend} backend. "
                        "The get_src_node_features_in_partition method is only available with "
                        "DistributedGraph objects, which are not created with the PyG backend."
                    )
                invar = self.g2m_graph.get_src_node_features_in_partition(
                    invar,
                    scatter_features=global_features_on_rank_0,
                )

        return invar

    def prepare_output(
        self,
        outvar: Float[torch.Tensor, "grid_nodes out_features"],
        produce_aggregated_output: bool,
        produce_aggregated_output_on_all_ranks: bool = True,
    ) -> (
        Float[torch.Tensor, "batch out_features height width"]
        | Float[torch.Tensor, "grid_nodes out_features"]
    ):
        r"""
        Prepare model output in the required layout.

        Parameters
        ----------
        outvar : torch.Tensor
            Output node features of shape :math:`(N_{grid}, C_{out})`.
        produce_aggregated_output : bool
            Whether to gather outputs to a global tensor.
        produce_aggregated_output_on_all_ranks : bool, optional, default=True
            Whether to gather outputs on all ranks or only rank 0. Defaults to True

        Returns
        -------
        torch.Tensor
            Output features in either global grid format
            :math:`(B, C_{out}, H, W)` or distributed node format
            :math:`(N_{grid}, C_{out})`.
        """
        self._validate_grid_outputs(outvar)
        if produce_aggregated_output or not self.is_distributed:
            # default case: output of shape [N, C, H, W]
            if self.is_distributed:
                if not hasattr(self.m2g_graph, "get_global_dst_node_features"):
                    raise NotImplementedError(
                        f"Distributed mode is not supported with {self.graph_backend} backend. "
                        "The get_global_dst_node_features method is only available with "
                        "DistributedGraph objects, which are not created with the PyG backend."
                    )
                outvar = self.m2g_graph.get_global_dst_node_features(
                    outvar,
                    get_on_all_ranks=produce_aggregated_output_on_all_ranks,
                )

            # Reshape global grid features to (B, C_out, H, W).
            outvar = outvar.permute(1, 0)
            outvar = outvar.view(self.output_dim_grid_nodes, *self.input_res)
            outvar = torch.unsqueeze(outvar, dim=0)

        return outvar

    def to(self, *args: Any, **kwargs: Any) -> Self:
        r"""
        Move the model and its graph buffers to a device or dtype.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to ``torch._C._nn._parse_to``.
        **kwargs : Any
            Keyword arguments passed to ``torch._C._nn._parse_to``.

        Returns
        -------
        GraphCastNet
            The updated model instance.
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
