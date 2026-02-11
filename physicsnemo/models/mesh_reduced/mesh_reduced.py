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

from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import Tensor

from physicsnemo.core.version_check import OptionalImport, require_version_spec
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet
from physicsnemo.nn.module.gnn_layers.graph_types import GraphType  # noqa

# Optional imports for GNN dependencies (lazy, cached, with helpful errors)
_torch_geometric = OptionalImport("torch_geometric")
_torch_cluster = OptionalImport("torch_cluster")
_torch_scatter = OptionalImport("torch_scatter")


class Mesh_Reduced(torch.nn.Module):
    r"""PbGMR-GMUS architecture.
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
    output_encode_dim : int, optional, default=3
        Number of encoding outputs (per pivotal position).
    processor_size : int, optional, default=15
        Number of message passing blocks.
    num_layers_node_processor : int, optional, default=2
        Number of MLP layers for processing nodes in each message passing block.
    num_layers_edge_processor : int, optional, default=2
        Number of MLP layers for processing edge features in each message passing block.
    hidden_dim_processor : int, optional, default=128
        Hidden layer size for the message passing blocks.
    hidden_dim_node_encoder : int, optional, default=128
        Hidden layer size for the node feature encoder.
    num_layers_node_encoder : int, optional, default=2
        Number of MLP layers for the node feature encoder.
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden layer size for the edge feature encoder.
    num_layers_edge_encoder : int, optional, default=2
        Number of MLP layers for the edge feature encoder.
    hidden_dim_node_decoder : int, optional, default=128
        Hidden layer size for the node feature decoder.
    num_layers_node_decoder : int, optional, default=2
        Number of MLP layers for the node feature decoder.
    k : int, optional, default=3
        Number of nearest neighbors for interpolation.
    aggregation : Literal["sum", "mean"], optional, default="mean"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}^{batch}, D_{in}^{node})`.
    edge_features : torch.Tensor
        Input edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
    graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
        Connectivity/topology only. Do not duplicate node or edge features on the graph;
        pass them via ``node_features`` and ``edge_features``. If present on
        the graph, they will be ignored by the model.
        ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
        ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
        The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
        PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
        :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.
    position_mesh : torch.Tensor
        Per-graph reference mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
        These positions are repeated internally across the batch.
    position_pivotal : torch.Tensor
        Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.
        These positions are repeated internally across the batch.

    Returns
    -------
    torch.Tensor
        Decoded node features of shape :math:`(N_{nodes}^{batch}, D_{out}^{decode})`.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced
    >>>
    >>> # Choose a consistent device
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>>
    >>> # Instantiate model
    >>> model = Mesh_Reduced(
    ...     input_dim_nodes=4,
    ...     input_dim_edges=3,
    ...     output_decode_dim=2,
    ... ).to(device)
    >>>
    >>> # Build a simple PyG graph
    >>> # Note: num_nodes must match len(position_mesh) for batch alignment
    >>> num_mesh = 20
    >>> num_nodes, num_edges = num_mesh, 30
    >>> edge_index = torch.randint(0, num_nodes, (2, num_edges))
    >>> graph = Data(edge_index=edge_index, num_nodes=num_nodes).to(device)
    >>> # For a single graph, set a batch vector of zeros
    >>> graph.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    >>>
    >>> # Node/edge features
    >>> node_features = torch.randn(num_nodes, 4, device=device)
    >>> edge_features = torch.randn(num_edges, 3, device=device)
    >>>
    >>> # Per-graph positions (repeated internally across the batch)
    >>> position_mesh = torch.randn(num_mesh, 3, device=device)       # (N_mesh, D_pos)
    >>> position_pivotal = torch.randn(5, 3, device=device)     # (N_pivotal, D_pos)
    >>>
    >>> # Encode to pivotal space, then decode back to mesh space
    >>> enc = model.encode(node_features, edge_features, graph, position_mesh, position_pivotal)
    >>> out = model.decode(enc, edge_features, graph, position_mesh, position_pivotal)
    >>> out.size()
    torch.Size([20, 2])

    Notes
    -----
    Reference: `Predicting physics in mesh-reduced space with temporal attention <https://arxiv.org/pdf/2201.09113>`.
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
        aggregation: Literal["sum", "mean"] = "mean",
    ):
        super().__init__()
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

        # Public constructor attributes for validation/serialization
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_encode_dim = output_encode_dim
        self.output_decode_dim = output_decode_dim

    @require_version_spec("torch_cluster")
    @require_version_spec("torch_scatter")
    def knn_interpolate(
        self,
        x: Tensor,
        pos_x: Tensor,
        pos_y: Tensor,
        batch_x: Tensor | None = None,
        batch_y: Tensor | None = None,
        k: int = 3,
        num_workers: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Perform k-nearest neighbor interpolation from ``pos_x`` to ``pos_y``.

        Parameters
        ----------
        x : torch.Tensor
            Source features of shape :math:`(N_x, D_x)`.
        pos_x : torch.Tensor
            Source positions of shape :math:`(N_x, D_{pos})`.
        pos_y : torch.Tensor
            Target positions of shape :math:`(N_y, D_{pos})`.
        batch_x : torch.Tensor, optional
            Batch indices for ``pos_x`` of shape :math:`(N_x,)`. If provided,
            neighbors are computed per-graph. Default is ``None``.
        batch_y : torch.Tensor, optional
            Batch indices for ``pos_y`` of shape :math:`(N_y,)`. If provided,
            neighbors are computed per-graph. Default is ``None``.
        k : int, optional, default=3
            Number of nearest neighbors.
        num_workers : int, optional, default=1
            Number of workers for the KNN search.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple ``(y, col, row, weights)`` where:

            - ``y``: interpolated features of shape :math:`(N_y, D_x)`
            - ``col``: indices into ``pos_x`` (source) of shape :math:`(k \cdot N_y,)`
            - ``row``: indices into ``pos_y`` (target) of shape :math:`(k \cdot N_y,)`
            - ``weights``: interpolation weights of shape :math:`(k \cdot N_y, 1)`
        """
        with torch.no_grad():
            row, col = _torch_cluster.knn(
                pos_x,
                pos_y,
                k,
                batch_x=batch_x,
                batch_y=batch_y,
                num_workers=num_workers,
            )
            # row: indices in pos_y, col: indices in pos_x
            diff = pos_x[col] - pos_y[row]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y = _torch_scatter.scatter(
            x[col] * weights, row, dim=0, dim_size=pos_y.size(0), reduce="sum"
        )
        y = y / _torch_scatter.scatter(
            weights, row, dim=0, dim_size=pos_y.size(0), reduce="sum"
        )

        return y.float(), col, row, weights

    @require_version_spec("torch_geometric")
    def encode(
        self,
        x: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        position_mesh: Tensor,
        position_pivotal: Tensor,
    ) -> Tensor:
        r"""Encode mesh features to pivotal space.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape :math:`(N_{nodes}^{batch}, D_{in}^{node})`.
        edge_features : torch.Tensor
            Edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
        graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
            PyG graph container with batch information.
        position_mesh : torch.Tensor
            Per-graph reference mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
        position_pivotal : torch.Tensor
            Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.

        Returns
        -------
        torch.Tensor
            Encoded pivotal features of shape :math:`(N_{pivotal}^{batch}, D_{enc})`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D node features (N_nodes, D_in) but got shape {tuple(x.shape)}"
                )
            if edge_features.ndim != 2:
                raise ValueError(
                    f"Expected 2D edge features (N_edges, D_in) but got shape {tuple(edge_features.shape)}"
                )
            if position_mesh.ndim != 2 or position_pivotal.ndim != 2:
                raise ValueError(
                    f"Expected position tensors to be 2D, got {tuple(position_mesh.shape)} and {tuple(position_pivotal.shape)}"
                )
        x = self.encoder_processor(x, edge_features, graph)
        x = self.PivotalNorm(x)
        if isinstance(graph, _torch_geometric.data.Data):
            batch_mesh = graph.batch
            batch_size = (
                int(batch_mesh.max().item()) + 1 if batch_mesh.numel() > 0 else 1
            )
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")

        nodes_index = torch.arange(batch_size, device=x.device)
        position_mesh_batch = position_mesh.repeat(batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * batch_size, device=x.device)
        )

        x, _, _, _ = self.knn_interpolate(
            x=x,
            pos_x=position_mesh_batch,
            pos_y=position_pivotal_batch,
            batch_x=batch_mesh,
            batch_y=batch_pivotal,
            k=self.k,
        )
        return x

    @require_version_spec("torch_geometric")
    def decode(
        self,
        x: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        position_mesh: Tensor,
        position_pivotal: Tensor,
    ) -> Tensor:
        r"""Decode pivotal features back to mesh space.

        Parameters
        ----------
        x : torch.Tensor
            Input features in pivotal space of shape
            :math:`(N_{pivotal}^{batch}, D_{enc})`.
        edge_features : torch.Tensor
            Edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
        graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
            Graph connectivity/topology container (PyG).
            Connectivity/topology only. Do not duplicate node or edge features on the graph;
            pass them via ``node_features`` and ``edge_features``. If present on
            the graph, they will be ignored by the model.
            ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
            ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
            The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
            PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
            :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.
        position_mesh : torch.Tensor
            Per-graph mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
        position_pivotal : torch.Tensor
            Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.

        Returns
        -------
        torch.Tensor
            Decoded features in mesh space of shape
            :math:`(N_{nodes}^{batch}, D_{out}^{decode})`.
        """
        if not torch.compiler.is_compiling():
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
                )
            if position_mesh.ndim != 2 or position_pivotal.ndim != 2:
                raise ValueError(
                    f"Expected position tensors to be 2D, got shapes {tuple(position_mesh.shape)} and {tuple(position_pivotal.shape)}"
                )

        if isinstance(graph, _torch_geometric.data.Data):
            batch_mesh = graph.batch
            batch_size = (
                int(batch_mesh.max().item()) + 1 if batch_mesh.numel() > 0 else 1
            )
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")

        nodes_index = torch.arange(batch_size, device=x.device)
        position_mesh_batch = position_mesh.repeat(batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * batch_size, device=x.device)
        )

        x, _, _, _ = self.knn_interpolate(
            x=x,
            pos_x=position_pivotal_batch,
            pos_y=position_mesh_batch,
            batch_x=batch_pivotal,
            batch_y=batch_mesh,
            k=self.k,
        )

        x = self.decoder_processor(x, edge_features, graph)
        return x
