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

"""Graph backend for creating DGL or PyG graphs."""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from torch import Tensor, testing

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.models.graphcast.utils.graph_utils import (
    azimuthal_angle,
    geospatial_rotation,
    polar_angle,
    xyz2latlon,
)
from physicsnemo.nn.module.gnn_layers.utils import GraphType

# Lazy imports for optional dependencies
pyg_data = OptionalImport("torch_geometric.data")
pyg_utils = OptionalImport("torch_geometric.utils")
torch_sparse = OptionalImport("torch_sparse")

# Type aliases for type checking only
if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData
    from torch_sparse import SparseTensor


class PyGGraphBackend:
    """PyG graph backend."""

    name: str = "pyg"

    @staticmethod
    def create_graph(
        src: List,
        dst: List,
        to_bidirected: bool,
        add_self_loop: bool,
        dtype: torch.dtype = torch.int64,
    ) -> "PyGData":
        """Create PyG graph.

        dtype is ignored for PyG graph backend since PyG only supports int64 dtype.
        """

        edge_index = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
        if to_bidirected:
            edge_index = pyg_utils.to_undirected(edge_index)
        if add_self_loop:
            edge_index, _ = pyg_utils.add_self_loops(edge_index)

        return pyg_data.Data(edge_index=edge_index)

    @staticmethod
    def create_heterograph(
        src: List,
        dst: List,
        labels: str,
        dtype: torch.dtype = torch.int64,
    ) -> GraphType:
        """Create heterogeneous graph using PyG.

        Parameters
        ----------
        src : List
            List of source nodes
        dst : List
            List of destination nodes
        labels : str
            Label of the edge type
        dtype : torch.dtype, optional
            Graph index data type, ignored for PyG graph backend since PyG only supports int64 dtype.

        Returns
        -------
        GraphType
            Heterogeneous graph object
        """

        g = pyg_data.HeteroData()
        g[labels].edge_index = torch.stack(
            [torch.tensor(src), torch.tensor(dst)], dim=0
        ).long()

        return g

    @staticmethod
    def add_edge_features(
        graph: "PyGData",
        pos: Union[Tensor, Tuple[Tensor, Tensor]],
        normalize: bool = True,
    ) -> "PyGData":
        """Add edge features to PyG graph."""

        if isinstance(pos, tuple):
            src_pos, dst_pos = pos
        else:
            src_pos = dst_pos = pos

        if isinstance(graph, pyg_data.Data):
            src, dst = graph.edge_index
        elif isinstance(graph, pyg_data.HeteroData):
            src, dst = graph[graph.edge_types[0]].edge_index
        else:
            raise ValueError(f"Invalid graph type: {type(graph)}")

        src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
        dst_latlon = xyz2latlon(dst_pos, unit="rad")
        dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

        # Azimuthal & polar rotation (same logic as DGL version)
        theta_azimuthal = azimuthal_angle(dst_lon)
        theta_polar = polar_angle(dst_lat)

        src_pos = geospatial_rotation(
            src_pos, theta=theta_azimuthal, axis="z", unit="rad"
        )
        dst_pos = geospatial_rotation(
            dst_pos, theta=theta_azimuthal, axis="z", unit="rad"
        )

        # Validation checks
        try:
            testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
        except ValueError:
            raise ValueError(
                "Invalid projection of edge nodes to local coordinate system"
            )

        src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
        dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")

        # More validation checks
        try:
            testing.assert_close(dst_pos[:, 0], torch.ones_like(dst_pos[:, 0]))
            testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
            testing.assert_close(dst_pos[:, 2], torch.zeros_like(dst_pos[:, 2]))
        except ValueError:
            raise ValueError(
                "Invalid projection of edge nodes to local coordinate system"
            )

        # Prepare edge features
        disp = src_pos - dst_pos
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

        if normalize:
            max_disp_norm = torch.max(disp_norm)
            graph.edge_attr = torch.cat(
                (disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1
            )
        else:
            graph.edge_attr = torch.cat((disp, disp_norm), dim=-1)

        return graph

    @staticmethod
    def add_node_features(graph: "PyGData", pos: Tensor) -> "PyGData":
        """Add node features to PyG graph."""

        latlon = xyz2latlon(pos)
        lat, lon = latlon[:, 0], latlon[:, 1]
        graph.x = torch.stack((torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1)
        return graph

    @staticmethod
    def khop_adj_all_k(graph: "PyGData", kmax: int):
        """Construct the union of k-hop adjacencies up to distance `kmax` for a graph."""

        if not isinstance(graph, PyGData):
            raise ValueError(
                f"Invalid graph type: {type(graph)}, only Data type is supported."
            )

        if graph.edge_index is None:
            raise ValueError("Graph must have edge_index defined.")

        n_nodes = graph.num_nodes

        # Build SparseTensor adjacency: shape [n_nodes, n_nodes]
        # row = source, col = target
        adj = SparseTensor.from_edge_index(
            graph.edge_index, sparse_sizes=(n_nodes, n_nodes)
        )

        adj_k = adj.clone()
        adj_all = adj.clone()

        for _ in range(2, kmax + 1):
            adj_k = adj @ adj_k
            adj_all = adj_all + adj_k

        return adj_all.to_dense().bool()
