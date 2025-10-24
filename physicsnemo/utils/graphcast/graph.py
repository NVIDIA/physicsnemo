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

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from physicsnemo.models.gnn_layers.utils import GraphType
from physicsnemo.utils.graphcast.graph_backend import DglGraphBackend, PyGGraphBackend

from .graph_utils import (
    get_face_centroids,
    latlon2xyz,
    max_edge_length,
    xyz2latlon,
)
from .icosahedral_mesh import (
    faces_to_edges,
    get_hierarchy_of_triangular_meshes_for_sphere,
    merge_meshes,
)

logger = logging.getLogger(__name__)


class Graph:
    """Graph class for creating the graph2mesh, latent mesh, and mesh2graph graphs.

    Parameters
    ----------
    lat_lon_grid : Tensor
        Tensor with shape (lat, lon, 2) that includes the latitudes and longitudes
        meshgrid.
    mesh_level: int, optional
        Level of the latent mesh, by default 6
    multimesh: bool, optional
        If the latent mesh is a multimesh, by default True
        If True, the latent mesh includes the nodes corresponding
        to the specified `mesh_level`and incorporates the edges from
        all mesh levels ranging from level 0 up to and including `mesh_level`.
    khop_neighbors: int, optional
        This option is used to retrieve a list of indices for the k-hop neighbors
        of all mesh nodes. It is applicable when a graph transformer is used as the
        processor. If set to 0, this list is not computed. If a message passing
        processor is used, it is forced to 0. By default 0.
    dtype : torch.dtype, optional
        Data type of the graph, by default torch.float
    """

    def __init__(
        self,
        lat_lon_grid: Tensor,
        mesh_level: int = 6,
        multimesh: bool = True,
        khop_neighbors: int = 0,
        dtype=torch.float,
        backend: str = "dgl",
    ) -> None:
        self.khop_neighbors = khop_neighbors
        self.dtype = dtype
        if backend == "dgl":
            self.backend = DglGraphBackend
        elif backend == "pyg":
            self.backend = PyGGraphBackend
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # flatten lat/lon gird
        self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        # create the multi-mesh
        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
        finest_mesh = _meshes[-1]  # get the last one in the list of meshes
        self.finest_mesh_src, self.finest_mesh_dst = faces_to_edges(finest_mesh.faces)
        self.finest_mesh_vertices = np.array(finest_mesh.vertices)
        if multimesh:
            mesh = merge_meshes(_meshes)
            self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
            self.mesh_vertices = np.array(mesh.vertices)
        else:
            mesh = finest_mesh
            self.mesh_src, self.mesh_dst = self.finest_mesh_src, self.finest_mesh_dst
            self.mesh_vertices = self.finest_mesh_vertices
        self.mesh_faces = mesh.faces

    def create_mesh_graph(self, verbose: bool = True) -> GraphType:
        """Create the multimesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        GraphType
            Multimesh graph
        """
        mesh_graph = self.backend.create_graph(
            self.mesh_src,
            self.mesh_dst,
            to_bidirected=True,
            add_self_loop=False,
            dtype=torch.int32,
        )
        mesh_pos = torch.tensor(
            self.mesh_vertices,
            dtype=torch.float32,
        )
        mesh_graph = self.backend.add_edge_features(mesh_graph, mesh_pos)
        mesh_graph = self.backend.add_node_features(mesh_graph, mesh_pos)
        if self.backend.name == "dgl":
            mesh_graph.ndata["lat_lon"] = xyz2latlon(mesh_pos)
            # ensure fields set to dtype to avoid later conversions
            mesh_graph.ndata["x"] = mesh_graph.ndata["x"].to(dtype=self.dtype)
            mesh_graph.edata["x"] = mesh_graph.edata["x"].to(dtype=self.dtype)
        elif self.backend.name == "pyg":
            mesh_graph.lat_lon = xyz2latlon(mesh_pos)
            # ensure fields set to dtype to avoid later conversions
            mesh_graph.x = mesh_graph.x.to(dtype=self.dtype)
            mesh_graph.edge_attr = mesh_graph.edge_attr.to(dtype=self.dtype)

        if self.khop_neighbors > 0:
            # Make a graph whose edges connect the k-hop neighbors of the original graph.
            mask = ~self.backend.khop_adj_all_k(
                graph=mesh_graph, kmax=self.khop_neighbors
            )
        else:
            mask = None
        if verbose:
            print("mesh graph:", mesh_graph)
        return mesh_graph, mask

    def create_g2m_graph(self, verbose: bool = True) -> GraphType:
        """Create the graph2mesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        GraphType
            Graph2mesh graph.
        """
        # get the max edge length of icosphere with max order

        max_edge_len = max_edge_length(
            self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst
        )

        # create the grid2mesh bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(self.mesh_vertices)
        distances, indices = neighbors.kneighbors(cartesian_grid)

        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * max_edge_len:
                    src.append(i)
                    dst.append(indices[i][j])
                    # NOTE this gives 1,618,820 edges, in the paper it is 1,618,746

        g2m_graph = self.backend.create_heterograph(
            src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32
        )
        if self.backend.name == "dgl":
            g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
            g2m_graph.dstdata["pos"] = torch.tensor(
                self.mesh_vertices,
                dtype=torch.float32,
            )
            g2m_graph.srcdata["lat_lon"] = self.lat_lon_grid_flat
            g2m_graph.dstdata["lat_lon"] = xyz2latlon(g2m_graph.dstdata["pos"])

            g2m_graph = self.backend.add_edge_features(
                g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
            )

            # avoid potential conversions at later points
            g2m_graph.srcdata["pos"] = g2m_graph.srcdata["pos"].to(dtype=self.dtype)
            g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
            g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(
                dtype=self.dtype
            )
            g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(
                dtype=self.dtype
            )
            g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)
        elif self.backend.name == "pyg":
            g2m_graph["grid"].pos = cartesian_grid.to(torch.float32)
            g2m_graph["mesh"].pos = torch.tensor(
                self.mesh_vertices,
                dtype=torch.float32,
            )

            g2m_graph["grid"].lat_lon = self.lat_lon_grid_flat
            g2m_graph["mesh"].lat_lon = xyz2latlon(g2m_graph["mesh"].pos)

            g2m_graph = self.backend.add_edge_features(
                g2m_graph, (g2m_graph["grid"].pos, g2m_graph["mesh"].pos)
            )

            g2m_graph["grid"].pos = g2m_graph["grid"].pos.to(dtype=self.dtype)
            g2m_graph["mesh"].pos = g2m_graph["mesh"].pos.to(dtype=self.dtype)

            g2m_graph.edge_attr = g2m_graph.edge_attr.to(dtype=self.dtype)

        if verbose:
            print("g2m graph:", g2m_graph)
        return g2m_graph

    def create_m2g_graph(self, verbose: bool = True) -> GraphType:
        """Create the mesh2grid graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        GraphType
            Mesh2grid graph.
        """
        # create the mesh2grid bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        face_centroids = get_face_centroids(self.mesh_vertices, self.mesh_faces)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(face_centroids)
        _, indices = neighbors.kneighbors(cartesian_grid)
        indices = indices.flatten()

        src = [p for i in indices for p in self.mesh_faces[i]]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]
        m2g_graph = self.backend.create_heterograph(
            src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32
        )  # number of edges is 3,114,720, exactly matches with the paper

        if self.backend.name == "dgl":
            m2g_graph.srcdata["pos"] = torch.tensor(
                self.mesh_vertices,
                dtype=torch.float32,
            )
            m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)

            m2g_graph.srcdata["lat_lon"] = xyz2latlon(m2g_graph.srcdata["pos"])
            m2g_graph.dstdata["lat_lon"] = self.lat_lon_grid_flat

            m2g_graph = self.backend.add_edge_features(
                m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
            )
            # avoid potential conversions at later points
            m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
            m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
            m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(
                dtype=self.dtype
            )
            m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(
                dtype=self.dtype
            )
            m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)
        elif self.backend.name == "pyg":
            m2g_graph["mesh"].pos = torch.tensor(
                self.mesh_vertices,
                dtype=torch.float32,
            )
            m2g_graph["grid"].pos = cartesian_grid.to(dtype=torch.float32)

            m2g_graph["mesh"].lat_lon = xyz2latlon(m2g_graph["mesh"].pos)
            m2g_graph["grid"].lat_lon = self.lat_lon_grid_flat

            m2g_graph = self.backend.add_edge_features(
                m2g_graph, (m2g_graph["mesh"].pos, m2g_graph["grid"].pos)
            )

            m2g_graph["mesh"].pos = m2g_graph["mesh"].pos.to(dtype=self.dtype)
            m2g_graph["grid"].pos = m2g_graph["grid"].pos.to(dtype=self.dtype)

            m2g_graph.edge_attr = m2g_graph.edge_attr.to(dtype=self.dtype)

        if verbose:
            print("m2g graph:", m2g_graph)
        return m2g_graph
