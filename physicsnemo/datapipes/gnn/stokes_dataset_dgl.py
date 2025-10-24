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

import os
import re
from typing import Any, List, Union

import numpy as np
import torch

from .utils import load_json, read_vtp_file, save_json

try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Stokes flow Dataset requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    import vtk
except ImportError:
    raise ImportError(
        "Stokes flow Dataset requires the vtk and pyvista libraries. Install with "
        + "pip install vtk pyvista"
    )


class StokesDataset(DGLDataset):
    """
    In-memory Stokes flow Dataset

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    invar_keys: List[str], optional
        The input node features to consider. Default includes 'pos' and 'marker'
    outvar_keys: List[str], optional
        The output features to consider. Default includes 'u', 'v', and 'p'.
    normalize_keys List[str], optional
        The features to normalize. Default includes 'u', 'v', and 'p'.
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        num_samples=10,
        invar_keys=["pos", "marker"],
        outvar_keys=["u", "v", "p"],
        normalize_keys=["u", "v", "p"],
        force_reload=False,
        name="dataset",
        verbose=False,
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.split = split
        self.num_samples = num_samples
        self.data_dir = os.path.join(data_dir, self.split)
        self.input_keys = invar_keys
        self.output_keys = outvar_keys

        print(f"Preparing the {split} dataset...")

        all_entries = os.listdir(self.data_dir)

        data_list = [
            os.path.join(self.data_dir, entry)
            for entry in all_entries
            if os.path.isfile(os.path.join(self.data_dir, entry))
        ]

        numbers = []
        for directory in data_list:
            match = re.search(r"\d+", directory)
            if match:
                numbers.append(int(match.group()))

        numbers = [int(n) for n in numbers]

        # sort
        args = np.argsort(numbers)
        self.data_list = [data_list[index] for index in args]
        numbers = [numbers[index] for index in args]

        # create the graphs with edge features
        self.length = min(len(self.data_list), self.num_samples)

        if self.num_samples > self.length:
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({self.length}) is less than the number of samples "
                f"({self.num_samples})"
            )

        self.graphs = []
        for i in range(self.length):
            # create the dgl graph
            file_path = self.data_list[i]
            polydata = read_vtp_file(file_path)
            graph = self._create_dgl_graph(polydata, outvar_keys, dtype=torch.int32)
            self.graphs.append(graph)

        self.graphs = self.add_edge_features()

        if self.split == "train":
            self.node_stats = self._get_node_stats(keys=normalize_keys)
            self.edge_stats = self._get_edge_stats()
        else:
            self.node_stats = load_json("node_stats.json")
            self.edge_stats = load_json("edge_stats.json")

        self.graphs = self.normalize_node()
        self.graphs = self.normalize_edge()

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph

    def __len__(self):
        return self.length

    def add_edge_features(self):
        """
        adds relative displacement & displacement norm as edge features
        """
        for i in range(len(self.graphs)):
            pos = self.graphs[i].ndata["pos"]
            row, col = self.graphs[i].edges()
            disp = torch.tensor(pos[row.long()] - pos[col.long()])
            disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
            self.graphs[i].edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return self.graphs

    def normalize_node(self):
        """normalizes node features"""
        invar_keys = set(
            [
                key.replace("_mean", "").replace("_std", "")
                for key in self.node_stats.keys()
            ]
        )
        for i in range(len(self.graphs)):
            for key in invar_keys:
                self.graphs[i].ndata[key] = (
                    self.graphs[i].ndata[key] - self.node_stats[key + "_mean"]
                ) / self.node_stats[key + "_std"]

            self.graphs[i].ndata["x"] = torch.cat(
                [self.graphs[i].ndata[key] for key in self.input_keys], dim=-1
            )
            self.graphs[i].ndata["y"] = torch.cat(
                [self.graphs[i].ndata[key] for key in self.output_keys], dim=-1
            )
        return self.graphs

    def normalize_edge(self):
        """normalizes a tensor"""
        for i in range(len(self.graphs)):
            self.graphs[i].edata["x"] = (
                self.graphs[i].edata["x"] - self.edge_stats["edge_mean"]
            ) / self.edge_stats["edge_std"]

        return self.graphs

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar

    def _get_edge_stats(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.length):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.length
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0) / self.length
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats(self, keys):
        stats = {}
        for key in keys:
            stats[key + "_mean"] = 0
            stats[key + "_meansqr"] = 0

        for i in range(self.length):
            for key in keys:
                stats[key + "_mean"] += (
                    torch.mean(self.graphs[i].ndata[key], dim=0) / self.length
                )
                stats[key + "_meansqr"] += (
                    torch.mean(torch.square(self.graphs[i].ndata[key]), dim=0)
                    / self.length
                )

        for key in keys:
            stats[key + "_std"] = torch.sqrt(
                stats[key + "_meansqr"] - torch.square(stats[key + "_mean"])
            )
            stats.pop(key + "_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    @staticmethod
    def _create_dgl_graph(
        polydata: Any,
        outvar_keys: List[str],
        to_bidirected: bool = True,
        add_self_loop: bool = False,
        dtype: Union[torch.dtype, str] = torch.int32,
    ) -> dgl.DGLGraph:
        """
        Create a DGL graph from vtkPolyData.

        Parameters
        ----------
        polydata : vtkPolyData
            vtkPolyData from which the DGL graph is created.
        outvar_keys : list of str
            List of keys for the node attributes to be extracted from the vtkPolyData.
        to_bidirected : bool, optional
            Whether to make the graph bidirected. Default is True.
        add_self_loop : bool, optional
            Whether to add self-loops in the graph. Default is False.
        dtype : torch.dtype or str, optional
            Data type for the graph. Default is torch.int32.

        Returns
        -------
        dgl.DGLGraph
            The DGL graph created from the vtkPolyData.
        """

        # Extract point data and connectivity information from the vtkPolyData
        points = polydata.GetPoints()
        vertices = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )

        polys = polydata.GetPolys()
        polys.InitTraversal()
        edge_list = []
        id_list = vtk.vtkIdList()

        for _ in range(polys.GetNumberOfCells()):
            polys.GetNextCell(id_list)
            num_ids = id_list.GetNumberOfIds()
            for j in range(num_ids):
                edge_list.append(  # noqa: PERF401
                    (id_list.GetId(j), id_list.GetId((j + 1) % num_ids))
                )

        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)
        if add_self_loop:
            graph = dgl.add_self_loop(graph)

        # Assign node features using the vertex data
        graph.ndata["pos"] = torch.tensor(vertices[:, :2], dtype=torch.float32)

        # Add one-hot embedding of markers
        point_data = polydata.GetPointData()
        marker = np.array(point_data.GetArray("marker"))
        num_classes = 5
        one_hot_marker = np.eye(num_classes)[marker.astype(int)]
        graph.ndata["marker"] = torch.tensor(one_hot_marker, dtype=torch.float32)

        # Extract node attributes from the vtkPolyData
        point_data = polydata.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            if array_name in outvar_keys:
                array_data = np.zeros(
                    (points.GetNumberOfPoints(), array.GetNumberOfComponents())
                )
                for j in range(points.GetNumberOfPoints()):
                    array.GetTuple(j, array_data[j])

                # Assign node attributes to the DGL graph
                graph.ndata[array_name] = torch.tensor(array_data, dtype=torch.float32)

        return graph
