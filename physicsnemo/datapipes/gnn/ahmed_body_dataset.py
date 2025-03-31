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

import concurrent.futures as cf
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch import Tensor

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

from .utils import load_json, read_vtp_file, save_json

try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Ahmed Body Dataset requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    import pyvista as pv
    import vtk
except ImportError:
    raise ImportError(
        "Ahmed Body Dataset requires the vtk and pyvista libraries. Install with "
        + "pip install vtk pyvista"
    )

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """VTP file info storage."""

    velocity: float
    reynolds_number: float
    length: float
    width: float
    height: float
    ground_clearance: float
    slant_angle: float
    fillet_radius: float


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "AhmedBody"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


class AhmedBodyDataset(DGLDataset, Datapipe):
    """
    In-memory Ahmed body Dataset

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    invar_keys: Iterable[str], optional
        The input node features to consider. Default includes 'pos', 'velocity', 'reynolds_number', 'length', 'width', 'height', 'ground_clearance', 'slant_angle', and 'fillet_radius'.
    outvar_keys: Iterable[str], optional
        The output features to consider. Default includes 'p' and 'wallShearStress'.
    normalize_keys Iterable[str], optional
        The features to normalize. Default includes 'p', 'wallShearStress', 'velocity', 'length', 'width', 'height', 'ground_clearance', 'slant_angle', and 'fillet_radius'.
    normalization_bound: Tuple[float, float], optional
        The lower and upper bounds for normalization. Default is (-1, 1).
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    compute_drag: bool, optional
        If True, also returns the coefficient and mesh area and normals that are required for computing the drag coefficient.
    num_workers: int, optional
        Number of dataset pre-loading workers. If None, will be chosen automatically.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_samples: int = 10,
        invar_keys: Iterable[str] = (
            "pos",
            "velocity",
            "reynolds_number",
            "length",
            "width",
            "height",
            "ground_clearance",
            "slant_angle",
            "fillet_radius",
        ),
        outvar_keys: Iterable[str] = ("p", "wallShearStress"),
        normalize_keys: Iterable[str] = (
            "p",
            "wallShearStress",
            "velocity",
            "reynolds_number",
            "length",
            "width",
            "height",
            "ground_clearance",
            "slant_angle",
            "fillet_radius",
        ),
        normalization_bound: Tuple[float, float] = (-1.0, 1.0),
        force_reload: bool = False,
        name: str = "dataset",
        verbose: bool = False,
        compute_drag: bool = False,
        num_workers: Optional[int] = None,
    ):
        DGLDataset.__init__(
            self,
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        Datapipe.__init__(
            self,
            meta=MetaData(),
        )
        self.split = split
        self.num_samples = num_samples
        data_dir = Path(data_dir)
        self.data_dir = data_dir / self.split
        if not self.data_dir.is_dir():
            raise IOError(f"Directory not found {self.data_dir}")
        self.info_dir = data_dir / (self.split + "_info")
        if not self.info_dir.is_dir():
            raise IOError(f"Directory not found {self.info_dir}")
        self.input_keys = list(invar_keys)
        self.output_keys = list(outvar_keys)
        self.normalize_keys = list(normalize_keys)
        self.normalization_bound = normalization_bound
        self.compute_drag = compute_drag

        # Get case ids from the list of .vtp files.
        case_files = []
        case_info_files = []
        self.case_ids = []
        for case_file in sorted(self.data_dir.glob("*.vtp")):
            case_id = int(str(case_file.stem).removeprefix("case"))
            # Check if there is a corresponding info file.
            case_info_file = self.info_dir / f"case{case_id}_info.txt"
            if not case_info_file.is_file():
                raise IOError(f"File not found {case_info_file}")
            case_files.append(str(case_file))
            case_info_files.append(str(case_info_file))
            self.case_ids.append(case_id)

        self.length = min(len(self.case_ids), self.num_samples)
        logging.info(f"Using {self.length} {split} samples.")

        if self.num_samples > self.length:
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({self.length}) is less than the number of samples "
                f"({self.num_samples})"
            )

        self.graphs = [None] * self.length
        if self.compute_drag:
            self.normals = [None] * self.length
            self.areas = [None] * self.length
            self.coeff = [None] * self.length

        # create graphs from VTP files using multiprocessing.
        if num_workers is None or num_workers <= 0:

            def get_num_workers():
                # Make sure we don't oversubscribe CPUs on a node.
                # TODO(akamenev): this should be in DistributedManager.
                local_node_size = max(
                    int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1)), 1
                )
                num_workers = len(os.sched_getaffinity(0)) // local_node_size
                return max(num_workers - 1, 1)

            num_workers = get_num_workers()
        with cf.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=torch.multiprocessing.get_context("spawn"),
        ) as executor:
            for (i, graph, coeff, normal, area) in executor.map(
                self.create_graph,
                range(self.length),
                case_files[: self.length],
                case_info_files[: self.length],
                chunksize=max(1, self.length // num_workers),
            ):
                self.graphs[i] = graph
                if self.compute_drag:
                    self.coeff[i] = coeff
                    self.normals[i] = normal
                    self.areas[i] = area

        # add the edge features
        self.graphs = self.add_edge_features()

        # normalize the node and edge features
        if self.split == "train":
            self.node_stats = self._get_node_stats(keys=self.normalize_keys)
            self.edge_stats = self._get_edge_stats()
        else:
            if not os.path.exists("node_stats.json"):
                raise FileNotFoundError(
                    "node_stats.json not found! Node stats must be computed on the training set."
                )
            if not os.path.exists("edge_stats.json"):
                raise FileNotFoundError(
                    "edge_stats.json not found! Edge stats must be computed on the training set."
                )
            self.node_stats = load_json("node_stats.json")
            self.edge_stats = load_json("edge_stats.json")

        self.graphs = self.normalize_node()
        self.graphs = self.normalize_edge()

    def create_graph(self, index: int, file_path: str, info_path: str) -> None:
        """Creates a graph from VTP file.

        This method is used in parallel loading of graphs.

        Returns
        -------
            Tuple that contains graph index, graph, and optionally coeff, normal and area values.
        """
        polydata = read_vtp_file(file_path)
        graph = self._create_dgl_graph(polydata, self.output_keys, dtype=torch.int32)
        info = self._read_info_file(info_path)
        for v in vars(info):
            if v not in self.input_keys:
                continue
            graph.ndata[v] = getattr(info, v) * torch.ones_like(
                graph.ndata["pos"][:, [0]]
            )

        coeff = None
        normal = None
        area = None
        if "normals" in self.input_keys or self.compute_drag:
            mesh = pv.read(file_path)
            mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            if "normals" in self.input_keys:
                graph.ndata["normals"] = torch.from_numpy(
                    mesh.cell_data_to_point_data()["Normals"]
                )
            if self.compute_drag:
                mesh = mesh.compute_cell_sizes()
                mesh = mesh.cell_data_to_point_data()
                frontal_area = info.width * info.height / 2 * (10 ** (-6))
                coeff = 2.0 / ((info.velocity**2) * frontal_area)
                normal = torch.from_numpy(mesh["Normals"])
                area = torch.from_numpy(mesh["Area"])
        return index, graph, coeff, normal, area

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.compute_drag:
            case_id = self.case_ids[idx]
            return graph, case_id, self.normals[idx], self.areas[idx], self.coeff[idx]
        return graph

    def __len__(self):
        return self.length

    def add_edge_features(self) -> List[dgl.DGLGraph]:
        """
        Add relative displacement and displacement norm as edge features for each graph
        in the list of graphs. The calculations are done using the 'pos' attribute in the
        node data of each graph. The resulting edge features are stored in the 'x' attribute
        in the edge data of each graph.

        This method will modify the list of graphs in-place.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with updated edge features.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        for graph in self.graphs:
            pos = graph.ndata.get("pos")
            if pos is None:
                raise ValueError(
                    "'pos' does not exist in the node data of one or more graphs."
                )

            row, col = graph.edges()
            row = row.long()
            col = col.long()

            disp = pos[row] - pos[col]
            disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
            graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return self.graphs

    def normalize_node(self) -> List[dgl.DGLGraph]:
        """
        Normalize node data in each graph in the list of graphs.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with normalized and concatenated node data.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        if not hasattr(self, "node_stats") or not isinstance(self.node_stats, dict):
            raise ValueError(
                "The 'node_stats' attribute does not exist or is not a dictionary."
            )

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

    def normalize_edge(self) -> List[dgl.DGLGraph]:
        """
        Normalize edge data 'x' in each graph in the list of graphs.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with normalized edge data 'x'.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        if not hasattr(self, "edge_stats") or not isinstance(self.edge_stats, dict):
            raise ValueError(
                "The 'edge_stats' attribute does not exist or is not a dictionary."
            )

        for i in range(len(self.graphs)):
            self.graphs[i].edata["x"] = (
                self.graphs[i].edata["x"] - self.edge_stats["edge_mean"]
            ) / self.edge_stats["edge_std"]
        return self.graphs

    def denormalize(self, pred, gt, device) -> Tuple[Tensor, Tensor]:
        """
        Denormalize the graph node data.

        Parameters
        -----------
        pred: Tensor
            Normalized prediction
        gt: Tensor
            Normalized ground truth
        device: Any
            The device

        Returns
        --------
        Tuple(Tensor, Tensor)
            Denormalized prediction and ground truth
        """

        stats = self.node_stats
        stats = {key: val.to(device) for key, val in stats.items()}
        p_pred = pred[:, [0]]
        s_pred = pred[:, 1:]
        p_gt = gt[:, [0]]
        s_gt = gt[:, 1:]
        p_pred = p_pred * stats["p_std"] + stats["p_mean"]
        s_pred = s_pred * stats["wallShearStress_std"] + stats["wallShearStress_mean"]
        p_gt = p_gt * stats["p_std"] + stats["p_mean"]
        s_gt = s_gt * stats["wallShearStress_std"] + stats["wallShearStress_mean"]
        pred = torch.cat((p_pred, s_pred), dim=-1)
        gt = torch.cat((p_gt, s_gt), dim=-1)
        return pred, gt

    def _get_edge_stats(self) -> Dict[str, Any]:
        """
        Computes the mean and standard deviation of each edge attribute 'x' in the
        graphs, and saves to a JSON file.

        Returns
        -------
        dict
            A dictionary with keys 'edge_mean' and 'edge_std' and the corresponding values being
            1-D tensors containing the mean or standard deviation value for each dimension of the edge attribute 'x'.
        """
        if not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

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

    def _get_node_stats(self, keys: List[str]) -> Dict[str, Any]:
        """
        Computes the mean and standard deviation values of each node attribute
        for the list of keys in the graphs, and saves to a JSON file.

        Parameters
        ----------
        keys : list of str
            List of keys for the node attributes.

        Returns
        -------
        dict
            A dictionary with each key being a string of format '[key]_mean' or '[key]_std'
            and each value being a 1-D tensor containing the mean or standard deviation for each
            dimension of the node attribute.
        """
        if not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

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
    def _read_info_file(file_path: str) -> FileInfo:
        """
        Parse the values of specific parameters from a given text file.

        Parameters
        ----------
        file_path : str
            Path to the text file.

        Returns
        -------
        FileInfo
            A FileInfo object.
        """
        with open(file_path, mode="rt", encoding="utf-8") as file:
            info = yaml.safe_load(file)
            return FileInfo(
                info["Velocity"],
                info["Re (based on length)"],
                info["Length"],
                info["Width"],
                info["Height"],
                info["GroundClearance"],
                info["SlantAngle"],
                info["FilletRadius"],
            )

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
        if points is None:
            raise ValueError("Failed to get points from the polydata.")

        vertices = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )

        polys = polydata.GetPolys()
        if polys is None:
            raise ValueError("Failed to get polygons from the polydata.")

        polys.InitTraversal()

        edge_list = []
        for i in range(polys.GetNumberOfCells()):
            id_list = vtk.vtkIdList()
            polys.GetNextCell(id_list)
            for j in range(id_list.GetNumberOfIds() - 1):
                edge_list.append(  # noqa: PERF401
                    (id_list.GetId(j), id_list.GetId(j + 1))
                )

        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)
        if add_self_loop:
            graph = dgl.add_self_loop(graph)

        # Assign node features using the vertex data
        graph.ndata["pos"] = torch.tensor(vertices, dtype=torch.float32)

        # Extract node attributes from the vtkPolyData
        point_data = polydata.GetPointData()
        if point_data is None:
            raise ValueError("Failed to get point data from the polydata.")

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
