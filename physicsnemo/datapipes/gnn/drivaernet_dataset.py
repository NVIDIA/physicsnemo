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
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch_geometric as pyg
import yaml
from torch import Tensor
from torch.utils.data import Dataset

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData
from physicsnemo.nn.module.gnn_layers.utils import PyGData

try:
    import pyvista as pv
    import vtk
except ImportError:
    raise ImportError(
        "DrivAerNet Dataset requires the vtk and pyvista libraries. "
        "Install with pip install vtk pyvista"
    )


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "DrivAerNet"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


class DrivAerNetDataset(Dataset, Datapipe):
    """
    DrivAerNet dataset.

    Note: DrivAerNetDataset caches graphs in __getitem__ call which
    helps to avoid long initialization delay but increases first epoch time.

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    coeff_filename: str, optional
        DrivAerNet coefficients file name, default is from the dataset location.
    invar_keys: Iterable[str], optional
        The input node features to consider. Default includes 'pos'.
    outvar_keys: Iterable[str], optional
        The output features to consider. Default includes 'p' and 'wallShearStress'.
    normalize_keys Iterable[str], optional
        The features to normalize. Default includes 'p' and 'wallShearStress'.
    cache_dir: str, optional
        Path to the cache directory to store graphs in PyG format for fast loading.
        Default is ./cache/.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    force_reload: bool, optional
        If True, forces a reload of the cached data, by default False.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        num_samples: int = 10,
        coeff_filename: str = "AeroCoefficients_DrivAerNet_FilteredCorrected.csv",
        invar_keys: Iterable[str] = ("pos",),
        outvar_keys: Iterable[str] = ("p", "wallShearStress"),
        normalize_keys: Iterable[str] = ("p", "wallShearStress"),
        cache_dir: str | Path = "./cache/",
        name: str = "dataset",
        force_reload: bool = False,
        **kwargs,
    ) -> None:
        Datapipe.__init__(self, meta=MetaData())

        self.name = name
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(
                f"Path {self.data_dir} does not exist or is not a directory."
            )
        self.p_vtk_dir = self.data_dir / "SurfacePressureVTK"
        self.wss_vtk_dir = self.data_dir / "WallShearStressVTK"

        self.split = split.lower()
        if split not in (splits := ["train", "val", "test"]):
            raise ValueError(f"{split = } is not supported, must be one of {splits}.")

        self.force_reload = force_reload
        self.num_samples = num_samples
        self.input_keys = list(invar_keys)
        self.output_keys = list(outvar_keys)
        self.normalize_keys = list(normalize_keys)

        self.cache_dir = (
            self._get_cache_dir(self.data_dir, Path(cache_dir))
            if cache_dir is not None
            else None
        )

        # Load split design ids used to select a corresponding data split.
        design_ids = pd.read_csv(
            self.data_dir / f"{split}_design_ids.txt", header=None, index_col=0
        )

        # Read coefficients file which contains Cd, Cl etc.
        coeffs = pd.read_csv(self.data_dir / coeff_filename, index_col="Design")
        coeffs = coeffs.join(design_ids, how="inner")

        # Read projected areas file which is in YAML-like format with entries that look like:
        # combined_DrivAer_F_D_WM_WW_1234.stl: 2.574603830871618
        with open(self.data_dir / "projected_areas.txt", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        proj_areas = pd.DataFrame.from_dict(
            {k.removeprefix("combined_").removesuffix(".stl"): v for k, v in y.items()},
            orient="index",
            columns=["proj_area_x"],
        )

        # TODO(akamenev):
        # DrivAerNet issue #1: there are 10 entries missing in
        # projected_areas.txt:
        # train: DrivAer_F_D_WM_WW_0132, 0797, 1118, 1421, 1556, 1891, 2353, 2459.
        # val: DrivAer_F_D_WM_WW_0603, 3199.
        #
        # DrivAerNet issue #2: there are 2 entries for which WSS vtk files are empty.
        #
        # Filter both of them out (can do it via join but this is more explicit).
        missing_ids = {
            "DrivAer_F_D_WM_WW_0132",
            "DrivAer_F_D_WM_WW_0603",
            "DrivAer_F_D_WM_WW_0797",
            "DrivAer_F_D_WM_WW_1118",
            "DrivAer_F_D_WM_WW_1421",
            "DrivAer_F_D_WM_WW_1556",
            "DrivAer_F_D_WM_WW_1891",
            "DrivAer_F_D_WM_WW_2353",
            "DrivAer_F_D_WM_WW_2459",
            "DrivAer_F_D_WM_WW_3199",
        }
        empty_wss = {
            "DrivAer_F_D_WM_WW_0978",
            "DrivAer_F_D_WM_WW_3641",
        }
        coeffs = coeffs.drop(missing_ids | empty_wss, errors="ignore")

        # Merge projected areas into the coeffs dataframe.
        coeffs = coeffs.join(proj_areas, how="inner")

        if self.num_samples > len(coeffs):
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({len(coeffs)}) is less than the number of samples "
                f"({self.num_samples})"
            )

        coeffs.sort_index(inplace=True)
        self.coeffs = coeffs.iloc[: self.num_samples]

        # TODO(akamenev): these are estimates from small sample, need to compute from full data.
        self.nstats = {
            k: {"mean": v[0], "std": v[1]}
            for k, v in {
                "p": (-94.50448, 117.25317),
                "wallShearStress": (
                    torch.tensor([-0.56926626, 0.0027714, -0.07354721]),
                    torch.tensor([0.82198745, 0.45956784, 0.7490267]),
                ),
            }.items()
        }

        self.estats = {
            "x": {
                "mean": torch.tensor([0, 0, 0, 0.01338306]),
                "std": torch.tensor([0.00512953, 0.00953013, 0.00923065, 0.00482016]),
            }
        }

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, idx: int) -> PyGData:
        if not 0 <= idx < len(self):
            raise IndexError(f"Invalid {idx = }, must be in [0, {len(self)})")

        coeffs = self.coeffs.iloc[idx]
        gname = coeffs.name

        if self.cache_dir is None:
            # Caching is disabled - create the graph.
            graph = self._create_graph(gname)
        else:
            cached_graph_filename = self.cache_dir / (gname + ".pt")
            if not self.force_reload and cached_graph_filename.is_file():
                graph = torch.load(cached_graph_filename, weights_only=False)
            else:
                graph = self._create_graph(gname)
                Path.mkdir(self.cache_dir, parents=True, exist_ok=True)
                torch.save(graph, cached_graph_filename)

        # Set graph inputs/outputs.
        graph.x = torch.cat([graph[k] for k in self.input_keys], dim=-1)
        graph.y = torch.cat([graph[k] for k in self.output_keys], dim=-1)

        return {
            "name": gname,
            "graph": graph,
            "c_d": torch.tensor(coeffs["Average Cd"], dtype=torch.float32),
        }

    @staticmethod
    def _get_cache_dir(data_dir, cache_dir):
        if not cache_dir.is_absolute():
            cache_dir = data_dir / cache_dir
        return cache_dir.resolve()

    def _create_graph(
        self,
        name: str,
        to_bidirected: bool = True,
    ) -> PyGData:
        """Creates a PyG graph from DrivAerNet VTK data.

        Parameters
        ----------
        name : str
            Name of the graph in DrivAerNet.
        to_bidirected : bool, optional
            Whether to make the graph bidirected. Default is True.

        Returns
        -------
        PyGData
            The PyG graph.
        """

        def extract_edges(mesh: pv.PolyData) -> list[tuple[int, int]]:
            # Extract connectivity information from the mesh.
            # Traversal API is faster comparing to iterating over mesh.cell.
            polys = mesh.GetPolys()
            if polys is None:
                raise ValueError("Failed to get polygons from the mesh.")

            polys.InitTraversal()

            edge_list = []
            for _ in range(polys.GetNumberOfCells()):
                id_list = vtk.vtkIdList()
                polys.GetNextCell(id_list)
                num_ids = id_list.GetNumberOfIds()
                for j in range(num_ids - 1):
                    edge_list.append(  # noqa: PERF401
                        (id_list.GetId(j), id_list.GetId(j + 1))
                    )
                # Add the final edge between the last and the first vertices.
                edge_list.append((id_list.GetId(num_ids - 1), id_list.GetId(0)))

            return edge_list

        def permute_mesh(p_vtk_path: Path, wss_vtk_path: Path) -> Tensor:
            # The issue with DrivAerNet dataset is pressure and WSS meshes
            # are stored in different files. Even though each file contains
            # the same mesh coordinates, the nodes are permuted (order does not match)
            # which makes it impossible to do simple point_data assignment.
            # This method permutes WSS mesh by using vtkProbeFilter.

            p_reader = vtk.vtkPolyDataReader()
            p_reader.SetFileName(p_vtk_path)
            p_reader.Update()
            p_out = p_reader.GetOutput()

            wss_reader = vtk.vtkPolyDataReader()
            wss_reader.SetFileName(wss_vtk_path)
            wss_reader.Update()
            wss_out = wss_reader.GetOutput()

            probe = vtk.vtkProbeFilter()
            # p mesh is the input for which corresponding values from
            # wss mesh are retrieved.
            probe.SetInputData(p_out)
            probe.SetSourceData(wss_out)
            probe.Update()

            probe_out = probe.GetOutput()
            wss_arr = probe_out.GetPointData().GetArray("wallShearStress")
            num_points = p_out.GetNumberOfPoints()
            wss = torch.empty((num_points, 3), dtype=torch.float32)
            for i in range(num_points):
                x, y, z = wss_arr.GetTuple3(i)
                wss[i, 0] = x
                wss[i, 1] = y
                wss[i, 2] = z

            return wss

        # Load the pressure mesh even if p is not selected.
        # The p and wss meshes contain the same mesh nodes,
        # so use nodes from p for simplicity.
        p_vtk_path = self.p_vtk_dir / (name + ".vtk")
        p_mesh = pv.read(p_vtk_path)

        edge_list = extract_edges(p_mesh)

        # Create PyG graph using the connectivity information
        edges = torch.tensor(edge_list).t()
        if to_bidirected:
            edges = pyg.utils.to_undirected(edges)
        graph = pyg.data.Data(edge_index=edges)

        # Assign node features using the vertex data
        graph.pos = torch.tensor(p_mesh.points, dtype=torch.float32)

        if (k := "p") in self.output_keys:
            graph[k] = torch.tensor(p_mesh.point_data[k], dtype=torch.float32)

        if (k := "wallShearStress") in self.output_keys:
            wss_vtk_path = self.wss_vtk_dir / (name + ".vtk")
            graph[k] = permute_mesh(p_vtk_path, wss_vtk_path)

        # Normalize nodes.
        for k in self.input_keys + self.output_keys:
            if k not in self.normalize_keys:
                continue
            v = (graph[k] - self.nstats[k]["mean"]) / self.nstats[k]["std"]
            graph[k] = v.unsqueeze(-1) if v.ndim == 1 else v

        # Add edge features which contain relative edge nodes displacement and
        # displacement norm. Stored as `x` in the graph edge data.
        u, v = graph.edge_index
        pos = graph.pos
        disp = pos[u] - pos[v]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edge_attr = torch.cat((disp, disp_norm), dim=-1)

        # Normalize edges.
        graph.edge_attr = (graph.edge_attr - self.estats["x"]["mean"]) / self.estats[
            "x"
        ]["std"]

        return graph

    @torch.no_grad
    def denormalize(
        self, pred: Tensor, gt: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Denormalizes the inputs using previously collected statistics."""

        def denorm(x: Tensor, name: str):
            stats = self.nstats[name]
            mean = torch.as_tensor(stats["mean"]).to(device)
            std = torch.as_tensor(stats["std"]).to(device)
            return x * std + mean

        pred_d = []
        gt_d = []
        pred_d.append(denorm(pred[:, :1], "p"))
        gt_d.append(denorm(gt[:, :1], "p"))

        if (k := "wallShearStress") in self.output_keys:
            pred_d.append(denorm(pred[:, 1:4], k))
            gt_d.append(denorm(gt[:, 1:4], k))

        return torch.cat(pred_d, dim=-1), torch.cat(gt_d, dim=-1)
