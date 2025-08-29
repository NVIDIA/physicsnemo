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

import dgl
import numpy as np
import torch
import vtk
from torch import Tensor

try:
    import pyvista as pv
except:
    raise ImportError(
        "Stokes Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )


def relative_lp_error(pred, y, p=2):
    """
    Calculate relative L2 error norm

    Parameters:
    -----------
    pred: torch.Tensor
        Prediction
    y: torch.Tensor
        Ground truth

    Returns:
    --------
    error: float
        Calculated relative L2 error norm (percentage) on cpu
    """

    error = torch.mean(torch.norm(pred - y, p=p) / torch.norm(y, p=p)).cpu().numpy()
    return error * 100


# Inflow boundary condition
def parabolic_inflow(y, U_max):
    """parabolic inflow"""
    u = 4 * U_max * y * (0.4 - y) / (0.4**2)
    v = np.zeros_like(y)
    return u, v


def get_dataset(path, return_graph=False):
    """get_dataset file."""
    pv_mesh = pv.read(path)

    coords = np.array(pv_mesh.points[:, 0:2])

    # Extract the boundary markers
    mask = pv_mesh.point_data["marker"]

    inflow_coord_idx = mask == 1
    outflow_coord_idx = mask == 2
    wall_coords_idx = mask == 3
    polygon_coords_idx = mask == 4

    inflow_coords = coords[inflow_coord_idx]
    outflow_coords = coords[outflow_coord_idx]
    wall_coords = coords[wall_coords_idx]
    polygon_coords = coords[polygon_coords_idx]

    ref_u = np.array(pv_mesh.point_data["u"]).reshape(-1, 1)
    ref_v = np.array(pv_mesh.point_data["v"]).reshape(-1, 1)
    ref_p = np.array(pv_mesh.point_data["p"]).reshape(-1, 1)

    gnn_u = np.array(pv_mesh.point_data["pred_u"]).reshape(-1, 1)
    gnn_v = np.array(pv_mesh.point_data["pred_v"]).reshape(-1, 1)
    gnn_p = np.array(pv_mesh.point_data["pred_p"]).reshape(-1, 1)

    nu = 0.01

    if return_graph:
        # generate DGL graph
        polys = pv_mesh.GetPolys()
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

        graph = dgl.graph(edge_list, idtype=torch.int32)

        # Assign node features using the vertex data
        points = pv_mesh.GetPoints()
        vertices = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )
        graph.ndata["pos"] = torch.tensor(vertices[:, :2], dtype=torch.float32)

        # Add one-hot embedding of markers
        point_data = pv_mesh.GetPointData()
        marker = np.array(point_data.GetArray("marker"))
        num_classes = 5
        one_hot_marker = np.eye(num_classes)[marker.astype(int)]
        graph.ndata["marker"] = torch.tensor(one_hot_marker, dtype=torch.float32)

        # Extract node attributes from the vtkPolyData
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            if array_name in ["u", "v", "p"]:
                array_data = np.zeros(
                    (points.GetNumberOfPoints(), array.GetNumberOfComponents())
                )
                for j in range(points.GetNumberOfPoints()):
                    array.GetTuple(j, array_data[j])

                # Assign node attributes to the DGL graph
                graph.ndata[array_name] = torch.tensor(array_data, dtype=torch.float32)

        # compute freq features
        B = 10 * torch.randn((2, 64))
        x_proj = torch.matmul(graph.ndata["pos"], B)
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        graph.ndata["freq"] = x_proj

        graph.ndata["x"] = torch.cat(
            [graph.ndata[key] for key in ["pos", "marker", "freq"]], dim=-1
        )
        graph.ndata["y"] = torch.cat(
            [graph.ndata[key] for key in ["u", "v", "p"]], dim=-1
        )

        pos = graph.ndata["pos"]
        row, col = graph.edges()
        disp = torch.tensor(pos[row.long()] - pos[col.long()])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)
        return (
            ref_u,
            ref_v,
            ref_p,
            gnn_u,
            gnn_v,
            gnn_p,
            coords,
            inflow_coords,
            outflow_coords,
            wall_coords,
            polygon_coords,
            nu,
            graph,
        )
    else:
        return (
            ref_u,
            ref_v,
            ref_p,
            gnn_u,
            gnn_v,
            gnn_p,
            coords,
            inflow_coords,
            outflow_coords,
            wall_coords,
            polygon_coords,
            nu,
        )
