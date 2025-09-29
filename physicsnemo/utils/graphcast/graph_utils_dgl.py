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

from typing import List, Tuple

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from torch import Tensor, testing


def create_graph(
    src: List,
    dst: List,
    to_bidirected: bool = True,
    add_self_loop: bool = False,
    dtype: torch.dtype = torch.int32,
) -> DGLGraph:
    """
    Creates a DGL graph from an adj matrix in COO format.

    Parameters
    ----------
    src : List
        List of source nodes
    dst : List
        List of destination nodes
    to_bidirected : bool, optional
        Whether to make the graph bidirectional, by default True
    add_self_loop : bool, optional
        Whether to add self loop to the graph, by default False
    dtype : torch.dtype, optional
        Graph index data type, by default torch.int32

    Returns
    -------
    DGLGraph
        The dgl Graph.
    """
    graph = dgl.graph((src, dst), idtype=dtype)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    return graph


def create_heterograph(
    src: List,
    dst: List,
    labels: str,
    dtype: torch.dtype = torch.int32,
    num_nodes_dict: dict = None,
) -> DGLGraph:
    """Creates a heterogeneous DGL graph from an adj matrix in COO format.

    Parameters
    ----------
    src : List
        List of source nodes
    dst : List
        List of destination nodes
    labels : str
        Label of the edge type
    dtype : torch.dtype, optional
        Graph index data type, by default torch.int32
    num_nodes_dict : dict, optional
        number of nodes for some node types, see dgl.heterograph for more information

    Returns
    -------
    DGLGraph
        The dgl Graph.
    """
    graph = dgl.heterograph(
        {labels: ("coo", (src, dst))}, num_nodes_dict=num_nodes_dict, idtype=dtype
    )
    return graph


def add_edge_features(graph: DGLGraph, pos: Tensor, normalize: bool = True) -> DGLGraph:
    """Adds edge features to the graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to add edge features to.
    pos : Tensor
        The node positions.
    normalize : bool, optional
        Whether to normalize the edge features, by default True

    Returns
    -------
    DGLGraph
        The graph with edge features.
    """

    if isinstance(pos, tuple):
        src_pos, dst_pos = pos
    else:
        src_pos = dst_pos = pos
    src, dst = graph.edges()

    src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    # azimuthal & polar rotation
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)

    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z", unit="rad")
    # y values should be zero
    try:
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")
    # x values should be one, y & z values should be zero
    try:
        testing.assert_close(dst_pos[:, 0], torch.ones_like(dst_pos[:, 0]))
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
        testing.assert_close(dst_pos[:, 2], torch.zeros_like(dst_pos[:, 2]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")

    # prepare edge features
    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    # normalize using the longest edge
    if normalize:
        max_disp_norm = torch.max(disp_norm)
        graph.edata["x"] = torch.cat(
            (disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1
        )
    else:
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)
    return graph


def add_node_features(graph: DGLGraph, pos: Tensor) -> DGLGraph:
    """Adds cosine of latitude, sine and cosine of longitude as the node features
    to the graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to add node features to.
    pos : Tensor
        The node positions.

    Returns
    -------
    graph : DGLGraph
        The graph with node features.
    """
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    graph.ndata["x"] = torch.stack(
        (torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1
    )
    return graph


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts latlon in degrees to xyz
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    latlon : Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    """
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts xyz to latlon in degrees
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    xyz : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    """
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == "rad":
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError("Not a valid unit")


def geospatial_rotation(
    invar: Tensor, theta: Tensor, axis: str, unit: str = "rad"
) -> Tensor:
    """Rotation using right hand rule

    Parameters
    ----------
    invar : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    theta : Tensor
        Tensor of shape (N, ) containing the rotation angle
    axis : str
        Axis of rotation
    unit : str, optional
        Unit of the theta, by default "rad"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing the rotated x, y, z coordinates
    """

    # get the right unit
    if unit == "deg":
        invar = rad2deg(invar)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")

    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")

    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def azimuthal_angle(lon: Tensor) -> Tensor:
    """
    Gives the azimuthal angle of a point on the sphere

    Parameters
    ----------
    lon : Tensor
        Tensor of shape (N, ) containing the longitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the azimuthal angle
    """
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def polar_angle(lat: Tensor) -> Tensor:
    """
    Gives the polar angle of a point on the sphere

    Parameters
    ----------
    lat : Tensor
        Tensor of shape (N, ) containing the latitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the polar angle
    """
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def deg2rad(deg: Tensor) -> Tensor:
    """Converts degrees to radians

    Parameters
    ----------
    deg :
        Tensor of shape (N, ) containing the degrees

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the radians
    """
    return deg * np.pi / 180


def rad2deg(rad):
    """Converts radians to degrees

    Parameters
    ----------
    rad :
        Tensor of shape (N, ) containing the radians

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the degrees
    """
    return rad * 180 / np.pi


def cell_to_adj(cells: List[List[int]]):
    """creates adjancy matrix in COO format from mesh cells

    Parameters
    ----------
    cells : List[List[int]]
        List of cells, each cell is a list of 3 vertices

    Returns
    -------
    src, dst : List[int], List[int]
        List of source and destination vertices
    """
    num_cells = np.shape(cells)[0]
    src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
    dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
    return src, dst


def max_edge_length(
    vertices: List[List[float]], source_nodes: List[int], destination_nodes: List[int]
) -> float:
    """
    Compute the maximum edge length in a graph.

    Parameters:
    vertices (List[List[float]]): A list of tuples representing the coordinates of the vertices.
    source_nodes (List[int]): A list of indices representing the source nodes of the edges.
    destination_nodes (List[int]): A list of indices representing the destination nodes of the edges.

    Returns:
    The maximum edge length in the graph (float).
    """
    vertices_np = np.array(vertices)
    source_coords = vertices_np[source_nodes]
    dest_coords = vertices_np[destination_nodes]

    # Compute the squared distances for all edges
    squared_differences = np.sum((source_coords - dest_coords) ** 2, axis=1)

    # Compute the maximum edge length
    max_length = np.sqrt(np.max(squared_differences))

    return max_length


def get_face_centroids(
    vertices: List[Tuple[float, float, float]], faces: List[List[int]]
) -> List[Tuple[float, float, float]]:
    """
    Compute the centroids of triangular faces in a graph.

    Parameters:
    vertices (List[Tuple[float, float, float]]): A list of tuples representing the coordinates of the vertices.
    faces (List[List[int]]): A list of lists, where each inner list contains three indices representing a triangular face.

    Returns:
    List[Tuple[float, float, float]]: A list of tuples representing the centroids of the faces.
    """
    centroids = []

    for face in faces:
        # Extract the coordinates of the vertices for the current face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Compute the centroid of the triangle
        centroid = (
            (v0[0] + v1[0] + v2[0]) / 3,
            (v0[1] + v1[1] + v2[1]) / 3,
            (v0[2] + v1[2] + v2[2]) / 3,
        )

        centroids.append(centroid)

    return centroids
