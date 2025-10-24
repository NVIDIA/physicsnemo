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

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from typing import List, Dict


def midpoint_data_interp(
    pt1: np.ndarray, pt2: np.ndarray, points: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """
    Interpolate data on the midpoint of two points

    Parameters:
    -----------
    pt1 : np.ndarray
        Numpy array defining first point. Expected shape [1, 3]
    pt2 : np.ndarray
        Numpy array defining second point. Expected shape [1, 3]
    points : np.ndarray
        Numpy array containing all the points in the mesh. Expected shape [N, 3]
    field : np.ndarray
        Numpy array containing field values at all the points in the mesh.
        Expected shape [N, m]

    Returns:
    --------
    np.ndarray
        Value at the midpoint
    """
    idx1 = np.where(np.all(points == pt1, axis=1))[0]
    idx2 = np.where(np.all(points == pt2, axis=1))[0]

    return 0.5 * (field[idx1][0] + field[idx2][0])


def generate_mesh(
    nodes: np.ndarray, faces: np.ndarray, fields: np.ndarray
) -> pv.PolyData:
    """
    Generate mesh from given nodes, faces and fields arrays

    Args:
        nodes (np.ndarray): Nodes of the mesh
        faces (np.ndarray): Faces of the mesh
        fields (np.ndarray): Field values at each node

    Returns:
        pv.PolyData: Output mesh
    """
    points_3d = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    mesh = pv.PolyData(points_3d, faces_pv)
    for k, v in fields.items():
        mesh.point_data[k] = v

    return mesh


def compute_gradients(mesh: pv.PolyData, scalars: List[str]) -> pv.PolyData:
    """
    Compute the gradients of requested scalars for the given mesh

    Args:
        mesh (pv.PolyData): Input mesh
        scalars (List[str]): List of scalars to compute gradients for

    Returns:
        pv.PolyData: Output mesh with gradient information
    """

    for s in scalars:
        mesh = mesh.compute_derivative(scalars=s, gradient=f"grad_{s}")

    return mesh


def physicsnemo_geometry_interpolator(
    mesh: pv.PolyData, physicsnemo_geometry, num_samples: int
) -> Dict[str, np.ndarray]:
    """
    Interpolate mesh results on the boundary of a physicsnemo geometry object

    Args:
        mesh (pv.PolyData): Input mesh
        physicsnemo_geometry : PhysicsNeMo Geometry
        num_samples (int): Number of samples

    Returns:
        Dict[str, np.ndarray]: Samples with interpolated data
    """

    samples = physicsnemo_geometry.sample_boundary(num_samples)

    coords = np.concatenate((samples["x"], samples["y"]), axis=1)
    for k in mesh.point_data.keys():
        if k == "pyvistaOriginalPointIds":
            pass
        else:
            interp_vals = griddata(
                mesh.points.view(np.ndarray)[:, 0:2],
                mesh.point_data[k].view(np.ndarray),
                coords,
                method="linear",
            )

            samples[k] = interp_vals.reshape(-1, 1)

    return samples


def physicsnemo_geometry_interior_interpolator(
    mesh: pv.PolyData, physicsnemo_geometry, num_samples: int
) -> Dict[str, np.ndarray]:
    """
    Interpolate mesh results in the interior of a physicsnemo geometry object

    Args:
        mesh (pv.PolyData): Input mesh
        physicsnemo_geometry: PhysicsNeMo Geometry
        num_samples (int): Number of samples

    Returns:
        Dict[str, np.ndarray]: Samples with interpolated data
    """
    samples = physicsnemo_geometry.sample_interior(num_samples)

    coords = np.concatenate((samples["x"], samples["y"]), axis=1)
    for k in mesh.point_data.keys():
        if k == "pyvistaOriginalPointIds":
            pass
        else:
            interp_vals = griddata(
                mesh.points.view(np.ndarray)[:, 0:2],
                mesh.point_data[k].view(np.ndarray),
                coords,
                method="linear",
            )

            samples[k] = interp_vals.reshape(-1, 1)

    return samples
