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

from typing import Dict, List, Optional, Union

import numpy as np
import pyvista as pv


def line_integral(
    edges: np.ndarray,
    points: np.ndarray,
    field: np.ndarray,
) -> float:
    """
    Compute integral along a line / edge. Currently assumes the curve in xy plane.

    Parameters:
    -----------
    edges : np.ndarray
        Edges of the curve in [M, 2] format
    points : np.ndarray
        Coordinates of points for the edge
    field : np.ndarray
        Field values at each edge center.

    Returns:
    --------
    float
        Integral along the given curve / line.
    """
    tangents = []
    lengths = []

    for i in range(edges.shape[0]):
        vec = points[edges[i, 1]] - points[edges[i, 0]]
        tangent = vec / np.linalg.norm(vec)
        tangents.append(tangent)
        lengths.append(np.linalg.norm(vec))

    tangents = np.array(tangents)
    lengths = np.array(lengths)

    # integrate the results
    if len(field.shape) == 2:
        # Vector quantity
        integral = np.sum(np.sum(field * tangents, axis=1) * lengths)
    else:
        # Scalar quantity
        integral = np.sum(field * lengths)

    return integral


def surface_integral(
    mesh: pv.PolyData,
    data_type: str = "point_data",
    array_name: Optional[Union[str, List[str]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes the surface integral of a given mesh

    Parameters:
    -----------
    mesh : pv.PolyData
        Mesh data with field information
    data_type : str, optional
        Whether to use "cell_data" or "point_data" to integrate. by default "point_data"
    array_name : Optional[Union[str, List[str]]], optional
        Array names to integrate. by default None which integrates all arrays.

    Returns:
    --------
    Dict[np.ndarray]
        Dictionary containing surface integrals for requested arrays.
    """
    # compute normals
    mesh = mesh.compute_normals()

    data = getattr(mesh, data_type)

    if not data.keys():
        raise ValueError(
            f"No data arrays found using type: {data_type}, try switching between 'point_data' and 'cell_data'"
        )

    available_vars = data.keys()

    if array_name is None:
        data_arr = data.keys()
    else:
        if not isinstance(array_name, (list, tuple)):
            data_arr = [array_name]
        else:
            data_arr = array_name

        if not set(data_arr) <= set(available_vars):
            raise ValueError(
                f"Requested vars not found in the provided mesh file. Choose from {available_vars}"
            )

    for arr in data_arr:
        if len(data[arr].shape) == 1:
            # Scalar quantity
            data[f"integral_{arr}"] = data[arr]
        elif len(data[arr].shape) == 2:
            # Vector quantity
            data[f"integral_{arr}"] = np.sum(data[arr] * data["Normals"], axis=1)

    # integrate the results
    integrated = mesh.integrate_data()

    results = {}
    for arr in data_arr:
        results[f"integral_{arr}"] = np.array(integrated[f"integral_{arr}"])

    return results
