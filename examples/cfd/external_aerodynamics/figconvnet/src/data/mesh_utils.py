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
from typing import Tuple

import torch
import numpy as np
from jaxtyping import Float


def compute_drag_coefficient(
    poly_normals: np.ndarray,
    poly_area: np.ndarray,
    coeff: float,
    poly_pressure: np.ndarray,
    poly_wss: np.ndarray,
    dir_movement: np.ndarray = np.array([-1, 0, 0]),
):
    """Compute drag coefficient of the mesh assuming the movement direction is negative x-axis.
    Reference: https://www.idealsimulations.com/simworks-tutorials/drag-coefficient-of-a-sphere-cfd-simulation/

    Parameters:
    -----------
    poly_normals: The surface normals on cells (e.g. polygons, triangles) on the mesh
    poly_area: The surface areas of each cell
    coeff: 2/(A * rho * U^2) where rho is the density, U the velocity, and A the cross-sectional area along the movement direction
    poly_pressure: The pressure on each cell
    poly_wss: The wall shear stress on each cell
    dir_movement: The direction of movement, default is -x axis

    Returns:
    --------
    c_drag: float:
        Computed drag coefficient
    """

    # Compute coefficients
    c_p = coeff * np.dot(np.dot(poly_normals, dir_movement), poly_area * poly_pressure)
    c_f = coeff * np.abs(np.dot(np.dot(poly_wss, dir_movement), poly_area))

    # Compute total drag coefficients
    c_drag = c_p + c_f
    return c_drag


def bbox_to_centers(
    bbox_min: Float[torch.Tensor, "3"],
    bbox_max: Float[torch.Tensor, "3"],
    resolution: Tuple[int, int, int] = [64, 64, 64],
):
    """Compute the centers of the cells in a 3D grid defined by a bounding box.

    Parameters:
    -----------
    bbox_min: torch.Tensor[3]
        The minimum coordinates of the bounding box
    bbox_max: torch.Tensor[3]
        The maximum coordinates of the bounding box
    resolution: Tuple[int, int, int]
        The resolution of the grid

    Returns:
    --------
    centers: torch.Tensor[resolution[0] * resolution[1] * resolution[2], 3]
        The centers of the cells in the grid
    """

    # Compute the cell size
    cell_size = (bbox_max - bbox_min) / torch.tensor(resolution)

    # Compute the grid
    x = torch.linspace(bbox_min[0], bbox_max[0], resolution[0])
    y = torch.linspace(bbox_min[1], bbox_max[1], resolution[1])
    z = torch.linspace(bbox_min[2], bbox_max[2], resolution[2])

    # Compute the centers of the cells
    centers = (
        torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3) + cell_size / 2
    )
    return centers
