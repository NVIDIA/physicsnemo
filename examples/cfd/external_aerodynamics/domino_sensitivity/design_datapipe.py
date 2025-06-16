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

"""
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds 
in npy format. 

The datapipe processes surface meshes to create structured representations suitable for
machine learning tasks, computing various geometric properties and signed distance fields.
"""

from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    Callable,
    Sequence,
)

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import (
    normalize,
    create_grid,
    calculate_center_of_mass,
)
from torch.utils.data import Dataset
from physicsnemo.utils.sdf import signed_distance_field
from scipy.spatial import KDTree
from numpy.typing import NDArray


class DesignDatapipe(Dataset):
    def __init__(
        self,
        mesh: pv.PolyData,
        bounding_box: np.ndarray | tuple[NDArray[np.float32], NDArray[np.float32]],
        bounding_box_surface: np.ndarray
        | tuple[NDArray[np.float32], NDArray[np.float32]],
        grid_resolution: Sequence[int],
        stencil_size: int = 7,
        seed: int = 0,
    ):
        """Initialize a DesignDatapipe dataset based on a surface mesh sample.

        Args:
            mesh: A PyVista PolyData mesh representing the surface geometry.
            bounding_box: A 2x3 numpy array containing the min and max coordinates of the volume
                bounding box. Shape: [[x_min, y_min, z_min], [x_max, y_max, z_max]].
            bounding_box_surface: A 2x3 numpy array containing the min and max coordinates of the
                surface bounding box. Shape: [[x_min, y_min, z_min], [x_max, y_max, z_max]].
            grid_resolution: A sequence of 3 integers specifying the number of points along each
                dimension (nx, ny, nz) for the structured grid.
            stencil_size: The size of the stencil used for local operations. Defaults to 7.
            seed: Random seed for reproducibility. Defaults to 0.

        Raises:
            ValueError: If grid_resolution does not contain exactly 3 values
            ValueError: If bounding_box or bounding_box_surface are not 2x3 arrays
        """
        if len(grid_resolution) != 3:
            raise ValueError("grid_resolution must contain exactly 3 values")

        self.mesh = mesh
        rng = np.random.RandomState(seed)

        # Compute basic mesh properties
        length_scale = np.amax(self.mesh.points, 0) - np.amin(self.mesh.points, 0)

        stl_centers = self.mesh.cell_centers().points
        stl_faces = self.mesh.regular_faces
        mesh_indices_flattened = stl_faces.flatten()

        surface_areas = mesh.compute_cell_sizes(
            length=False, area=True, volume=False
        ).cell_data["Area"]

        surface_normals = -1.0 * np.array(mesh.cell_normals, dtype=np.float32)

        center_of_mass = calculate_center_of_mass(stl_centers, surface_areas)

        s_max = np.asarray(bounding_box_surface[1])
        s_min = np.asarray(bounding_box_surface[0])

        v_max = np.asarray(bounding_box[1])
        v_min = np.asarray(bounding_box[0])

        nx, ny, nz = grid_resolution
        grid = create_grid(v_max, v_min, grid_resolution)
        grid_reshaped = grid.reshape(nx * ny * nz, 3)

        # SDF on grid
        sdf_grid = signed_distance_field(
            mesh_vertices=mesh.points,
            mesh_indices=mesh_indices_flattened,
            input_points=grid_reshaped,
            use_sign_winding_number=True,
        )
        sdf_grid = np.array(sdf_grid).reshape(nx, ny, nz)

        s_grid = create_grid(s_max, s_min, grid_resolution)
        surf_grid_reshaped = s_grid.reshape(nx * ny * nz, 3)

        surf_sdf_grid = signed_distance_field(
            mesh_vertices=mesh.points,
            mesh_indices=mesh_indices_flattened,
            input_points=surf_grid_reshaped,
            use_sign_winding_number=True,
        )
        surf_sdf_grid = np.array(surf_sdf_grid).reshape(nx, ny, nz)

        # Sample surface_vertices
        grid = normalize(grid, v_max, v_min)
        s_grid = normalize(s_grid, s_max, s_min)

        surface_coordinates = stl_centers
        interp_func = KDTree(surface_coordinates)

        dd, ii = interp_func.query(surface_coordinates, k=stencil_size)
        surface_neighbors = surface_coordinates[ii]
        surface_neighbors = surface_neighbors[:, 1:] + 1e-6
        surface_neighbors_normals = surface_normals[ii]
        surface_neighbors_normals = surface_neighbors_normals[:, 1:]
        surface_neighbors_area = surface_areas[ii]
        surface_neighbors_area = surface_neighbors_area[:, 1:]

        pos_normals_com_surface = surface_coordinates - center_of_mass

        surface_coordinates = normalize(surface_coordinates, s_max, s_min)
        surface_neighbors = normalize(surface_neighbors, s_max, s_min)

        # Volume processing
        volume_coordinates = (v_max - v_min) * rng.rand(1000, 3) + v_min

        sdf_nodes, sdf_node_closest_point = signed_distance_field(
            mesh.points,
            mesh_indices_flattened,
            volume_coordinates,
            include_hit_points=True,
            use_sign_winding_number=True,
        )
        sdf_nodes = np.array(sdf_nodes).reshape(-1, 1)
        sdf_node_closest_point = np.array(sdf_node_closest_point)
        pos_normals_closest = volume_coordinates - sdf_node_closest_point
        pos_normals_com = volume_coordinates - center_of_mass
        volume_coordinates = normalize(volume_coordinates, v_max, v_min)
        vol_grid_max_min = np.float32(np.asarray([v_min, v_max]))
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

        self.out_dict = dict(
            pos_volume_closest=pos_normals_closest,
            pos_volume_center_of_mass=pos_normals_com,
            pos_surface_center_of_mass=pos_normals_com_surface,
            geometry_coordinates=stl_centers,
            grid=grid,
            surf_grid=s_grid,
            sdf_grid=sdf_grid,
            sdf_surf_grid=surf_sdf_grid,
            sdf_nodes=sdf_nodes,
            surface_mesh_centers=surface_coordinates,
            surface_mesh_neighbors=surface_neighbors,
            surface_normals=surface_normals,
            surface_areas=surface_areas,
            surface_neighbors_normals=surface_neighbors_normals,
            surface_neighbors_areas=surface_neighbors_area,
            volume_mesh_centers=volume_coordinates,
            volume_min_max=vol_grid_max_min,
            surface_min_max=surf_grid_max_min,
            length_scale=length_scale,
        )

    def __len__(self) -> int:
        """Return the number of faces in the mesh."""
        return self.mesh.n_faces_strict

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing surface mesh data for the specified index
        """
        keys = [
            "surface_mesh_centers",
            "surface_mesh_neighbors",
            "surface_normals",
            "surface_neighbors_normals",
            "surface_areas",
            "surface_neighbors_areas",
            "pos_surface_center_of_mass",
        ]

        return {k: self.out_dict[k][idx].astype(np.float32) for k in keys}


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    mesh: pv.PolyData = pv.read("./geometries/drivaer_1_single_solid_decimated3.stl")
    bounding_box: np.ndarray = np.array([[-3.5, -2.25, -0.32], [8.5, 2.25, 3.00]])
    bounding_box_surface: np.ndarray = np.array([[-1.1, -1.2, -0.32], [4.5, 1.2, 1.2]])

    fd = DesignDatapipe(
        mesh=mesh,
        bounding_box=bounding_box,
        bounding_box_surface=bounding_box_surface,
        grid_resolution=[128, 64, 48],
    )

    train_dataloader = DataLoader(fd, batch_size=256_000, shuffle=False)

    for i, sample_batched in enumerate(train_dataloader):
        print(f"{i=}, {sample_batched['surface_mesh_centers'].shape=}")
