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

"""

import time, random, copy
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from physicsnemo.utils.sdf import signed_distance_field
from physicsnemo.utils.domino.utils import *

AIR_DENSITY = 1.205
STREAM_VELOCITY = 30.00


def combine_stls(stl_path, stl_files):
    meshes = []
    for file in stl_files:
        if ".stl" in file:
            stl_file_path = os.path.join(stl_path, file)
            reader = pv.get_reader(stl_file_path)
            mesh_stl = reader.read()
            meshes.append(mesh_stl)
    combined_mesh = pv.merge(meshes)
    return combined_mesh


class DesignDatapipe(Dataset):
    """
    Datapipe for converting openfoam dataset to npy

    """

    def __init__(
        self,
        mesh_stl,
        bounding_box,
        bounding_box_surface,
        grid_resolution,
        stream_velocity,
        air_density,
        stencil_size=7,
        device: int = 0,
    ):
        self.mesh_stl = mesh_stl
        self.stl_vertices = self.mesh_stl.points
        self.num_points = self.mesh_stl.cell_centers().points.shape[0]
        self.bounding_box = bounding_box
        self.bounding_box_surface = bounding_box_surface
        self.device = device
        self.stencil_size = stencil_size
        self.grid_resolution = grid_resolution
        self.stream_velocity = stream_velocity
        self.air_density = air_density
        self.out_dict = self.process_stl()

    def __len__(self):
        return self.num_points
        # return 16

    def __getitem__(self, idx):
        surface_mesh_centers = self.out_dict["surface_mesh_centers"][idx]
        surface_mesh_neighbors = self.out_dict["surface_mesh_neighbors"][idx]
        surface_normals = self.out_dict["surface_normals"][idx]
        surface_neighbors_normals = self.out_dict["surface_neighbors_normals"][idx]
        surface_areas = self.out_dict["surface_areas"][idx]
        surface_neighbors_areas = self.out_dict["surface_neighbors_areas"][idx]
        pos_normals_com_surface = self.out_dict["pos_surface_center_of_mass"][idx]

        out_dict_new = {}
        out_dict_new["surface_mesh_centers"] = np.float32(surface_mesh_centers)
        out_dict_new["surface_mesh_neighbors"] = np.float32(surface_mesh_neighbors)
        out_dict_new["surface_normals"] = np.float32(surface_normals)
        out_dict_new["surface_neighbors_normals"] = np.float32(
            surface_neighbors_normals
        )
        out_dict_new["surface_areas"] = np.float32(surface_areas)
        out_dict_new["surface_neighbors_areas"] = np.float32(surface_neighbors_areas)
        out_dict_new["pos_surface_center_of_mass"] = np.float32(pos_normals_com_surface)

        return out_dict_new

    def process_stl(
        self,
    ):
        mesh_stl = self.mesh_stl
        length_scale = np.amax(
            np.amax(self.stl_vertices, 0) - np.amin(self.stl_vertices, 0)
        )
        stl_centers = mesh_stl.cell_centers().points
        # Assuming triangular elements
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
        mesh_indices_flattened = stl_faces.flatten()
        print(stl_centers.shape, self.stl_vertices.shape)

        surface_areas = mesh_stl.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        surface_areas = np.array(surface_areas.cell_data["Area"])

        surface_normals = -1.0 * np.array(mesh_stl.cell_normals, dtype=np.float32)

        center_of_mass = calculate_center_of_mass(stl_centers, surface_areas)

        s_max = np.asarray(self.bounding_box_surface[1])
        s_min = np.asarray(self.bounding_box_surface[0])

        v_max = np.asarray(self.bounding_box[1])
        v_min = np.asarray(self.bounding_box[0])

        # General processing
        nx, ny, nz = self.grid_resolution

        grid = create_grid(v_max, v_min, self.grid_resolution)
        grid_reshaped = grid.reshape(nx * ny * nz, 3)

        # SDF on grid
        sdf_grid = signed_distance_field(
            mesh_vertices=self.stl_vertices,
            mesh_indices=mesh_indices_flattened,
            input_points=grid_reshaped,
            use_sign_winding_number=True,
        )
        sdf_grid = sdf_grid.numpy().reshape(nx, ny, nz)

        s_grid = create_grid(s_max, s_min, self.grid_resolution)
        surf_grid_reshaped = s_grid.reshape(nx * ny * nz, 3)

        surf_sdf_grid = signed_distance_field(
            mesh_vertices=self.stl_vertices,
            mesh_indices=mesh_indices_flattened,
            input_points=surf_grid_reshaped,
            use_sign_winding_number=True,
        )
        surf_sdf_grid = surf_sdf_grid.numpy().reshape(nx, ny, nz)

        # Sample surface_vertices
        grid = 2.0 * (grid - v_min) / (v_max - v_min) - 1.0
        s_grid = 2.0 * (s_grid - s_min) / (s_max - s_min) - 1.0

        # Surface processing
        surface_coordinates = stl_centers
        interp_func = KDTree(surface_coordinates)
        dd, ii = interp_func.query(surface_coordinates, k=self.stencil_size)
        surface_neighbors = surface_coordinates[ii]
        surface_neighbors = surface_neighbors[:, 1:] + 1e-6
        surface_neighbors_normals = surface_normals[ii]
        surface_neighbors_normals = surface_neighbors_normals[:, 1:]
        surface_neighbors_area = surface_areas[ii]
        surface_neighbors_area = surface_neighbors_area[:, 1:]

        pos_normals_com_surface = surface_coordinates - center_of_mass

        surface_coordinates = (
            2.0 * (surface_coordinates - s_min) / (s_max - s_min) - 1.0
        )
        surface_neighbors = 2.0 * (surface_neighbors - s_min) / (s_max - s_min) - 1.0

        # Volume processing
        volume_coordinates = (v_max - v_min) * np.random.rand(10, 3) + v_min

        sdf_nodes, sdf_node_closest_point = signed_distance_field(
            self.stl_vertices,
            mesh_indices_flattened,
            volume_coordinates,
            include_hit_points=True,
            use_sign_winding_number=True,
        )
        sdf_nodes = sdf_nodes.numpy().reshape(-1, 1)
        sdf_node_closest_point = sdf_node_closest_point.numpy()
        pos_normals_closest = volume_coordinates - sdf_node_closest_point
        pos_normals_com = volume_coordinates - center_of_mass
        volume_coordinates = 2.0 * (volume_coordinates - v_min) / (v_max - v_min) - 1.0
        vol_grid_max_min = np.float32(np.asarray([v_min, v_max]))
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

        geometry_points = 300_000
        geometry_coordinates_sampled, idx_geometry = shuffle_array(
            stl_centers, geometry_points
        )

        # surface_points = 16
        # surface_coordinates = surface_coordinates[:surface_points]
        # surface_neighbors = surface_neighbors[:surface_points]
        # surface_normals = surface_normals[:surface_points]
        # surface_neighbors_normals = surface_neighbors_normals[:surface_points]
        # surface_areas = surface_areas[:surface_points]
        # surface_neighbors_area = surface_neighbors_area[:surface_points]
        # pos_normals_com_surface = pos_normals_com_surface[:surface_points]

        return {
            "pos_volume_closest": pos_normals_closest,
            "pos_volume_center_of_mass": pos_normals_com,
            "pos_surface_center_of_mass": pos_normals_com_surface,
            "geometry_coordinates": geometry_coordinates_sampled,
            "grid": grid,
            "surf_grid": s_grid,
            "sdf_grid": sdf_grid,
            "sdf_surf_grid": surf_sdf_grid,
            "sdf_nodes": sdf_nodes,
            "surface_mesh_centers": surface_coordinates,
            "surface_mesh_neighbors": surface_neighbors,
            "surface_normals": surface_normals,
            "surface_neighbors_normals": surface_neighbors_normals,
            "surface_areas": surface_areas,
            "surface_neighbors_areas": surface_neighbors_area,
            "volume_mesh_centers": volume_coordinates,
            "volume_min_max": vol_grid_max_min,
            "surface_min_max": surf_grid_max_min,
            "length_scale": length_scale,
            "stream_velocity": np.expand_dims(
                np.array(self.stream_velocity, dtype=np.float32), -1
            ),
            "air_density": np.expand_dims(
                np.array(self.air_density, dtype=np.float32), -1
            ),
        }


if __name__ == "__main__":
    stl_path = "/raid/rranade/home/rranade/data/"
    dirnames = get_filenames(stl_path)
    filepath = os.path.join(stl_path, dirnames[0])
    stl_files = get_filenames(filepath)
    mesh_stl = combine_stls(filepath, stl_files)

    bounding_box = [[-3.5, -2.25, -0.32], [8.5, 2.25, 3.00]]
    bounding_box_surface = [[-1.1, -1.2, -0.32], [4.5, 1.2, 1.2]]

    fd = DesignDatapipe(
        mesh_stl,
        bounding_box,
        bounding_box_surface,
        grid_resolution=[128, 64, 48],
        stream_velocity=30.0,
        air_density=1.205,
        device=0,
    )

    train_dataloader = DataLoader(fd, batch_size=256_000, shuffle=False)

    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched["surface_mesh_centers"].shape)
