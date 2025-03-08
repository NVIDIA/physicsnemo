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
This code provides the datapipe for reading the processed npy files,
generating multi-res grids, calculating signed distance fields, 
positional encodings, sampling random points in the volume and on surface, 
normalizing fields and returning the output tensors as a dictionary.

This datapipe also non-dimensionalizes the fields, so the order in which the variables should 
be fixed: velocity, pressure, turbulent viscosity for volume variables and 
pressure, wall-shear-stress for surface variables. The different parameters such as 
variable names, domain resolution, sampling size etc. are configurable in config.yaml. 
"""

from pathlib import Path
from typing import (
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from torch.utils.data import Dataset

from modulus.utils.domino.utils import (
    KDTree,
    area_weighted_shuffle_array,
    calculate_center_of_mass,
    calculate_normal_positional_encoding,
    create_grid,
    get_filenames,
    normalize,
    pad,
    shuffle_array,
    standardize,
)
from modulus.utils.sdf import signed_distance_field


class DoMINODataPipe(Dataset):
    """
    Datapipe for DoMINO

    """

    def __init__(
        self,
        data_path: Union[str, Path],  # Input data path
        phase: Literal["train", "val", "test"] = "train",  # Train, test or val
        surface_variables: Optional[Sequence] = (
            "pMean",
            "wallShearStress",
        ),  # Names of surface variables
        volume_variables: Optional[Sequence] = (
            "UMean",
            "pMean",
        ),  # Names of volume variables
        sampling: bool = False,  # Sampling True or False
        device: int = 0,  # GPU device id
        grid_resolution: Optional[Sequence] = (
            256,
            96,
            64,
        ),  # Resolution of latent grid
        normalize_coordinates: bool = False,  # Normalize coordinates?
        sample_in_bbox: bool = False,  # Sample points in a specified bounding box
        volume_points_sample: int = 1024,  # Number of volume points sampled per batch
        surface_points_sample: int = 1024,  # Number of surface points sampled per batch
        geom_points_sample: int = 300000,  # Number of STL points sampled per batch
        positional_encoding: bool = False,  # Positional encoding, True or False
        volume_factors=None,  # Non-dimensionalization factors for volume variables
        surface_factors=None,  # Non-dimensionalization factors for surface variables
        scaling_type=None,  # Scaling min_max or mean_std
        model_type=None,  # Model_type, surface, volume or combined
        bounding_box_dims=None,  # Dimensions of bounding box
        bounding_box_dims_surf=None,  # Dimensions of bounding box
        compute_scaling_factors=False,
        num_surface_neighbors=11,  # Surface neighbors to consider
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()

        self.data_path = data_path

        if phase not in [
            "train",
            "val",
            "test",
        ]:
            raise AssertionError(
                f"phase should be one of ['train', 'val', 'test'], got {phase}"
            )

        if not self.data_path.exists():
            raise AssertionError(f"Path {self.data_path} does not exist")

        if not self.data_path.is_dir():
            raise AssertionError(f"Path {self.data_path} is not a directory")

        self.sampling = sampling
        self.grid_resolution = grid_resolution
        self.normalize_coordinates = normalize_coordinates
        self.model_type = model_type
        self.bounding_box_dims = []
        self.bounding_box_dims.append(np.asarray(bounding_box_dims.max))
        self.bounding_box_dims.append(np.asarray(bounding_box_dims.min))

        self.bounding_box_dims_surf = []
        self.bounding_box_dims_surf.append(np.asarray(bounding_box_dims_surf.max))
        self.bounding_box_dims_surf.append(np.asarray(bounding_box_dims_surf.min))

        self.filenames = get_filenames(self.data_path)
        total_files = len(self.filenames)

        self.phase = phase
        if phase == "train":
            self.indices = np.array(range(total_files))
        elif phase == "val":
            self.indices = np.array(range(total_files))
        elif phase == "test":
            self.indices = np.array(range(total_files))

        np.random.shuffle(self.indices)
        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.volume_points = volume_points_sample
        self.surface_points = surface_points_sample
        self.geom_points_sample = geom_points_sample
        self.sample_in_bbox = sample_in_bbox
        self.device = device
        self.positional_encoding = positional_encoding
        self.volume_factors = volume_factors
        self.surface_factors = surface_factors
        self.scaling_type = scaling_type
        self.compute_scaling_factors = compute_scaling_factors
        self.num_surface_neighbors = num_surface_neighbors

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        cfd_filename = self.filenames[index]

        filepath = self.data_path / cfd_filename
        data_dict = np.load(filepath, allow_pickle=True).item()

        stl_vertices = data_dict["stl_coordinates"]
        stl_centers = data_dict["stl_centers"]
        mesh_indices_flattened = data_dict["stl_faces"]
        stl_sizes = data_dict["stl_areas"]

        # Check if stream velocity in keys
        if "stream_velocity" in data_dict.keys():
            STREAM_VELOCITY = data_dict["stream_velocity"]
            AIR_DENSITY = data_dict["air_density"]
        else:
            AIR_DENSITY = 1.205
            STREAM_VELOCITY = 30.00

        #
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        if self.bounding_box_dims_surf is None:
            s_max = np.amax(stl_vertices, 0)
            s_min = np.amin(stl_vertices, 0)
        else:
            s_max = np.float32(self.bounding_box_dims_surf[0])
            s_min = np.float32(self.bounding_box_dims_surf[1])

        nx, ny, nz = self.grid_resolution

        surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

        # SDF calculation on the grid using WARP
        sdf_surf_grid = (
            signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                surf_grid_reshaped,
                use_sign_winding_number=True,
            )
            .numpy()
            .reshape(nx, ny, nz)
        )
        surf_grid = np.float32(surf_grid)
        sdf_surf_grid = np.float32(sdf_surf_grid)
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

        if self.model_type == "volume" or self.model_type == "combined":
            volume_coordinates = data_dict["volume_mesh_centers"]
            volume_fields = data_dict["volume_fields"]

            if not self.compute_scaling_factors:
                if self.bounding_box_dims is None:
                    c_max = s_max + (s_max - s_min) / 2
                    c_min = s_min - (s_max - s_min) / 2
                    c_min[2] = s_min[2]
                else:
                    c_max = np.float32(self.bounding_box_dims[0])
                    c_min = np.float32(self.bounding_box_dims[1])

                ids_in_bbox = np.where(
                    (volume_coordinates[:, 0] > c_min[0])
                    & (volume_coordinates[:, 0] < c_max[0])
                    & (volume_coordinates[:, 1] > c_min[1])
                    & (volume_coordinates[:, 1] < c_max[1])
                    & (volume_coordinates[:, 2] > c_min[2])
                    & (volume_coordinates[:, 2] < c_max[2])
                )

                if self.sample_in_bbox:
                    volume_coordinates = volume_coordinates[ids_in_bbox]
                    volume_fields = volume_fields[ids_in_bbox]

                dx, dy, dz = (
                    (c_max[0] - c_min[0]) / nx,
                    (c_max[1] - c_min[1]) / ny,
                    (c_max[2] - c_min[2]) / nz,
                )

                # Generate a grid of specified resolution to map the bounding box
                # The grid is used for capturing structured geometry features and SDF representation of geometry
                grid = create_grid(c_max, c_min, [nx, ny, nz])
                grid_reshaped = grid.reshape(nx * ny * nz, 3)

                # SDF calculation on the grid using WARP
                sdf_grid = (
                    signed_distance_field(
                        stl_vertices,
                        mesh_indices_flattened,
                        grid_reshaped,
                        use_sign_winding_number=True,
                    )
                    .numpy()
                    .reshape(nx, ny, nz)
                )

                if self.sampling:
                    volume_coordinates_sampled, idx_volume = shuffle_array(
                        volume_coordinates, self.volume_points
                    )
                    if volume_coordinates_sampled.shape[0] < self.volume_points:
                        volume_coordinates_sampled = pad(
                            volume_coordinates_sampled,
                            self.volume_points,
                            pad_value=-10.0,
                        )
                    volume_fields = volume_fields[idx_volume]
                    volume_coordinates = volume_coordinates_sampled

                sdf_nodes, sdf_node_closest_point = signed_distance_field(
                    stl_vertices,
                    mesh_indices_flattened,
                    volume_coordinates,
                    include_hit_points=True,
                    use_sign_winding_number=True,
                )
                sdf_nodes = sdf_nodes.numpy().reshape(-1, 1)
                sdf_node_closest_point = sdf_node_closest_point.numpy()

                if self.positional_encoding:
                    pos_normals_closest_vol = calculate_normal_positional_encoding(
                        volume_coordinates,
                        sdf_node_closest_point,
                        cell_length=[dx, dy, dz],
                    )
                    pos_normals_com_vol = calculate_normal_positional_encoding(
                        volume_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                    )
                else:
                    pos_normals_closest_vol = (
                        volume_coordinates - sdf_node_closest_point
                    )
                    pos_normals_com_vol = volume_coordinates - center_of_mass

                if self.normalize_coordinates:
                    volume_coordinates = normalize(volume_coordinates, c_max, c_min)
                    grid = normalize(grid, c_max, c_min)

                if self.scaling_type is not None:
                    if self.volume_factors is not None:
                        if self.scaling_type == "mean_std_scaling":
                            vol_mean = self.volume_factors[0]
                            vol_std = self.volume_factors[1]
                            volume_fields = standardize(
                                volume_fields, vol_mean, vol_std
                            )
                        elif self.scaling_type == "min_max_scaling":
                            vol_min = self.volume_factors[1]
                            vol_max = self.volume_factors[0]
                            volume_fields = normalize(volume_fields, vol_max, vol_min)

                volume_fields = np.float32(volume_fields)
                pos_normals_closest_vol = np.float32(pos_normals_closest_vol)
                pos_normals_com_vol = np.float32(pos_normals_com_vol)
                volume_coordinates = np.float32(volume_coordinates)
                sdf_nodes = np.float32(sdf_nodes)
                sdf_grid = np.float32(sdf_grid)
                grid = np.float32(grid)
                vol_grid_max_min = np.float32(np.asarray([c_min, c_max]))
            else:
                pos_normals_closest_vol = None
                pos_normals_com_vol = None
                sdf_nodes = None
                sdf_grid = None
                grid = None
                vol_grid_max_min = None

        else:
            volume_coordinates = None
            volume_fields = None
            pos_normals_closest_vol = None
            pos_normals_com_vol = None
            sdf_nodes = None
            sdf_grid = None
            grid = None
            vol_grid_max_min = None

        if self.model_type == "surface" or self.model_type == "combined":
            surface_coordinates = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_sizes = data_dict["surface_areas"]
            surface_fields = data_dict["surface_fields"]

            if not self.compute_scaling_factors:

                c_max = np.float32(self.bounding_box_dims[0])
                c_min = np.float32(self.bounding_box_dims[1])

                ids_in_bbox = np.where(
                    (surface_coordinates[:, 0] > c_min[0])
                    & (surface_coordinates[:, 0] < c_max[0])
                    & (surface_coordinates[:, 1] > c_min[1])
                    & (surface_coordinates[:, 1] < c_max[1])
                    & (surface_coordinates[:, 2] > c_min[2])
                    & (surface_coordinates[:, 2] < c_max[2])
                )
                surface_coordinates = surface_coordinates[ids_in_bbox]
                surface_normals = surface_normals[ids_in_bbox]
                surface_sizes = surface_sizes[ids_in_bbox]
                surface_fields = surface_fields[ids_in_bbox]

                # Get neighbors
                interp_func = KDTree(surface_coordinates)
                dd, ii = interp_func.query(
                    surface_coordinates, k=self.num_surface_neighbors
                )
                surface_neighbors = surface_coordinates[ii]
                surface_neighbors = surface_neighbors[:, 1:]

                surface_neighbors_normals = surface_normals[ii]
                surface_neighbors_normals = surface_neighbors_normals[:, 1:]
                surface_neighbors_sizes = surface_sizes[ii]
                surface_neighbors_sizes = surface_neighbors_sizes[:, 1:]

                dx, dy, dz = (
                    (s_max[0] - s_min[0]) / nx,
                    (s_max[1] - s_min[1]) / ny,
                    (s_max[2] - s_min[2]) / nz,
                )

                if self.positional_encoding:
                    pos_normals_com_surface = calculate_normal_positional_encoding(
                        surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                    )
                else:
                    pos_normals_com_surface = surface_coordinates - center_of_mass

                if self.normalize_coordinates:
                    surface_coordinates = normalize(surface_coordinates, s_max, s_min)
                    surface_neighbors = normalize(surface_neighbors, s_max, s_min)
                    surf_grid = normalize(surf_grid, s_max, s_min)

                if self.sampling:
                    (
                        surface_coordinates_sampled,
                        idx_surface,
                    ) = area_weighted_shuffle_array(
                        surface_coordinates, self.surface_points, surface_sizes
                    )
                    if surface_coordinates_sampled.shape[0] < self.surface_points:
                        surface_coordinates_sampled = pad(
                            surface_coordinates_sampled,
                            self.surface_points,
                            pad_value=-10.0,
                        )

                    surface_fields = surface_fields[idx_surface]
                    pos_normals_com_surface = pos_normals_com_surface[idx_surface]
                    surface_normals = surface_normals[idx_surface]
                    surface_sizes = surface_sizes[idx_surface]
                    surface_neighbors = surface_neighbors[idx_surface]
                    surface_neighbors_normals = surface_neighbors_normals[idx_surface]
                    surface_neighbors_sizes = surface_neighbors_sizes[idx_surface]
                    surface_coordinates = surface_coordinates_sampled

                if self.scaling_type is not None:
                    if self.surface_factors is not None:
                        if self.scaling_type == "mean_std_scaling":
                            surf_mean = self.surface_factors[0]
                            surf_std = self.surface_factors[1]
                            surface_fields = standardize(
                                surface_fields, surf_mean, surf_std
                            )
                        elif self.scaling_type == "min_max_scaling":
                            surf_min = self.surface_factors[1]
                            surf_max = self.surface_factors[0]
                            surface_fields = normalize(
                                surface_fields, surf_max, surf_min
                            )

                surface_coordinates = np.float32(surface_coordinates)
                surface_fields = np.float32(surface_fields)
                surface_sizes = np.float32(surface_sizes)
                surface_normals = np.float32(surface_normals)
                surface_neighbors = np.float32(surface_neighbors)
                surface_neighbors_normals = np.float32(surface_neighbors_normals)
                surface_neighbors_sizes = np.float32(surface_neighbors_sizes)
                pos_normals_com_surface = np.float32(pos_normals_com_surface)
            else:
                surface_sizes = None
                surface_normals = None
                surface_neighbors = None
                surface_neighbors_normals = None
                surface_neighbors_sizes = None
                pos_normals_com_surface = None

        else:
            surface_coordinates = None
            surface_fields = None
            surface_sizes = None
            surface_normals = None
            surface_neighbors = None
            surface_neighbors_normals = None
            surface_neighbors_sizes = None
            pos_normals_com_surface = None

        if self.sampling:
            geometry_points = self.geom_points_sample
            geometry_coordinates_sampled, idx_geometry = shuffle_array(
                stl_vertices, geometry_points
            )
            if geometry_coordinates_sampled.shape[0] < geometry_points:
                geometry_coordinates_sampled = pad(
                    geometry_coordinates_sampled, geometry_points, pad_value=-100.0
                )
            geom_centers = geometry_coordinates_sampled
        else:
            geom_centers = stl_vertices

        geom_centers = np.float32(geom_centers)

        if self.model_type == "combined":
            # Add the parameters to the dictionary
            return {
                "pos_volume_closest": pos_normals_closest_vol,
                "pos_volume_center_of_mass": pos_normals_com_vol,
                "pos_surface_center_of_mass": pos_normals_com_surface,
                "geometry_coordinates": geom_centers,
                "grid": grid,
                "surf_grid": surf_grid,
                "sdf_grid": sdf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "sdf_nodes": sdf_nodes,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "surface_fields": surface_fields,
                "volume_min_max": vol_grid_max_min,
                "surface_min_max": surf_grid_max_min,
                "length_scale": length_scale,
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), -1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), -1
                ),
            }
        elif self.model_type == "surface":
            return {
                "pos_surface_center_of_mass": pos_normals_com_surface,
                "geometry_coordinates": geom_centers,
                "surf_grid": surf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "surface_fields": surface_fields,
                "surface_min_max": surf_grid_max_min,
                "length_scale": length_scale,
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), -1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), -1
                ),
            }
        elif self.model_type == "volume":
            return {
                "pos_volume_closest": pos_normals_closest_vol,
                "pos_volume_center_of_mass": pos_normals_com_vol,
                "geometry_coordinates": geom_centers,
                "grid": grid,
                "surf_grid": surf_grid,
                "sdf_grid": sdf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "sdf_nodes": sdf_nodes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "volume_min_max": vol_grid_max_min,
                "surface_min_max": surf_grid_max_min,
                "length_scale": length_scale,
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), -1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), -1
                ),
            }


if __name__ == "__main__":
    fm_data = DoMINODataPipe(
        data_path="/code/processed_data/new_models_1/",
        phase="train",
        sampling=False,
        sample_in_bbox=False,
    )
