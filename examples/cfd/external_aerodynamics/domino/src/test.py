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
This code defines a distributed pipeline for testing the DoMINO model on
CFD datasets. It includes the instantiating the DoMINO model and datapipe, 
automatically loading the most recent checkpoint, reading the VTP/VTU/STL 
testing files, calculation of parameters required for DoMINO model and 
evaluating the model in parallel using DistributedDataParallel across multiple 
GPUs. This is a common recipe that enables training of combined models for surface 
and volume as well either of them separately. The model predictions are loaded in 
the the VTP/VTU files and saved in the specified directory. The eval tab in 
config.yaml can be used to specify the input and output directories.
"""

import os, re
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import pandas as pd
import pyvista as pv

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

import vtk
from vtk.util import numpy_support

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *
from physicsnemo.utils.sdf import signed_distance_field

AIR_DENSITY = 1.205
STREAM_VELOCITY = 30.00


def loss_fn(output, target):
    masked_loss = torch.mean(((output - target) ** 2.0), (0, 1, 2))
    loss = torch.mean(masked_loss)
    return loss


def test_step(data_dict, model, device, cfg, vol_factors, surf_factors):
    avg_tloss_vol = 0.0
    avg_tloss_surf = 0.0
    running_tloss_vol = 0.0
    running_tloss_surf = 0.0

    if cfg.model.model_type == "volume" or cfg.model.model_type == "combined":
        output_features_vol = True
    else:
        output_features_vol = None

    if cfg.model.model_type == "surface" or cfg.model.model_type == "combined":
        output_features_surf = True
    else:
        output_features_surf = None

    with torch.no_grad():
        point_batch_size = 256000
        data_dict = dict_to_device(data_dict, device)

        # Non-dimensionalization factors
        air_density = data_dict["air_density"]
        stream_velocity = data_dict["stream_velocity"]
        length_scale = data_dict["length_scale"]

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        if output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]
            # Scaling factors
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]

            # Normalize based on computational domain
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
            encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

            # Normalize based on BBox around surface (car)
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.geo_rep_surface1(
                geo_centers_surf, s_grid, sdf_surf_grid
            )
            encoding_g_vol += encoding_g_surf

        if output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        if output_features_vol is not None:
            # First calculate volume predictions if required
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            target_vol = data_dict["volume_fields"]
            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
            p_grid = data_dict["grid"]

            prediction_vol = np.zeros_like(target_vol.cpu().numpy())
            num_points = volume_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            start_time = time.time()

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with torch.no_grad():
                    target_batch = target_vol[:, start_idx:end_idx]
                    volume_mesh_centers_batch = volume_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    sdf_nodes_batch = sdf_nodes[:, start_idx:end_idx]
                    pos_volume_closest_batch = pos_volume_closest[:, start_idx:end_idx]
                    pos_normals_com_batch = pos_volume_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.geo_encoding_local(
                        0.5 * encoding_g_vol,
                        volume_mesh_centers_batch,
                        p_grid,
                        mode="volume",
                    )
                    if cfg.model.use_sdf_in_basis_func:
                        pos_encoding = torch.cat(
                            (
                                sdf_nodes_batch,
                                pos_volume_closest_batch,
                                pos_normals_com_batch,
                            ),
                            axis=-1,
                        )
                    else:
                        pos_encoding = pos_normals_com_batch
                    pos_encoding = model.position_encoder(
                        pos_encoding, eval_mode="volume"
                    )
                    tpredictions_batch = model.calculate_solution(
                        volume_mesh_centers_batch,
                        geo_encoding_local,
                        pos_encoding,
                        stream_velocity,
                        air_density,
                        num_sample_points=cfg.eval.stencil_size,
                        eval_mode="volume",
                    )
                    running_tloss_vol += loss_fn(tpredictions_batch, target_batch)
                    prediction_vol[
                        :, start_idx:end_idx
                    ] = tpredictions_batch.cpu().numpy()

            prediction_vol = unnormalize(prediction_vol, vol_factors[0], vol_factors[1])

            prediction_vol[:, :, :3] = (
                prediction_vol[:, :, :3] * stream_velocity[0, 0].cpu().numpy()
            )
            prediction_vol[:, :, 3] = (
                prediction_vol[:, :, 3]
                * stream_velocity[0, 0].cpu().numpy() ** 2.0
                * air_density[0, 0].cpu().numpy()
            )
            prediction_vol[:, :, 4] = (
                prediction_vol[:, :, 4]
                * stream_velocity[0, 0].cpu().numpy()
                * length_scale[0].cpu().numpy()
            )
        else:
            prediction_vol = None

        if output_features_surf is not None:
            # Next calculate surface predictions
            # Sampled points on surface
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = torch.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            num_points = surface_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            target_surf = data_dict["surface_fields"]
            prediction_surf = np.zeros_like(target_surf.cpu().numpy())

            start_time = time.time()

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with torch.no_grad():
                    target_batch = target_surf[:, start_idx:end_idx]
                    surface_mesh_centers_batch = surface_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    surface_mesh_neighbors_batch = surface_mesh_neighbors[
                        :, start_idx:end_idx
                    ]
                    surface_normals_batch = surface_normals[:, start_idx:end_idx]
                    surface_neighbors_normals_batch = surface_neighbors_normals[
                        :, start_idx:end_idx
                    ]
                    surface_areas_batch = surface_areas[:, start_idx:end_idx]
                    surface_neighbors_areas_batch = surface_neighbors_areas[
                        :, start_idx:end_idx
                    ]
                    pos_surface_center_of_mass_batch = pos_surface_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.geo_encoding_local(
                        0.5 * encoding_g_surf,
                        surface_mesh_centers_batch,
                        s_grid,
                        mode="surface",
                    )
                    pos_encoding = pos_surface_center_of_mass_batch
                    pos_encoding = model.position_encoder(
                        pos_encoding, eval_mode="surface"
                    )

                    if cfg.model.surface_neighbors:
                        tpredictions_batch = model.calculate_solution_with_neighbors(
                            surface_mesh_centers_batch,
                            geo_encoding_local,
                            pos_encoding,
                            surface_mesh_neighbors_batch,
                            surface_normals_batch,
                            surface_neighbors_normals_batch,
                            surface_areas_batch,
                            surface_neighbors_areas_batch,
                            stream_velocity,
                            air_density,
                        )
                    else:
                        tpredictions_batch = model.calculate_solution(
                            surface_mesh_centers_batch,
                            geo_encoding_local,
                            pos_encoding,
                            stream_velocity,
                            air_density,
                            num_sample_points=1,
                            eval_mode="surface",
                        )
                    running_tloss_surf += loss_fn(tpredictions_batch, target_batch)
                    prediction_surf[
                        :, start_idx:end_idx
                    ] = tpredictions_batch.cpu().numpy()

            prediction_surf = (
                unnormalize(prediction_surf, surf_factors[0], surf_factors[1])
                * stream_velocity[0, 0].cpu().numpy() ** 2.0
                * air_density[0, 0].cpu().numpy()
            )

        else:
            prediction_surf = None

    return prediction_vol, prediction_surf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    input_path = cfg.eval.test_path

    model_type = cfg.model.model_type

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    vol_save_path = os.path.join(
        cfg.eval.scaling_param_path, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        cfg.eval.scaling_param_path, "surface_scaling_factors.npy"
    )
    if os.path.exists(vol_save_path):
        vol_factors = np.load(vol_save_path)
    else:
        vol_factors = None

    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
    else:
        surf_factors = None

    print("Vol factors:", vol_factors)
    print("Surf factors:", surf_factors)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device)

    model = torch.compile(model, disable=True)

    checkpoint = torch.load(
        to_absolute_path(os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)),
        map_location=dist.device,
    )

    model.load_state_dict(checkpoint)

    print("Model loaded")

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        model = model.module

    dirnames = get_filenames(input_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / dist.world_size)
    dirnames_per_gpu = dirnames[int(num_files * dev_id) : int(num_files * (dev_id + 1))]

    pred_save_path = cfg.eval.save_path
    create_directory(pred_save_path)

    for count, dirname in enumerate(dirnames_per_gpu):
        # print(f"Processing file {dirname}")
        filepath = os.path.join(input_path, dirname)
        tag = int(re.findall(r"(\w+?)(\d+)", dirname)[0][1])
        stl_path = os.path.join(filepath, f"drivaer_{tag}.stl")
        vtp_path = os.path.join(filepath, f"boundary_{tag}.vtp")
        vtu_path = os.path.join(filepath, f"volume_{tag}.vtu")

        vtp_pred_save_path = os.path.join(
            pred_save_path, f"boundary_{tag}_predicted.vtp"
        )
        vtu_pred_save_path = os.path.join(pred_save_path, f"volume_{tag}_predicted.vtu")

        # Read STL
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        if cfg.data.bounding_box_surface is None:
            s_max = np.amax(stl_vertices, 0)
            s_min = np.amin(stl_vertices, 0)
        else:
            bounding_box_dims_surf = []
            bounding_box_dims_surf.append(np.asarray(cfg.data.bounding_box_surface.max))
            bounding_box_dims_surf.append(np.asarray(cfg.data.bounding_box_surface.min))
            s_max = np.float32(bounding_box_dims_surf[0])
            s_min = np.float32(bounding_box_dims_surf[1])

        nx, ny, nz = cfg.model.interp_res

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

        # Read VTP
        if model_type == "surface" or model_type == "combined":
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(vtp_path)
            reader.Update()
            polydata_surf = reader.GetOutput()

            celldata_all = get_node_to_elem(polydata_surf)

            celldata = celldata_all.GetCellData()
            surface_fields = get_fields(celldata, surface_variable_names)
            surface_fields = np.concatenate(surface_fields, axis=-1)

            mesh = pv.PolyData(polydata_surf)
            surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)

            interp_func = KDTree(surface_coordinates)
            dd, ii = interp_func.query(surface_coordinates, k=cfg.eval.stencil_size + 1)

            surface_neighbors = surface_coordinates[ii]
            surface_neighbors = surface_neighbors[:, 1:]

            surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )
            surface_neighbors_normals = surface_normals[ii]
            surface_neighbors_normals = surface_neighbors_normals[:, 1:]
            surface_neighbors_sizes = surface_sizes[ii]
            surface_neighbors_sizes = surface_neighbors_sizes[:, 1:]

            dx, dy, dz = (
                (s_max[0] - s_min[0]) / nx,
                (s_max[1] - s_min[1]) / ny,
                (s_max[2] - s_min[2]) / nz,
            )

            if cfg.model.positional_encoding:
                pos_surface_center_of_mass = calculate_normal_positional_encoding(
                    surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_surface_center_of_mass = surface_coordinates - center_of_mass

            surface_coordinates = normalize(surface_coordinates, s_max, s_min)
            surface_neighbors = normalize(surface_neighbors, s_max, s_min)
            surf_grid = normalize(surf_grid, s_max, s_min)

        else:
            surface_coordinates = None
            surface_fields = None
            surface_sizes = None
            surface_normals = None
            surface_neighbors = None
            surface_neighbors_normals = None
            surface_neighbors_sizes = None
            pos_surface_center_of_mass = None

        # Read VTU
        if model_type == "volume" or model_type == "combined":
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_path)
            reader.Update()
            polydata_vol = reader.GetOutput()
            volume_coordinates, volume_fields = get_volume_data(
                polydata_vol, volume_variable_names
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)
            # print(f"Processed vtu {vtu_path}")

            bounding_box_dims = []
            bounding_box_dims.append(np.asarray(cfg.data.bounding_box.max))
            bounding_box_dims.append(np.asarray(cfg.data.bounding_box.min))

            v_max = np.amax(volume_coordinates, 0)
            v_min = np.amin(volume_coordinates, 0)
            if bounding_box_dims is None:
                c_max = s_max + (s_max - s_min) / 2
                c_min = s_min - (s_max - s_min) / 2
                c_min[2] = s_min[2]
            else:
                c_max = np.float32(bounding_box_dims[0])
                c_min = np.float32(bounding_box_dims[1])

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

            # SDF calculation
            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                volume_coordinates,
                include_hit_points=True,
                use_sign_winding_number=True,
            )
            sdf_nodes = sdf_nodes.numpy().reshape(-1, 1)
            sdf_node_closest_point = sdf_node_closest_point.numpy()

            if cfg.model.positional_encoding:
                pos_volume_closest = calculate_normal_positional_encoding(
                    volume_coordinates, sdf_node_closest_point, cell_length=[dx, dy, dz]
                )
                pos_volume_center_of_mass = calculate_normal_positional_encoding(
                    volume_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_volume_closest = volume_coordinates - sdf_node_closest_point
                pos_volume_center_of_mass = volume_coordinates - center_of_mass

            volume_coordinates = normalize(volume_coordinates, c_max, c_min)
            grid = normalize(grid, c_max, c_min)
            vol_grid_max_min = np.asarray([c_min, c_max])

        else:
            volume_coordinates = None
            volume_fields = None
            pos_volume_closest = None
            pos_volume_center_of_mass = None

        # print(f"Processed sdf and normalized")

        geom_centers = np.float32(stl_vertices)

        if model_type == "combined":
            # Add the parameters to the dictionary
            data_dict = {
                "pos_volume_closest": pos_volume_closest,
                "pos_volume_center_of_mass": pos_volume_center_of_mass,
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
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
                "length_scale": np.array(length_scale, dtype=np.float32),
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), axis=-1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), axis=-1
                ),
            }
        elif model_type == "surface":
            data_dict = {
                "pos_surface_center_of_mass": np.float32(pos_surface_center_of_mass),
                "geometry_coordinates": np.float32(geom_centers),
                "surf_grid": np.float32(surf_grid),
                "sdf_surf_grid": np.float32(sdf_surf_grid),
                "surface_mesh_centers": np.float32(surface_coordinates),
                "surface_mesh_neighbors": np.float32(surface_neighbors),
                "surface_normals": np.float32(surface_normals),
                "surface_neighbors_normals": np.float32(surface_neighbors_normals),
                "surface_areas": np.float32(surface_sizes),
                "surface_neighbors_areas": np.float32(surface_neighbors_sizes),
                "surface_fields": np.float32(surface_fields),
                "surface_min_max": np.float32(surf_grid_max_min),
                "length_scale": np.array(length_scale, dtype=np.float32),
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), axis=-1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), axis=-1
                ),
            }
        elif model_type == "volume":
            data_dict = {
                "pos_volume_closest": pos_volume_closest,
                "pos_volume_center_of_mass": pos_volume_center_of_mass,
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
                "length_scale": np.array(length_scale, dtype=np.float32),
                "stream_velocity": np.expand_dims(
                    np.array(STREAM_VELOCITY, dtype=np.float32), axis=-1
                ),
                "air_density": np.expand_dims(
                    np.array(AIR_DENSITY, dtype=np.float32), axis=-1
                ),
            }

        data_dict = {
            key: torch.from_numpy(np.expand_dims(np.float32(value), 0))
            for key, value in data_dict.items()
        }

        prediction_vol, prediction_surf = test_step(
            data_dict, model, dist.device, cfg, vol_factors, surf_factors
        )

        if prediction_surf is not None:
            surface_sizes = np.expand_dims(surface_sizes, -1)

            pres_x_pred = np.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
            )
            shear_x_pred = np.sum(prediction_surf[0, :, 1] * surface_sizes[:, 0])

            pres_x_true = np.sum(
                surface_fields[:, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
            )
            shear_x_true = np.sum(surface_fields[:, 1] * surface_sizes[:, 0])

            force_x_pred = np.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
                - prediction_surf[0, :, 1] * surface_sizes[:, 0]
            )
            force_x_true = np.sum(
                surface_fields[:, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
                - surface_fields[:, 1] * surface_sizes[:, 0]
            )

            force_y_pred = np.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 1] * surface_sizes[:, 0]
                - prediction_surf[0, :, 2] * surface_sizes[:, 0]
            )
            force_y_true = np.sum(
                surface_fields[:, 0] * surface_normals[:, 1] * surface_sizes[:, 0]
                - surface_fields[:, 2] * surface_sizes[:, 0]
            )

            force_z_pred = np.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 2] * surface_sizes[:, 0]
                - prediction_surf[0, :, 3] * surface_sizes[:, 0]
            )
            force_z_true = np.sum(
                surface_fields[:, 0] * surface_normals[:, 2] * surface_sizes[:, 0]
                - surface_fields[:, 3] * surface_sizes[:, 0]
            )
            print("Drag=", dirname, force_x_pred, force_x_true)
            print("Lift=", dirname, force_z_pred, force_z_true)
            print("Side=", dirname, force_y_pred, force_y_true)

            l2_gt = np.sum(np.square(surface_fields), (0))
            l2_error = np.sum(np.square(prediction_surf[0] - surface_fields), (0))

            print(
                "Surface L-2 norm:",
                dirname,
                np.sqrt(l2_error) / np.sqrt(l2_gt),
            )

        if prediction_vol is not None:
            target_vol = volume_fields
            prediction_vol = prediction_vol[0]
            c_min = vol_grid_max_min[0]
            c_max = vol_grid_max_min[1]
            volume_coordinates = unnormalize(volume_coordinates, c_max, c_min)
            ids_in_bbox = np.where(
                (volume_coordinates[:, 0] < c_min[0])
                | (volume_coordinates[:, 0] > c_max[0])
                | (volume_coordinates[:, 1] < c_min[1])
                | (volume_coordinates[:, 1] > c_max[1])
                | (volume_coordinates[:, 2] < c_min[2])
                | (volume_coordinates[:, 2] > c_max[2])
            )
            target_vol[ids_in_bbox] = 0.0
            prediction_vol[ids_in_bbox] = 0.0
            l2_gt = np.sum(np.square(target_vol), (0))
            l2_error = np.sum(np.square(prediction_vol - target_vol), (0))
            print(
                "Volume L-2 norm:",
                dirname,
                np.sqrt(l2_error) / np.sqrt(l2_gt),
            )

        if prediction_surf is not None:
            surfParam_vtk = numpy_support.numpy_to_vtk(prediction_surf[0, :, 0:1])
            surfParam_vtk.SetName(f"{surface_variable_names[0]}Pred")
            celldata_all.GetCellData().AddArray(surfParam_vtk)

            surfParam_vtk = numpy_support.numpy_to_vtk(prediction_surf[0, :, 1:])
            surfParam_vtk.SetName(f"{surface_variable_names[1]}Pred")
            celldata_all.GetCellData().AddArray(surfParam_vtk)

            write_to_vtp(celldata_all, vtp_pred_save_path)

        if prediction_vol is not None:

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 0:3])
            volParam_vtk.SetName(f"{volume_variable_names[0]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 3:4])
            volParam_vtk.SetName(f"{volume_variable_names[1]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 4:5])
            volParam_vtk.SetName(f"{volume_variable_names[2]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            write_to_vtu(polydata_vol, vtu_pred_save_path)


if __name__ == "__main__":
    main()
