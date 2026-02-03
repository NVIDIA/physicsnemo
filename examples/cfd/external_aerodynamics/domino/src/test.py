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

# This will set up the cupy-ecosystem and pytorch to share memory pools
from physicsnemo.utils.memory import unified_gpu_memory

import numpy as np
import cupy as cp

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
from physicsnemo.models.domino.geometry_rep import scale_sdf
from physicsnemo.models.domino.utils import *
from physicsnemo.models.domino.utils.vtk_file_utils import *
from physicsnemo.nn.functional import knn, signed_distance_field
from utils import ScalingFactors, load_scaling_factors

# AIR_DENSITY = 1.205
# STREAM_VELOCITY = 30.00


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
        # data_dict = dict_to_device(data_dict, device)

        # Non-dimensionalization factors
        length_scale = data_dict["length_scale"]

        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]
        stream_velocity = global_params_reference[:, 0, :]
        air_density = global_params_reference[:, 1, :]

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
            if "volume_min_max" in data_dict.keys():
                vol_max = data_dict["volume_min_max"][:, 1]
                vol_min = data_dict["volume_min_max"][:, 0]
                geo_centers_vol = (
                    2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
                )
            else:
                geo_centers_vol = geo_centers

            # Normalize based on computational domain
            encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

        if output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        if (
            output_features_vol is not None
            and output_features_surf is not None
            and cfg.model.combine_volume_surface
        ):
            encoding_g = torch.cat((encoding_g_vol, encoding_g_surf), axis=1)
            encoding_g_surf = model.combined_unet_surf(encoding_g)
            encoding_g_vol = model.combined_unet_vol(encoding_g)

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

            prediction_vol = torch.zeros_like(target_vol)
            num_points = volume_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))
            sdf_scaling_factor = (
                cfg.model.geometry_rep.geo_processor.volume_sdf_scaling_factor
            )
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
                    scaled_sdf_nodes_batch = []
                    for p in range(len(sdf_scaling_factor)):
                        scaled_sdf_nodes_batch.append(
                            scale_sdf(sdf_nodes_batch, sdf_scaling_factor[p])
                        )
                    scaled_sdf_nodes_batch = torch.cat(scaled_sdf_nodes_batch, dim=-1)

                    pos_volume_closest_batch = pos_volume_closest[:, start_idx:end_idx]
                    pos_normals_com_batch = pos_volume_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.volume_local_geo_encodings(
                        0.5 * encoding_g_vol,
                        volume_mesh_centers_batch,
                        p_grid,
                    )
                    if cfg.model.use_sdf_in_basis_func:
                        pos_encoding_all = torch.cat(
                            (
                                sdf_nodes_batch,
                                scaled_sdf_nodes_batch,
                                pos_volume_closest_batch,
                                pos_normals_com_batch,
                            ),
                            axis=-1,
                        )
                    else:
                        pos_encoding_all = pos_normals_com_batch

                    pos_encoding = model.fc_p_vol(pos_encoding_all)
                    tpredictions_batch = model.solution_calculator_vol(
                        volume_mesh_centers_batch,
                        geo_encoding_local,
                        pos_encoding,
                        global_params_values,
                        global_params_reference,
                    )
                    running_tloss_vol += loss_fn(tpredictions_batch, target_batch)
                    prediction_vol[:, start_idx:end_idx] = tpredictions_batch

            if cfg.model.normalization == "min_max_scaling":
                prediction_vol = unnormalize(
                    prediction_vol, vol_factors[0], vol_factors[1]
                )
            elif cfg.model.normalization == "mean_std_scaling":
                prediction_vol = unstandardize(
                    prediction_vol, vol_factors[0], vol_factors[1]
                )
            # print(np.amax(prediction_vol, axis=(0, 1)), np.amin(prediction_vol, axis=(0, 1)))

            prediction_vol[:, :, :3] = prediction_vol[:, :, :3] * stream_velocity[0, 0]
            prediction_vol[:, :, 3] = (
                prediction_vol[:, :, 3]
                * stream_velocity[0, 0] ** 2.0
                * air_density[0, 0]
            )
            prediction_vol[:, :, 4] = (
                prediction_vol[:, :, 4] * stream_velocity[0, 0] * length_scale[0]
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
            prediction_surf = torch.zeros_like(target_surf)

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
                    geo_encoding_local = model.surface_local_geo_encodings(
                        0.5 * encoding_g_surf,
                        surface_mesh_centers_batch,
                        s_grid,
                    )
                    pos_encoding = model.fc_p_surf(pos_surface_center_of_mass_batch)

                    tpredictions_batch = model.solution_calculator_surf(
                        surface_mesh_centers_batch,
                        geo_encoding_local,
                        pos_encoding,
                        surface_mesh_neighbors_batch,
                        surface_normals_batch,
                        surface_neighbors_normals_batch,
                        surface_areas_batch,
                        surface_neighbors_areas_batch,
                        global_params_values,
                        global_params_reference,
                    )

                    running_tloss_surf += loss_fn(tpredictions_batch, target_batch)
                    prediction_surf[:, start_idx:end_idx] = tpredictions_batch

            if cfg.model.normalization == "min_max_scaling":
                prediction_surf = unnormalize(
                    prediction_surf, surf_factors[0], surf_factors[1]
                )
            elif cfg.model.normalization == "mean_std_scaling":
                prediction_surf = unstandardize(
                    prediction_surf, surf_factors[0], surf_factors[1]
                )
            prediction_surf = (
                prediction_surf * stream_velocity[0, 0] ** 2.0 * air_density[0, 0]
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

    global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        if cfg.variables.global_parameters[param].type == "vector":
            global_features += len(cfg.variables.global_parameters[param].reference)
        else:
            global_features += 1

    ######################################################
    # Get scaling factors - precompute them if this fails!
    ######################################################
    pickle_path = os.path.join(cfg.data.scaling_factors)

    vol_factors, surf_factors = load_scaling_factors(cfg)
    print("Vol factors:", vol_factors)
    print("Surf factors:", surf_factors)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=global_features,
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

    if dist.rank == 0:
        create_directory(pred_save_path)

    l2_surface_all = []
    l2_volume_all = []
    aero_forces_all = []
    for count, dirname in enumerate(dirnames_per_gpu):
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
        length_scale = np.array(
            np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0)),
            dtype=np.float32,
        )
        length_scale = torch.from_numpy(length_scale).to(torch.float32).to(dist.device)
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

        # Convert to torch tensors and load on device
        stl_vertices = torch.from_numpy(stl_vertices).to(torch.float32).to(dist.device)
        stl_sizes = torch.from_numpy(stl_sizes).to(torch.float32).to(dist.device)
        stl_centers = torch.from_numpy(stl_centers).to(torch.float32).to(dist.device)
        mesh_indices_flattened = (
            torch.from_numpy(mesh_indices_flattened).to(torch.int32).to(dist.device)
        )

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        s_max = (
            torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.max))
            .to(torch.float32)
            .to(dist.device)
        )
        s_min = (
            torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.min))
            .to(torch.float32)
            .to(dist.device)
        )

        nx, ny, nz = cfg.model.interp_res

        surf_grid = create_grid(
            s_max, s_min, torch.from_numpy(np.asarray([nx, ny, nz])).to(dist.device)
        )

        normed_stl_vertices_cp = normalize(stl_vertices, s_max, s_min)
        surf_grid_normed = normalize(surf_grid, s_max, s_min)

        # SDF calculation on the grid using WARP
        time_start = time.time()
        sdf_surf_grid, _ = signed_distance_field(
            normed_stl_vertices_cp,
            mesh_indices_flattened,
            surf_grid_normed,
            use_sign_winding_number=True,
        )

        surf_grid_max_min = torch.stack([s_min, s_max])

        # Get global parameters and global parameters scaling from config.yaml
        global_params_names = list(cfg.variables.global_parameters.keys())
        global_params_reference = {
            name: cfg.variables.global_parameters[name]["reference"]
            for name in global_params_names
        }
        global_params_types = {
            name: cfg.variables.global_parameters[name]["type"]
            for name in global_params_names
        }
        stream_velocity = global_params_reference["inlet_velocity"][0]
        air_density = global_params_reference["air_density"]

        # Arrange global parameters reference in a list, ensuring it is flat
        global_params_reference_list = []
        for name, type in global_params_types.items():
            if type == "vector":
                global_params_reference_list.extend(global_params_reference[name])
            elif type == "scalar":
                global_params_reference_list.append(global_params_reference[name])
            else:
                raise ValueError(
                    f"Global parameter {name} not supported for  this dataset"
                )
        global_params_reference = np.array(
            global_params_reference_list, dtype=np.float32
        )
        global_params_reference = torch.from_numpy(global_params_reference).to(
            dist.device
        )

        # Define the list of global parameter values for each simulation.
        # Note: The user must ensure that the values provided here correspond to the
        # `global_parameters` specified in `config.yaml` and that these parameters
        # exist within each simulation file.
        global_params_values_list = []
        for key in global_params_types.keys():
            if key == "inlet_velocity":
                global_params_values_list.append(stream_velocity)
            elif key == "air_density":
                global_params_values_list.append(air_density)
            else:
                raise ValueError(
                    f"Global parameter {key} not supported for  this dataset"
                )
        global_params_values_list = np.array(
            global_params_values_list, dtype=np.float32
        )
        global_params_values = torch.from_numpy(global_params_values_list).to(
            dist.device
        )

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

            surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )
            surface_coordinates = (
                torch.from_numpy(surface_coordinates).to(torch.float32).to(dist.device)
            )
            surface_normals = (
                torch.from_numpy(surface_normals).to(torch.float32).to(dist.device)
            )
            surface_sizes = (
                torch.from_numpy(surface_sizes).to(torch.float32).to(dist.device)
            )
            surface_fields = (
                torch.from_numpy(surface_fields).to(torch.float32).to(dist.device)
            )

            if cfg.model.num_neighbors_surface > 1:
                time_start = time.time()
                # print(f"file: {dirname}, surface coordinates shape: {surface_coordinates.shape}")
                # try:
                ii, dd = knn(
                    points=surface_coordinates,
                    queries=surface_coordinates,
                    k=cfg.model.num_neighbors_surface,
                )

                surface_neighbors = surface_coordinates[ii]
                surface_neighbors = surface_neighbors[:, 1:]

                surface_neighbors_normals = surface_normals[ii]
                surface_neighbors_normals = surface_neighbors_normals[:, 1:]
                surface_neighbors_sizes = surface_sizes[ii]
                surface_neighbors_sizes = surface_neighbors_sizes[:, 1:]
                # except:
                #     print(f"file: {dirname}, memory error in knn")
                #     print("setting surface neighbors to 0")
                #     surface_neighbors = surface_coordinates
                #     surface_neighbors_normals = surface_normals
                #     surface_neighbors_sizes = surface_sizes
                #     cfg.model.num_neighbors_surface = 1
            else:
                surface_neighbors = surface_coordinates
                surface_neighbors_normals = surface_normals
                surface_neighbors_sizes = surface_sizes

            if cfg.data.normalize_coordinates:
                surface_coordinates = normalize(surface_coordinates, s_max, s_min)
                surf_grid = normalize(surf_grid, s_max, s_min)
                center_of_mass_normalized = normalize(center_of_mass, s_max, s_min)
                surface_neighbors = normalize(surface_neighbors, s_max, s_min)
            else:
                center_of_mass_normalized = center_of_mass
            pos_surface_center_of_mass = surface_coordinates - center_of_mass_normalized

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
            volume_coordinates = (
                torch.from_numpy(volume_coordinates).to(torch.float32).to(dist.device)
            )
            volume_fields = (
                torch.from_numpy(volume_fields).to(torch.float32).to(dist.device)
            )

            c_max = (
                torch.from_numpy(np.asarray(cfg.data.bounding_box.max))
                .to(torch.float32)
                .to(dist.device)
            )
            c_min = (
                torch.from_numpy(np.asarray(cfg.data.bounding_box.min))
                .to(torch.float32)
                .to(dist.device)
            )

            # Generate a grid of specified resolution to map the bounding box
            # The grid is used for capturing structured geometry features and SDF representation of geometry
            grid = create_grid(
                c_max, c_min, torch.from_numpy(np.asarray([nx, ny, nz])).to(dist.device)
            )

            if cfg.data.normalize_coordinates:
                volume_coordinates = normalize(volume_coordinates, c_max, c_min)
                grid = normalize(grid, c_max, c_min)
                center_of_mass_normalized = normalize(center_of_mass, c_max, c_min)
                normed_stl_vertices_vol = normalize(stl_vertices, c_max, c_min)
            else:
                center_of_mass_normalized = center_of_mass

            # SDF calculation on the grid using WARP
            time_start = time.time()
            sdf_grid, _ = signed_distance_field(
                normed_stl_vertices_vol,
                mesh_indices_flattened,
                grid,
                use_sign_winding_number=True,
            )

            # SDF calculation
            time_start = time.time()
            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                normed_stl_vertices_vol,
                mesh_indices_flattened,
                volume_coordinates,
                use_sign_winding_number=True,
            )
            sdf_nodes = sdf_nodes.reshape(-1, 1)
            vol_grid_max_min = torch.stack([c_min, c_max])

            pos_volume_closest = volume_coordinates - sdf_node_closest_point
            pos_volume_center_of_mass = volume_coordinates - center_of_mass_normalized

        else:
            volume_coordinates = None
            volume_fields = None
            pos_volume_closest = None
            pos_volume_center_of_mass = None

        # print(f"Processed sdf and normalized")

        geom_centers = stl_vertices
        # print(f"Geom centers max: {np.amax(geom_centers, axis=0)}, min: {np.amin(geom_centers, axis=0)}")

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
                "length_scale": length_scale,
                "global_params_values": torch.unsqueeze(global_params_values, -1),
                "global_params_reference": torch.unsqueeze(global_params_reference, -1),
            }
        elif model_type == "surface":
            data_dict = {
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
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
                "global_params_values": torch.unsqueeze(global_params_values, -1),
                "global_params_reference": torch.unsqueeze(global_params_reference, -1),
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
                "length_scale": length_scale,
                "global_params_values": torch.unsqueeze(global_params_values, -1),
                "global_params_reference": torch.unsqueeze(global_params_reference, -1),
            }

        data_dict = {key: torch.unsqueeze(value, 0) for key, value in data_dict.items()}

        prediction_vol, prediction_surf = test_step(
            data_dict, model, dist.device, cfg, vol_factors, surf_factors
        )

        if prediction_surf is not None:
            surface_sizes = torch.unsqueeze(surface_sizes, -1)

            pres_x_pred = torch.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
            )
            shear_x_pred = torch.sum(prediction_surf[0, :, 1] * surface_sizes[:, 0])

            pres_x_true = torch.sum(
                surface_fields[:, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
            )
            shear_x_true = torch.sum(surface_fields[:, 1] * surface_sizes[:, 0])

            force_x_pred = torch.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
                - prediction_surf[0, :, 1] * surface_sizes[:, 0]
            )
            force_x_true = torch.sum(
                surface_fields[:, 0] * surface_normals[:, 0] * surface_sizes[:, 0]
                - surface_fields[:, 1] * surface_sizes[:, 0]
            )

            force_y_pred = torch.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 1] * surface_sizes[:, 0]
                - prediction_surf[0, :, 2] * surface_sizes[:, 0]
            )
            force_y_true = torch.sum(
                surface_fields[:, 0] * surface_normals[:, 1] * surface_sizes[:, 0]
                - surface_fields[:, 2] * surface_sizes[:, 0]
            )

            force_z_pred = torch.sum(
                prediction_surf[0, :, 0] * surface_normals[:, 2] * surface_sizes[:, 0]
                - prediction_surf[0, :, 3] * surface_sizes[:, 0]
            )
            force_z_true = torch.sum(
                surface_fields[:, 0] * surface_normals[:, 2] * surface_sizes[:, 0]
                - surface_fields[:, 3] * surface_sizes[:, 0]
            )
            print(
                "Drag=", dirname, force_x_pred.cpu().numpy(), force_x_true.cpu().numpy()
            )
            print(
                "Lift=", dirname, force_z_pred.cpu().numpy(), force_z_true.cpu().numpy()
            )
            print(
                "Side=", dirname, force_y_pred.cpu().numpy(), force_y_true.cpu().numpy()
            )
            aero_forces_all.append(
                [
                    dirname,
                    force_x_pred,
                    force_x_true,
                    force_z_pred,
                    force_z_true,
                    force_y_pred,
                    force_y_true,
                ]
            )

            l2_gt = torch.mean(torch.square(surface_fields), (0))
            l2_error = torch.mean(
                torch.square(prediction_surf[0] - surface_fields), (0)
            )
            l2_surface_all.append(
                np.sqrt(l2_error.cpu().numpy()) / np.sqrt(l2_gt.cpu().numpy())
            )

            print(
                "Surface L-2 norm:",
                dirname,
                np.sqrt(l2_error.cpu().numpy()) / np.sqrt(l2_gt.cpu().numpy()),
            )

        if prediction_vol is not None:
            target_vol = volume_fields
            prediction_vol = prediction_vol[0]
            c_min = vol_grid_max_min[0]
            c_max = vol_grid_max_min[1]
            volume_coordinates = unnormalize(volume_coordinates, c_max, c_min)
            ids_in_bbox = torch.where(
                (volume_coordinates[:, 0] < c_min[0])
                | (volume_coordinates[:, 0] > c_max[0])
                | (volume_coordinates[:, 1] < c_min[1])
                | (volume_coordinates[:, 1] > c_max[1])
                | (volume_coordinates[:, 2] < c_min[2])
                | (volume_coordinates[:, 2] > c_max[2])
            )
            target_vol[ids_in_bbox] = 0.0
            prediction_vol[ids_in_bbox] = 0.0
            l2_gt = torch.mean(torch.square(target_vol), (0))
            l2_error = torch.mean(torch.square(prediction_vol - target_vol), (0))
            print(
                "Volume L-2 norm:",
                dirname,
                np.sqrt(l2_error.cpu().numpy()) / np.sqrt(l2_gt.cpu().numpy()),
            )
            l2_volume_all.append(
                np.sqrt(l2_error.cpu().numpy()) / np.sqrt(l2_gt.cpu().numpy())
            )

        # import pdb; pdb.set_trace()
        if prediction_surf is not None:
            surfParam_vtk = numpy_support.numpy_to_vtk(
                prediction_surf[0, :, 0:1].cpu().numpy()
            )
            surfParam_vtk.SetName(f"{surface_variable_names[0]}Pred")
            celldata_all.GetCellData().AddArray(surfParam_vtk)

            surfParam_vtk = numpy_support.numpy_to_vtk(
                prediction_surf[0, :, 1:].cpu().numpy()
            )
            surfParam_vtk.SetName(f"{surface_variable_names[1]}Pred")
            celldata_all.GetCellData().AddArray(surfParam_vtk)

            write_to_vtp(celldata_all, vtp_pred_save_path)

        if prediction_vol is not None:
            volParam_vtk = numpy_support.numpy_to_vtk(
                prediction_vol[:, 0:3].cpu().numpy()
            )
            volParam_vtk.SetName(f"{volume_variable_names[0]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(
                prediction_vol[:, 3:4].cpu().numpy()
            )
            volParam_vtk.SetName(f"{volume_variable_names[1]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(
                prediction_vol[:, 4:5].cpu().numpy()
            )
            volParam_vtk.SetName(f"{volume_variable_names[2]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            write_to_vtu(polydata_vol, vtu_pred_save_path)

    l2_surface_all = np.asarray(l2_surface_all)  # num_files, 4
    l2_volume_all = np.asarray(l2_volume_all)  # num_files, 4
    l2_surface_mean = np.mean(l2_surface_all, 0)
    l2_volume_mean = np.mean(l2_volume_all, 0)
    print(
        f"Mean over all samples, surface={l2_surface_mean} and volume={l2_volume_mean}"
    )


if __name__ == "__main__":
    main()
