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
This code defines a standalone distributed inference pipeline the DoMINO model. 
This inference pipeline can be used to evaluate the model given an STL and
an inflow speed. The pre-trained model checkpoint can be specified in this script
or inferred from the config file. The results are calculated on a point cloud
sampled in the volume around the STL and on the surface of the STL. They are stored
in a dictionary, which can be written out for visualization.
"""

from pathlib import Path
import os
import time
import copy
import apex
import hydra
import re
from hydra import compose, initialize
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader
import vtk
from vtk.util import numpy_support

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import get_filenames, write_to_vtp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.distributed import DistributedManager

from numpy.typing import NDArray
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable
import warp as wp
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
from design_datapipe import DesignDatapipe


def combine_stls(stl_path: str, stl_files: List[str]) -> pv.PolyData:
    """Combines multiple STL files into a single PyVista mesh.

    Args:
        stl_path: Directory path containing the STL files
        stl_files: List of STL filenames to combine

    Returns:
        Combined PyVista PolyData mesh containing all STL geometries
    """
    meshes = []
    for file in stl_files:
        if ".stl" in file:
            stl_file_path = os.path.join(stl_path, file)
            reader = pv.get_reader(stl_file_path)
            mesh_stl = reader.read()
            meshes.append(mesh_stl)
    combined_mesh = pv.merge(meshes)
    return combined_mesh


class DoMINOInference:
    def __init__(
        self,
        cfg: DictConfig,
        dist: None | DistributedManager,
        cached_geo_encoding: bool = False,
    ):

        self.cfg = cfg
        self.dist = dist
        self.stream_velocity = None
        self.stencil_size = None
        self.stl_path = None
        self.stl_vertices = None
        self.stl_centers = None
        self.surface_areas = None
        self.mesh_indices_flattened = None
        self.length_scale = 1.0
        if self.dist is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = self.dist.device

        # self.air_density = torch.full((1, 1), 1.205, dtype=torch.float32).to(
        #     self.device
        # )
        self.air_density = 1.205
        self.num_vol_vars, self.num_surf_vars = self.get_num_variables()
        self.model = None
        self.grid_resolution = torch.tensor(self.cfg.model.interp_res).to(self.device)
        self.vol_factors = None
        self.bounding_box_min_max = None
        self.bounding_box_surface_min_max = None
        self.center_of_mass = None
        self.grid = None
        self.geometry_encoding = None
        self.geometry_encoding_surface = None
        self.cached_geo_encoding = cached_geo_encoding
        self.out_dict = {}

    def get_geometry_encoding(self):
        return self.geometry_encoding

    def get_geometry_encoding_surface(self):
        return self.geometry_encoding_surface

    def get_out_dict(self):
        return self.out_dict

    def clear_out_dict(self):
        self.out_dict.clear()

    # def initialize_data_processor(self):
    #     self.ifp = inferenceDataPipe(
    #         device=self.device,
    #         surface_vertices=self.stl_vertices,
    #         surface_indices=self.mesh_indices_flattened,
    #         surface_areas=self.surface_areas,
    #         surface_centers=self.stl_centers,
    #         grid_resolution=self.grid_resolution,
    #         normalize_coordinates=True,
    #         geom_points_sample=70_000,
    #         positional_encoding=False,
    #         use_sdf_basis=self.cfg.model.use_sdf_in_basis_func,
    #     )

    def load_bounding_box(self):
        if (
            self.cfg.data.bounding_box.min is not None
            and self.cfg.data.bounding_box.max is not None
        ):
            c_min = np.array(self.cfg.data.bounding_box.min, dtype=np.float32)

            c_max = np.array(self.cfg.data.bounding_box.max, dtype=np.float32)

            self.bounding_box_min_max = [c_min, c_max]

        if (
            self.cfg.data.bounding_box_surface.min is not None
            and self.cfg.data.bounding_box_surface.max is not None
        ):
            c_min = np.array(self.cfg.data.bounding_box_surface.min, dtype=np.float32)
            c_max = np.array(self.cfg.data.bounding_box_surface.max, dtype=np.float32)

            self.bounding_box_surface_min_max = [c_min, c_max]

    def load_volume_scaling_factors(self) -> torch.Tensor:
        vol_factors = np.array(
            [
                [2.1508515, 1.0027921, 1.0663894, 1.1288369, 0.05063211, 0.00381244],
                [
                    -1.9028450e00,
                    -1.0032533e00,
                    -1.0505041e00,
                    -1.4412953e00,
                    1.5563720e-18,
                    -2.7427445e-20,
                ],
            ],
            dtype=np.float32,
        )

        vol_factors = torch.from_numpy(vol_factors).to(self.device)

        return vol_factors

    def load_surface_scaling_factors(self) -> torch.Tensor:
        surf_factors = np.array(
            [
                [0.98881036, 0.00550783, 0.00854675, 0.00452144],
                [-2.4203062, -0.00740275, -0.00848471, -0.00448634],
            ],
            dtype=np.float32,
        )

        surf_factors = torch.from_numpy(surf_factors).to(self.device)
        return surf_factors

    def read_stl(self) -> None:
        stl_files = get_filenames(self.stl_path)
        mesh_stl = combine_stls(self.stl_path, stl_files)
        stl_vertices = mesh_stl.points
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_centers = mesh_stl.cell_centers().points
        # Assuming triangular elements
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
        mesh_indices_flattened = stl_faces.flatten()

        surface_areas = mesh_stl.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        surface_areas = np.array(surface_areas.cell_data["Area"])

        surface_normals = np.array(mesh_stl.cell_normals, dtype=np.float32)

        self.stl_vertices = torch.from_numpy(np.float32(stl_vertices)).to(self.device)
        self.stl_centers = torch.from_numpy(np.float32(stl_centers)).to(self.device)
        self.surface_areas = torch.from_numpy(np.float32(surface_areas)).to(self.device)
        self.stl_normals = -1.0 * torch.from_numpy(np.float32(surface_normals)).to(
            self.device
        )
        self.mesh_indices_flattened = torch.from_numpy(
            np.int32(mesh_indices_flattened)
        ).to(self.device)
        self.length_scale = length_scale
        self.mesh_stl = mesh_stl

    def read_stl_trimesh(
        self, stl_vertices, stl_faces, stl_centers, surface_normals, surface_areas
    ) -> None:
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        self.stl_vertices = torch.from_numpy(stl_vertices).to(self.device)
        self.stl_centers = torch.from_numpy(stl_centers).to(self.device)
        self.stl_normals = -1.0 * torch.from_numpy(surface_normals).to(self.device)
        self.surface_areas = torch.from_numpy(surface_areas).to(self.device)
        self.mesh_indices_flattened = torch.from_numpy(
            np.int32(mesh_indices_flattened)
        ).to(self.device)
        self.length_scale = length_scale

    def set_datapipe(
        self,
    ) -> None:
        fd = DesignDatapipe(
            self.mesh_stl,
            self.bounding_box_min_max,
            self.bounding_box_surface_min_max,
            grid_resolution=cfg.model.interp_res,
            stream_velocity=self.stream_velocity,
            air_density=self.air_density,
            device=self.device,
        )
        self.train_dataloader = DataLoader(fd, batch_size=8_000, shuffle=False)
        self.input_dict = fd.out_dict

    def get_num_variables(self) -> tuple[int, int]:
        volume_variable_names = list(self.cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if self.cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1

        surface_variable_names = list(self.cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if self.cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
        return num_vol_vars, num_surf_vars

    def initialize_model(self, model_path: str) -> None:
        model = (
            DoMINO(
                input_features=3,
                output_features_vol=self.num_vol_vars,
                output_features_surf=self.num_surf_vars,
                model_parameters=self.cfg.model,
            ).to(self.device)
            # .eval()
        )
        model = torch.compile(model, disable=True)

        checkpoint_iter = torch.load(
            to_absolute_path(model_path), map_location=self.dist.device
        )

        model.load_state_dict(checkpoint_iter)
        print("model loaded ...")

        if self.dist is not None:
            if self.dist.world_size > 1:
                model = DistributedDataParallel(
                    model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                )

        self.model = model
        self.vol_factors = self.load_volume_scaling_factors()
        self.surf_factors = self.load_surface_scaling_factors()
        self.load_bounding_box()

    def set_stream_velocity(self, stream_velocity):
        self.stream_velocity = stream_velocity

    def set_stencil_size(self, stencil_size):
        self.stencil_size = stencil_size

    def set_air_density(self, air_density):
        self.air_density = air_density

    def set_stl_path(self, filename):
        self.stl_path = filename

    def compute_sensitivities(self, target_force=300):
        self.input_dict = {
            key: torch.from_numpy(np.expand_dims(np.float32(value), 0))
            for key, value in self.input_dict.items()
        }
        # input_dict = dict_to_device(self.input_dict, self.device)
        input_dict = copy.deepcopy(self.input_dict)
        for param in self.model.parameters():
            param.requires_grad = False

        # optimizer = apex.optimizers.FusedAdam([input_dict["geometry_coordinates"]], lr=0.001)

        # print(input_dict["geometry_coordinates"].shape)
        input_dict = dict_to_device(input_dict, self.device)
        input_dict["geometry_coordinates"].requires_grad_(True)
        for i_batch, sample_batched in enumerate(self.train_dataloader):
            sample_batched = {
                key: torch.unsqueeze(value, 0) for key, value in sample_batched.items()
            }
            sampled_batched = dict_to_device(sample_batched, self.device)
            input_dict["surface_mesh_centers"] = sampled_batched["surface_mesh_centers"]
            input_dict["surface_mesh_neighbors"] = sampled_batched[
                "surface_mesh_neighbors"
            ]
            input_dict["surface_normals"] = sampled_batched["surface_normals"]
            input_dict["surface_neighbors_normals"] = sampled_batched[
                "surface_neighbors_normals"
            ]
            input_dict["surface_areas"] = sampled_batched["surface_areas"]
            input_dict["surface_neighbors_areas"] = sampled_batched[
                "surface_neighbors_areas"
            ]
            input_dict["pos_surface_center_of_mass"] = sampled_batched[
                "pos_surface_center_of_mass"
            ]

            # input_dict_on_device = dict_to_device(input_dict, self.device)
            # print(self.input_dict["geometry_coordinates"])
            # print(self.input_dict["geometry_coordinates"].requires_grad)

            print(
                f"Allocated memory after data loading: {(torch.cuda.memory_allocated()/(1024**3)):.2f} GB"
            )
            # import pdb
            # pdb.set_trace()
            # print(sampled_batched["geo"])
            # print(input_dict["geometry_coordinates"].requires_grad)
            # input_dict["geometry_coordinates"].requires_grad_(True)
            # print(input_dict["geometry_coordinates"].requires_grad, input_dict["geometry_coordinates"].shape)
            with autocast(enabled=True):
                prediction_vol, prediction_surf = self.model(input_dict)

                print(
                    f"Allocated memory after model eval: {(torch.cuda.memory_allocated()/(1024**3)):.2f} GB"
                )
                # print(prediction_vol.shape, prediction_surf.shape)
                stream_velocity = input_dict["stream_velocity"]
                air_density = input_dict["air_density"]
                # print(stream_velocity, air_density)
                prediction_surf = (
                    unnormalize(
                        prediction_surf, self.surf_factors[0], self.surf_factors[1]
                    )
                    * stream_velocity[0, 0] ** 2.0
                    * air_density[0, 0]
                )
                surface_normals = input_dict["surface_normals"]
                surface_sizes = torch.unsqueeze(input_dict["surface_areas"], -1)
                # # print(surface_normals.shape, surface_sizes.shape, prediction_surf.shape)
                d_force = torch.sum(
                    prediction_surf[0, :, 0]
                    * surface_normals[0, :, 0]
                    * surface_sizes[0, :, 0]
                    - prediction_surf[0, :, 1] * surface_sizes[0, :, 0]
                )
                # print(d_force.grad)
                if i_batch == 0:
                    drag_force = d_force
                else:
                    drag_force += d_force
                # if i_batch == 20:
                #     break
                print(drag_force, d_force, prediction_surf.shape, (i_batch + 1) * 8000)
                # print(drag_force)

            if i_batch == 0:
                pred_surf = prediction_surf[0, :].detach().cpu().numpy()
                surface_areas = np.expand_dims(
                    input_dict["surface_areas"][0].detach().cpu().numpy(), -1
                )
                # print(surface_areas.shape)
            else:
                surface_areas1 = np.expand_dims(
                    input_dict["surface_areas"][0].detach().cpu().numpy(), -1
                )
                surface_areas = np.concatenate((surface_areas, surface_areas1), 0)

                pred_surf1 = prediction_surf[0, :].detach().cpu().numpy()
                pred_surf = np.concatenate((pred_surf, pred_surf1), 0)

        loss = torch.square(drag_force - 420.0) / 420**2.0
        print(
            f"Allocated memory after loss calc: {(torch.cuda.memory_allocated()/(1024**3)):.2f} GB"
        )
        loss.backward()
        print(loss)
        print(input_dict["geometry_coordinates"].grad.shape)
        return (
            input_dict["geometry_coordinates"].grad.cpu().detach().numpy(),
            input_dict["geometry_coordinates"].detach().cpu().numpy(),
            surface_areas,
            pred_surf,
        )


if __name__ == "__main__":
    OmegaConf.register_new_resolver(name="eval", resolver=eval, replace=True)
    with initialize(version_base="1.3", config_path="conf"):
        cfg = compose(config_name="config")

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.world_size > 1:
        torch.distributed.barrier()

    input_path = Path("./geometries")
    dirnames = get_filenames(input_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / 1)
    dirnames_per_gpu = (
        dirnames  # [int(num_files * dev_id) : int(num_files * (dev_id + 1))]
    )

    domino = DoMINOInference(cfg, dist, False)
    domino.initialize_model(model_path="./DoMINO.0.0.pt")
    for count, dirname in enumerate(dirnames_per_gpu):
        # print(f"Processing file {dirname}")
        filepath = os.path.join(input_path, dirname)

        STREAM_VELOCITY = 38.889
        AIR_DENSITY = 1.205

        # Neighborhood points sampled for evaluation, tradeoff between accuracy and speed
        STENCIL_SIZE = (
            7  # Higher stencil size -> more accuracy but more evaluation time
        )

        domino.set_stl_path(filepath)
        domino.set_stream_velocity(STREAM_VELOCITY)
        domino.set_stencil_size(STENCIL_SIZE)

        domino.read_stl()

        domino.set_datapipe()

        # Calculate sensitivities
        sensitivities, coordinates, areas, prediction = domino.compute_sensitivities(
            target_force=350
        )
        sensitivities = sensitivities[0]
        coordinates = coordinates[0]
        # areas = np.expand_dims(areas[0], -1)
        print("areas:", areas.shape, sensitivities.shape)
        # sensitivities = sensitivities * areas
        print(sensitivities.shape, coordinates.shape)
        interp_func = KDTree(coordinates)
        dd, ii = interp_func.query(coordinates, k=10)

        for _ in range(10):
            sensitivities_neighbors = sensitivities[ii]
            sensitivities = np.mean(sensitivities_neighbors, 1)
            print(np.amax(sensitivities, 0), np.amin(sensitivities, 0))
        print(sensitivities.shape)
        # print(dd.shape)

        # all_data = np.concatenate((coordinates, sensitivities), axis=-1)
        # header = "X-coordinate, Y-coordinate, Z-coordinate, X-sensitivity, Y-sensitivity, Z-sensitivity"
        # np.savetxt(f"/lustre/rranade/modulus_dev/modulus_demo/modulus_rishi/modulus/examples/cfd/external_aerodynamics/domino_gtc_demo/sensitivity_pred_{dirname}.csv", all_data[0], comments=" ", delimiter=",", header=header)
        vtp_path = f"/lustre/rranade/modulus_dev/modulus_demo/modulus_rishi/modulus/examples/cfd/external_aerodynamics/domino_gtc_demo/sensitivity_pred_{dirname}_3.vtp"
        domino.mesh_stl.save(vtp_path)
        # vtp_path = f"/lustre/rranade/modulus_dev/modulus_demo/modulus_rishi/modulus/examples/cfd/external_aerodynamics/domino_gtc_demo/sensitivity_pred_{dirname}.vtp"
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f"{vtp_path}")
        reader.Update()
        polydata_surf = reader.GetOutput()

        surfParam_vtk = numpy_support.numpy_to_vtk(sensitivities[:, 0:3])
        surfParam_vtk.SetName(f"Sensitivity")
        polydata_surf.GetCellData().AddArray(surfParam_vtk)

        surfParam_vtk = numpy_support.numpy_to_vtk(prediction[:, 0:1])
        surfParam_vtk.SetName(f"Pressure")
        polydata_surf.GetCellData().AddArray(surfParam_vtk)

        surfParam_vtk = numpy_support.numpy_to_vtk(prediction[:, 1:])
        surfParam_vtk.SetName(f"Wall-shear-stress")
        polydata_surf.GetCellData().AddArray(surfParam_vtk)

        # surfParam_vtk = numpy_support.numpy_to_vtk(sensitivities[0, :, 1:2])
        # surfParam_vtk.SetName(f"Sensitivity-x")
        # polydata_surf.AddArray(surfParam_vtk)

        # surfParam_vtk = numpy_support.numpy_to_vtk(sensitivities[0, :, 2:3])
        # surfParam_vtk.SetName(f"Sensitivity-x")
        # polydata_surf.AddArray(surfParam_vtk)

        write_to_vtp(polydata_surf, vtp_path)
        # domino.initialize_data_processor()

        # Calculate sensitivities
        # domino.compute_sensitivities(target_force=350)

        # Calculate geometry encoding
        # domino.compute_geo_encoding()

        # # Calculate volume solutions
        # domino.compute_volume_solutions(
        #     num_sample_points=10_256_000, plot_solutions=False
        # )

        # Calculate surface solutions
        # domino.compute_surface_solutions()
        # domino.compute_forces()
        # out_dict = domino.get_out_dict()

        # print(
        #     "Dirname:",
        #     dirname,
        #     "Drag:",
        #     out_dict["drag_force"],
        #     "Lift:",
        #     out_dict["lift_force"],
        # )
