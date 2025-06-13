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

from functools import cached_property
from pathlib import Path
import os
import hydra
from omegaconf import DictConfig

import numpy as np
import torch

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import unnormalize
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.distributed import DistributedManager

from numpy.typing import NDArray
import pyvista as pv
from design_datapipe import DesignDatapipe
from dataclasses import dataclass


def combine_stls(stl_path: str, stl_files: list[str]) -> pv.PolyData:
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


@dataclass
class DoMINOInference:

    cfg: DictConfig
    model_checkpoint_path: Path | str | None = None
    dist: DistributedManager | None = None
    device: torch.device | None = None  # If not set, default set in __post_init__
    model: torch.nn.Module | None = None  # If not set, constructed in __post_init__

    def __post_init__(self):
        if self.device is None:  # Sets a default device, if not specified
            if self.dist is not None:
                self.device = self.dist.device
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        if self.model is None:
            self.model = DoMINO(
                input_features=3,
                output_features_vol=self.num_vol_vars,
                output_features_surf=self.num_surf_vars,
                model_parameters=self.cfg.model,
            ).to(self.device).eval()
    
            for param in self.model.parameters():
                param.requires_grad = False

            self.model = torch.compile(self.model, disable=True)  # TODO review

            if self.model_checkpoint_path is not None:
                with open(self.model_checkpoint_path, "rb") as f:
                    self.model.load_state_dict(
                        torch.load(f, map_location=self.device)
                    )
                print("Model loaded with checkpoint...")
            else:
                print("Model loaded without checkpoint...")

            if (self.dist is not None) and (self.dist.world_size > 1):
                    self.model = DistributedDataParallel(
                        self.model,
                        device_ids=[self.dist.local_rank],
                        output_device=self.dist.device,
                        broadcast_buffers=self.dist.broadcast_buffers,
                        find_unused_parameters=self.dist.find_unused_parameters,
                        gradient_as_bucket_view=True,
                        static_graph=True,
                    )

    @cached_property
    def num_vol_vars(self) -> int:
        return sum(
            3 if v == "vector" else 1
            for k, v in self.cfg.variables.volume.solution.items()
        )

    @cached_property
    def num_surf_vars(self) -> int:
        return sum(
            3 if v == "vector" else 1
            for k, v in self.cfg.variables.surface.solution.items()
        )

    @cached_property
    def bounding_box_min_max(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get the minimum and maximum coordinates of the bounding box from config.
        
        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32]]: Min and max coordinates
            
        Raises:
            ValueError: If min or max coordinates are not specified in config
        """
        try:
            return (
                np.array(self.cfg.data.bounding_box.min, dtype=np.float32),
                np.array(self.cfg.data.bounding_box.max, dtype=np.float32)
            )
        except AttributeError:
            raise ValueError("Config must specify both `bounding_box.min` and `bounding_box.max`")

    @cached_property
    def bounding_box_surface_min_max(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get the minimum and maximum coordinates of the surface bounding box from config.
        
        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32]]: Min and max coordinates
            
        Raises:
            ValueError: If min or max coordinates are not specified in config
        """
        try:
            return (
                np.array(self.cfg.data.bounding_box_surface.min, dtype=np.float32),
                np.array(self.cfg.data.bounding_box_surface.max, dtype=np.float32)
            )
        except AttributeError:
            raise ValueError("Config must specify both `bounding_box_surface.min` and `bounding_box_surface.max`")

    @cached_property
    def vol_factors(self) -> torch.Tensor:
        return torch.from_numpy(
            np.array(
                [
                    [
                        2.1508515,
                        1.0027921,
                        1.0663894,
                        1.1288369,
                        0.05063211,
                        0.00381244,
                    ],
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
        ).to(self.device)

    @cached_property
    def surf_factors(self) -> torch.Tensor:
        return torch.from_numpy(
            np.array(
                [
                    [0.98881036, 0.00550783, 0.00854675, 0.00452144],
                    [-2.4203062, -0.00740275, -0.00848471, -0.00448634],
                ],
                dtype=np.float32,
            )
        ).to(self.device)


    def __call__(
            self,
            stl_path: Path | str,
            stream_velocity: float = 38.889,
            stencil_size: int = 7,
            air_density: float = 1.205,
    ):
        mesh_stl = pv.read(stl_path)
        datapipe = DesignDatapipe(
            mesh=mesh_stl,
            bounding_box=self.bounding_box_min_max,
            bounding_box_surface=self.bounding_box_surface_min_max,
            grid_resolution=self.cfg.model.interp_res,
            stencil_size=stencil_size,
        )
        dataloader = torch.utils.data.DataLoader(datapipe, batch_size=4096, shuffle=False)

        input_dict = {
            k: torch.from_numpy(np.expand_dims(np.float32(v), axis=0)).to(self.device)
            for k, v in datapipe.out_dict.items()
        }
        input_dict["stream_velocity"] = torch.tensor(stream_velocity, dtype=torch.float32, device=self.device)
        input_dict["air_density"] = torch.tensor(air_density, dtype=torch.float32, device=self.device)

        surface_keys: list[str] = [
            "surface_mesh_centers",
            "surface_mesh_neighbors", 
            "surface_normals",
            "surface_neighbors_normals",
            "surface_areas",
            "surface_neighbors_areas",
            "pos_surface_center_of_mass"
        ]

        aerodynamic_force = torch.zeros(3, dtype=torch.float32, device=self.device)
        surface_area_batches: list[np.ndarray] = []
        pred_surf_batches: list[np.ndarray] = []

        for i_batch, sample_batched in enumerate(dataloader):
            sample_batched = {
                key: torch.unsqueeze(value, dim=0).to(self.device) for key, value in sample_batched.items()
            }
            # Update input dictionary with surface mesh data from sampled batch

            input_dict.update({k: sample_batched[k] for k in surface_keys})
            input_dict["geometry_coordinates"].requires_grad_(True)
            
            print(f"Allocated memory after data loading: {(torch.cuda.memory_allocated()/(1024**3)):.2f} GB")
            with torch.amp.autocast('cuda', enabled=True):
                prediction_vol, prediction_surf = self.model(input_dict)

                prediction_surf = (
                    unnormalize(
                        prediction_surf, self.surf_factors[0], self.surf_factors[1]
                    )
                    * stream_velocity ** 2.0
                    * air_density
                )
                surface_normals = input_dict["surface_normals"]

                aerodynamic_force_batch = torch.sum(
                    input_dict["surface_areas"][0][: , None] * (
                        surface_normals[0] * prediction_surf[0][:, 0]  # Pressure
                        - prediction_surf[0][:, 1:4]  # Wall shear stress
                    ),
                    dim=0
                )
                aerodynamic_force += aerodynamic_force_batch

                # TODO double check if indexing is correct in the expression above
                print(f"{aerodynamic_force_batch=}, {prediction_surf.shape=}, {(i_batch + 1) * 8000=}")

            surface_area_batches.append(
                input_dict["surface_areas"][0].detach().cpu().numpy()
            )
            pred_surf_batches.append(
                prediction_surf[0, :].detach().cpu().numpy()
            )
        
        surface_areas = np.concatenate(surface_area_batches, 0)
        pred_surf = np.concatenate(pred_surf_batches, 0)

        drag_force = aerodynamic_force[0]

        loss = (drag_force / 400) ** 2.0

        print(f"Allocated memory after loss calc: {(torch.cuda.memory_allocated()/(1024**3)):.2f} GB")

        loss.backward()

        sensitivities = input_dict["geometry_coordinates"].grad.cpu().detach().numpy()
        coordinates = input_dict["geometry_coordinates"].detach().cpu().numpy()

        return {
            "sensitivities": sensitivities,
            "coordinates": coordinates,
            "surface_areas": surface_areas,
            "pred_surf": pred_surf,
        }

if __name__ == "__main__":
    with hydra.initialize(version_base="1.3", config_path="conf"):
        cfg = hydra.compose(config_name="config")

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.world_size > 1:
        torch.distributed.barrier()

    # input_files = (Path(__file__).parent / "geometries").glob("*.stl")
    input_files = [Path(__file__).parent / "geometries" / "drivaer_1_single_solid_decimated3.stl"]

    domino = DoMINOInference(
        cfg=cfg,
        model_checkpoint_path=(Path(__file__).parent / "DoMINO.0.0.pt").absolute(),
        dist=dist,
    )

    for file in input_files:
        results = domino(
            stl_path=file.absolute(),
            stream_velocity=38.889,
            stencil_size=7,
            air_density=1.205,
        )