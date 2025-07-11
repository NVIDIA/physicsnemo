# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this input_file except in compliance with the License.
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
or inferred from the config input_file. The results are calculated on a point cloud
sampled in the volume around the STL and on the surface of the STL. They are stored
in a dictionary, which can be written out for visualization.
"""

from functools import cached_property
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

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
            self.model = (
                DoMINO(
                    input_features=3,
                    output_features_vol=self.num_vol_vars,
                    output_features_surf=self.num_surf_vars,
                    model_parameters=self.cfg.model,
                )
                .to(self.device)
                .eval()
            )

            for param in self.model.parameters():
                param.requires_grad = False

            self.model = torch.compile(self.model, disable=True)  # TODO review

            if self.model_checkpoint_path is not None:
                with open(self.model_checkpoint_path, "rb") as f:
                    self.model.load_state_dict(torch.load(f, map_location=self.device))
            else:
                import warnings

                warnings.warn(
                    "Model loaded without checkpoint. This is not recommended for production use.",
                    stacklevel=2,
                )

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
                np.array(self.cfg.data.bounding_box.max, dtype=np.float32),
            )
        except AttributeError:
            raise ValueError(
                "Config must specify both `bounding_box.min` and `bounding_box.max`"
            )

    @cached_property
    def bounding_box_surface_min_max(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get the minimum and maximum coordinates of the surface bounding box from config.

        Returns:
            tuple[NDArray[np.float32], NDArray[np.float32]]: Min and max coordinates

        Raises:
            ValueError: If min or max coordinates are not specified in config
        """
        try:
            return (
                np.array(self.cfg.data.bounding_box_surface.min, dtype=np.float32),
                np.array(self.cfg.data.bounding_box_surface.max, dtype=np.float32),
            )
        except AttributeError:
            raise ValueError(
                "Config must specify both `bounding_box_surface.min` and `bounding_box_surface.max`"
            )

    @cached_property
    def vol_factors(self) -> torch.Tensor:
        """
        Computes the factors that are later used to unnormalize the volume predictions.

        These are saved at training time based on statistics of the training data, and re-used at each inference call.
        """
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
        """
        Computes the factors that are later used to unnormalize the surface predictions.

        These are saved at training time based on statistics of the training data, and re-used at each inference call.
        """
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
        mesh: pv.PolyData,
        stream_velocity: float = 38.889,
        stencil_size: int = 7,
        air_density: float = 1.205,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        """Performs DoMINO inference on a given geometry to predict aerodynamic quantities.

        This method takes a PyVista mesh representing a 3D geometry and computes the
        aerodynamic predictions using the DoMINO model. It handles the data preprocessing,
        model inference, and post-processing of results.

        Args:
            mesh: PyVista PolyData mesh representing the 3D geometry to analyze
            stream_velocity: Inlet flow velocity in m/s. Defaults to 38.889 m/s.
            stencil_size: Number of neighboring points to consider for surface calculations.
                Defaults to 7.
            air_density: Air density in kg/m³. Defaults to 1.205 kg/m³.
            verbose: Whether to print verbose output. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - 'geometry_coordinates': Array of geometry point coordinates
                - 'geometry_normal_sensitivity': Array of sensitivity values for each point
                - 'pred_surf_pressure': Array of predicted surface pressure values [Pa]
                - 'pred_surf_wall_shear_stress': Array of predicted wall shear stress values [τx, τy, τz] [Pa]
                - 'aerodynamic_force': Array of total computed aerodynamic force [Fx, Fy, Fz] [N]

        Example:
            >>> import pyvista as pv
            >>> from domino_sensitivity import DoMINOInference
            >>>
            >>> # Load geometry
            >>> mesh = pv.read("car.stl")
            >>>
            >>> # Initialize inference
            >>> domino = DoMINOInference(cfg)
            >>>
            >>> # Run inference
            >>> results = domino(
            ...     mesh=mesh,
            ...     stream_velocity=30.0,
            ...     stencil_size=7,
            ...     air_density=1.205
            ... )
            >>>
            >>> # Access results
            >>> forces = results['aerodynamic_force']
            >>> print(f"Drag force: {forces[0]:.2f} N")
        """

        datapipe = DesignDatapipe(
            mesh=mesh,
            bounding_box=self.bounding_box_min_max,
            bounding_box_surface=self.bounding_box_surface_min_max,
            grid_resolution=self.cfg.model.interp_res,
            stencil_size=stencil_size,
            device=self.device,
        )
        dataloader = torch.utils.data.DataLoader(
            datapipe, batch_size=2**13, shuffle=False
        )

        input_dict: dict[str, torch.Tensor] = {
            k: torch.unsqueeze(v, dim=0) for k, v in datapipe.out_dict.items()
        }
        input_dict["stream_velocity"] = torch.tensor(stream_velocity)
        input_dict["air_density"] = torch.tensor(air_density)

        surface_keys: list[str] = [
            "surface_mesh_centers",
            "surface_mesh_neighbors",
            "surface_normals",
            "surface_neighbors_normals",
            "surface_areas",
            "surface_neighbors_areas",
            "pos_surface_center_of_mass",
        ]

        aerodynamic_force = np.zeros(3, dtype=np.float32)
        pred_surf_batches: list[np.ndarray] = []
        geometry_coordinates = (
            input_dict["geometry_coordinates"].detach().cpu().numpy()[0]
        )
        geometry_sensitivity: np.ndarray = np.zeros_like(geometry_coordinates)

        def _print_memory_usage(label: str) -> None:
            if verbose:
                print(
                    f"VRAM usage after {label:<20}: {(torch.cuda.memory_allocated()/(1024**3)):.2f} GB"
                )

        for sample_batched in tqdm(dataloader, desc="Processing batches"):
            # Update input dictionary with surface mesh data from sampled batch
            input_dict_batch: dict[str, torch.Tensor] = {
                **input_dict,
                **{k: torch.unsqueeze(sample_batched[k], dim=0) for k in surface_keys},
            }
            input_dict_batch["geometry_coordinates"].requires_grad_(True)

            _print_memory_usage(label="data loading")

            prediction_vol_batch, prediction_surf_batch = self.model(input_dict_batch)

            _print_memory_usage(label="model forward")

            # This is required to free memory. It's a bit atypical to do this,
            # but in this case it allows us to drop all references to the
            # PyTorch computational graph, which allows PyTorch to garbage
            # collect the primal values stored on the graph nodes before the
            # next forward pass.
            del prediction_vol_batch

            prediction_surf_batch = (
                unnormalize(
                    prediction_surf_batch, self.surf_factors[0], self.surf_factors[1]
                )
                * stream_velocity**2.0
                * air_density
            )
            surface_areas_batch = input_dict_batch["surface_areas"][0]
            surface_normals_batch = input_dict_batch["surface_normals"][0]
            pressure_batch = prediction_surf_batch[0][:, 0]
            wall_shear_stress_batch = prediction_surf_batch[0][:, 1:4]

            aerodynamic_force_batch = torch.sum(
                surface_areas_batch[:, None]
                * (
                    surface_normals_batch * pressure_batch[:, None]  # Pressure
                    - wall_shear_stress_batch  # Wall shear stress
                ),
                dim=0,  # Sums over all points in the batch
            )
            drag_force_batch = aerodynamic_force_batch[0]
            (
                -1 * drag_force_batch
            ).backward()  # Vectors represent how you should modify the geometry to *reduce* drag
            _print_memory_usage(label="surface backward")

            # Compute the sensitivity of the drag force to the geometry coordinates, from this batch
            geometry_sensitivity_batch = input_dict_batch["geometry_coordinates"].grad[
                0
            ]

            geometry_sensitivity += geometry_sensitivity_batch.cpu().detach().numpy()
            aerodynamic_force += aerodynamic_force_batch.cpu().detach().numpy()

            pred_surf_batches.append(prediction_surf_batch[0].detach().cpu().numpy())

        pred_surf = np.concatenate(pred_surf_batches, 0)

        return {
            "geometry_coordinates": geometry_coordinates,
            "geometry_sensitivity": geometry_sensitivity,
            "pred_surf_pressure": pred_surf[:, 0],
            "pred_surf_wall_shear_stress": pred_surf[:, 1:4],
            "aerodynamic_force": aerodynamic_force,
        }

    @staticmethod
    def postprocess_point_sensitivities(
        results: dict[str, np.ndarray], mesh: pv.PolyData, n_laplacian_iters: int = 20
    ) -> dict[str, np.ndarray]:
        """Postprocess the raw geometry sensitivities to compute normal and smoothed sensitivities.

        This function takes the raw geometry sensitivities and computes:
        1. Normal sensitivities by projecting onto cell normals
        2. Full sensitivity vectors by scaling cell normals by normal sensitivities
        3. Smoothed versions of both using Laplacian smoothing

        Parameters
        ----------
        results : dict[str, np.ndarray]
            Dictionary containing the raw results from the forward pass, including:
            - geometry_sensitivity: Raw sensitivity vectors for each cell (n_cells, 3)
            - Other keys are preserved in the output
        mesh : pv.PolyData
            PyVista mesh containing the geometry and cell normals
        n_laplacian_iters : int, optional
            Number of Laplacian smoothing iterations to apply, by default 20

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing processed sensitivities:
            - raw_sensitivity_cells: Original geometry sensitivity vectors (n_cells, 3)
            - raw_sensitivity_normal_cells: Scalar sensitivities projected onto cell normals (n_cells,)
            - sensitivity: Full sensitivity vectors scaled by normal components (n_cells, 3)
            - smooth_sensitivity_normal_point: Laplacian-smoothed normal sensitivities (n_cells,)
            - sensitivity_smoothed: Laplacian-smoothed full sensitivity vectors (n_cells, 3)
        """
        raw_sensitivity_cells = mesh.cell_data["geometry_sensitivity"]
        raw_sensitivity_normal_cells = np.einsum(
            "ij,ij->i",
            raw_sensitivity_cells,
            mesh.cell_normals,
        )

        from utilities.mesh_postprocessing import laplacian_smoothing

        mesh_pointdata = pv.PolyData(mesh.points, mesh.faces)
        mesh_pointdata.cell_data[
            "raw_sensitivity_normal_cells"
        ] = raw_sensitivity_normal_cells
        mesh_pointdata = mesh_pointdata.cell_data_to_point_data()

        smooth_sensitivity_normal_point = laplacian_smoothing(
            mesh_pointdata,
            mesh_pointdata.point_data["raw_sensitivity_normal_cells"],
            location="points",
            iterations=n_laplacian_iters,
        )
        smooth_sensitivity_point = np.einsum(
            "i,ij->ij",
            smooth_sensitivity_normal_point,
            mesh.point_normals,
        )

        mesh_pointdata.clear_data()
        mesh_pointdata.point_data[
            "smooth_sensitivity_normal_point"
        ] = smooth_sensitivity_normal_point
        mesh_pointdata = mesh_pointdata.point_data_to_cell_data()

        smooth_sensitivity_normal_cell = mesh_pointdata.cell_data[
            "smooth_sensitivity_normal_point"
        ]

        smooth_sensitivity_cell = np.einsum(
            "i,ij->ij",
            smooth_sensitivity_normal_cell,
            mesh.cell_normals,
        )

        return {
            "raw_sensitivity_cells": raw_sensitivity_cells,
            "raw_sensitivity_normal_cells": raw_sensitivity_normal_cells,
            "smooth_sensitivity_point": smooth_sensitivity_point,
            "smooth_sensitivity_normal_point": smooth_sensitivity_normal_point,
            "smooth_sensitivity_cell": smooth_sensitivity_cell,
            "smooth_sensitivity_normal_cell": smooth_sensitivity_normal_cell,
        }


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.9)

    config_path = Path() / ".." / "domino" / "src" / "conf"
    assert config_path.exists(), f"{config_path=} does not exist!"
    with hydra.initialize(version_base="1.3", config_path=str(config_path)):
        cfg: DictConfig = hydra.compose(config_name="config")

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.world_size > 1:
        torch.distributed.barrier()

    domino = DoMINOInference(
        cfg=cfg,
        model_checkpoint_path=(Path(__file__).parent / "DoMINO.0.41.pt").absolute(),
        dist=dist,
    )

    input_file = Path(__file__).parent / "geometries" / "drivaer_1_single_solid.stl"

    mesh: pv.PolyData = pv.read(input_file)
    results: dict[str, np.ndarray] = domino(
        mesh=mesh,
        stream_velocity=38.889,  # m/s
        stencil_size=7,
        air_density=1.205,  # kg/m^3
    )

    for key, value in results.items():
        if len(value) == mesh.n_cells:
            mesh.cell_data[key] = value
        elif len(value) == mesh.n_points:
            mesh.point_data[key] = value

    sensitivity_results: dict[str, np.ndarray] = domino.postprocess_point_sensitivities(
        results=results, mesh=mesh
    )

    for key, value in sensitivity_results.items():
        mesh[key] = value
    mesh.save(input_file.with_suffix(".vtk"))
