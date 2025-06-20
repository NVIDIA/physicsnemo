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

import hydra
import torch
from .main import DoMINOInference
from pathlib import Path
from physicsnemo.distributed import DistributedManager
import pyvista as pv
import numpy as np

if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.9)

    with hydra.initialize(version_base="1.3", config_path="conf"):
        cfg = hydra.compose(config_name="config")

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

    print("Doing Initial Run...")

    mesh: pv.PolyData = pv.read(input_file)
    results: dict[str, np.ndarray] = domino(
        mesh=mesh,
        stream_velocity=38.889,  # m/s
        stencil_size=7,
        air_density=1.205,  # kg/m^3
    )

    print(f"Initial drag force: {results['aerodynamic_force'][0]} N")

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

    def get_drag(
        epsilon: float,
        sensitivities: np.ndarray,
    ) -> float:
        print(f"Doing Warped Run with epsilon: {epsilon}")
        warped_mesh = pv.PolyData(
            mesh.points + epsilon * sensitivities,
            mesh.faces,
        )
        warped_results: dict[str, np.ndarray] = domino(
            mesh=warped_mesh,
            stream_velocity=38.889,  # m/s
            stencil_size=7,
            air_density=1.205,  # kg/m^3
        )
        drag = warped_results["aerodynamic_force"][0]
        drag_grad = (drag - results["aerodynamic_force"][0]) / epsilon
        print(
            f"Epsilon: {epsilon:12.2e}, Drag: {drag:20.16g}, Drag Grad: {drag_grad:20.12g}"
        )
        return float(drag)

    epsilons = [0] + [
        sign * number * 10**exponent
        for exponent in range(-16, -12)
        for number in np.logspace(0, 1, 6)
        for sign in [1, -1]
    ]

    mesh = mesh.cell_data_to_point_data(pass_cell_data=True)

    for epsilon in epsilons:
        with open(
            Path(__file__).parent
            / "gradient_checking_results"
            / "drag_gradients_raw.txt",
            "a",
        ) as f:
            drag = get_drag(epsilon, mesh.point_data["raw_sensitivity_cells"])
            f.write(f"{epsilon},{drag}\n")

    for epsilon in epsilons:
        with open(
            Path(__file__).parent
            / "gradient_checking_results"
            / "drag_gradients_smooth.txt",
            "a",
        ) as f:
            drag = get_drag(epsilon, mesh.point_data["smooth_sensitivity_point"])
            f.write(f"{epsilon},{drag}\n")
