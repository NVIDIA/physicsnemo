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

import numpy as np
import pytest
import torch

from physicsnemo.nn.module.interpolation import interpolation


@pytest.mark.parametrize("mem_speed_trade", [True, False])
def test_interpolation(mem_speed_trade):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # make context grid to do interpolation from
    grid = [(-1, 2, 30), (-1, 2, 30), (-1, 2, 30)]
    np_linspace = [np.linspace(x[0], x[1], x[2]) for x in grid]
    np_mesh_grid = np.meshgrid(*np_linspace, indexing="ij")
    np_mesh_grid = np.stack(np_mesh_grid, axis=0)
    mesh_grid = torch.tensor(np_mesh_grid, dtype=torch.float32).to(device)
    sin_grid = torch.sin(
        mesh_grid[0:1, :, :] + mesh_grid[1:2, :, :] ** 2 + mesh_grid[2:3, :, :] ** 3
    ).to(device)

    # make query points to evaluate on
    nr_points = 100
    query_points = (
        torch.stack(
            [
                torch.linspace(0.0, 1.0, nr_points),
                torch.linspace(0.0, 1.0, nr_points),
                torch.linspace(0.0, 1.0, nr_points),
            ],
            axis=-1,
        )
        .to(device)
        .requires_grad_(True)
    )

    # compute interpolation
    interpolation_types = [
        "nearest_neighbor",
        "linear",
        "smooth_step_1",
        "smooth_step_2",
        "gaussian",
    ]
    for i_type in interpolation_types:
        # perform interpolation
        computed_interpolation = interpolation(
            query_points,
            sin_grid,
            grid=grid,
            interpolation_type=i_type,
            mem_speed_trade=mem_speed_trade,
        )

        # compare to numpy
        np_computed_interpolation = computed_interpolation.cpu().detach().numpy()
        np_ground_truth = (
            (
                torch.sin(
                    query_points[:, 0:1]
                    + query_points[:, 1:2] ** 2
                    + query_points[:, 2:3] ** 3
                )
            )
            .cpu()
            .detach()
            .numpy()
        )
        difference = np.linalg.norm(
            (np_computed_interpolation - np_ground_truth) / nr_points
        )

        # verify
        assert difference < 1e-2, "Test failed!"
