# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

import warnings

import numpy as np
import pytest
import torch

from physicsnemo.nn.functional import grid_to_point_interpolation, interpolation


def _build_inputs(device: str):
    """Build a 3D sinusoidal context grid and a set of 1D-stacked query points."""
    grid = [(-1, 2, 30), (-1, 2, 30), (-1, 2, 30)]
    np_linspace = [np.linspace(x[0], x[1], x[2]) for x in grid]
    np_mesh_grid = np.meshgrid(*np_linspace, indexing="ij")
    np_mesh_grid = np.stack(np_mesh_grid, axis=0)
    mesh_grid = torch.tensor(np_mesh_grid, dtype=torch.float32).to(device)
    sin_grid = torch.sin(
        mesh_grid[0:1, :, :] + mesh_grid[1:2, :, :] ** 2 + mesh_grid[2:3, :, :] ** 3
    ).to(device)

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
    return grid, sin_grid, query_points, nr_points


@pytest.mark.parametrize("mem_speed_trade", [True, False])
def test_grid_to_point_interpolation(mem_speed_trade):
    """Accuracy test for the (non-deprecated) ``grid_to_point_interpolation``.

    Verifies all five interpolation kernels against a numpy ground truth on a
    sinusoidal field.  Uses the ``torch`` backend explicitly so the test runs
    deterministically without requiring ``warp``'s CUDA backend.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid, sin_grid, query_points, nr_points = _build_inputs(device)

    interpolation_types = [
        "nearest_neighbor",
        "linear",
        "smooth_step_1",
        "smooth_step_2",
        "gaussian",
    ]
    for i_type in interpolation_types:
        computed_interpolation = grid_to_point_interpolation(
            query_points,
            sin_grid,
            grid=grid,
            interpolation_type=i_type,
            mem_speed_trade=mem_speed_trade,
            implementation="torch",
        )

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

        assert difference < 1e-2, f"Test failed for interpolation_type={i_type!r}"


def test_interpolation_deprecated_alias_emits_warning_and_matches_new_api():
    """The deprecated ``interpolation`` alias must:

    * emit exactly one :class:`DeprecationWarning` per call, and
    * produce numerically-identical output to ``grid_to_point_interpolation``
      with ``implementation="torch"`` (the alias's preserved historical
      default).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid, sin_grid, query_points, _ = _build_inputs(device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_output = interpolation(
            query_points,
            sin_grid,
            grid=grid,
            interpolation_type="linear",
            mem_speed_trade=True,
        )

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1, (
        f"Expected exactly one DeprecationWarning from `interpolation`, "
        f"got {len(deprecation_warnings)}"
    )
    assert "grid_to_point_interpolation" in str(deprecation_warnings[0].message)

    new_api_output = grid_to_point_interpolation(
        query_points,
        sin_grid,
        grid=grid,
        interpolation_type="linear",
        mem_speed_trade=True,
        implementation="torch",
    )

    torch.testing.assert_close(deprecated_output, new_api_output)
