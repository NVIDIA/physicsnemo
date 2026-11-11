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

"""Measure conservation in GLOBE resampling and force postprocessing."""

import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures


@pytest.fixture(scope="module")
def drivaer_dataset():
    """Load the standalone example without changing the import path."""
    pytest.importorskip("pyvista")
    path = (
        Path(__file__).parents[3]
        / "examples/cfd/external_aerodynamics/globe/drivaer/dataset.py"
    )
    spec = importlib.util.spec_from_file_location("globe_drivaer_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("geometry_only", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_subsample_preserves_represented_measure(
    drivaer_dataset, geometry_only, explicit
):
    points = torch.tensor([[0.0, 0, 0], [2, 0, 0], [0, 1, 0]]).repeat(8, 1)
    mesh = Mesh(
        points=points,
        cells=torch.arange(24).reshape(8, 3),
        cell_data={"id": torch.arange(8)},
    )
    if explicit:
        set_cell_measures(mesh, torch.arange(1, 9, dtype=torch.float32))
    original = cell_measures(mesh).clone()
    for n_cells in (4, 2):
        torch.manual_seed(17)
        indices = torch.randperm(mesh.n_cells)[:n_cells]
        retained = cell_measures(mesh)[indices]
        torch.manual_seed(17)
        mesh = drivaer_dataset.DrivAerMLDataSet.subsample_mesh(
            mesh, n_cells, geometry_only=geometry_only
        )
        torch.testing.assert_close(cell_measures(mesh).sum(), original.sum())
        torch.testing.assert_close(
            cell_measures(mesh) / cell_measures(mesh).sum(),
            retained / retained.sum(),
        )


@pytest.mark.parametrize("n_cells", [2, 4])
@pytest.mark.parametrize("explicit", [False, True])
def test_postprocess_preserves_sampled_force_coefficients(
    drivaer_dataset, n_cells, explicit
):
    """Forces and the returned mesh both use the represented surface area."""
    full = Mesh(
        points=torch.tensor([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
        cells=torch.tensor([[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]),
        point_data={"C_p": torch.zeros(4), "C_f": torch.zeros(4, 3)},
        # Reference cell fields must not conflict with the predicted point fields.
        cell_data={"C_p": torch.zeros(4), "C_f": torch.zeros(4, 3)},
    )
    total_area = full.cell_areas.sum()
    if explicit:
        set_cell_measures(full, full.cell_areas * 1.5)
        total_area = total_area * 1.5
    torch.manual_seed(17)
    sampled = drivaer_dataset.DrivAerMLDataSet.subsample_mesh(
        full, n_cells, geometry_only=False
    )
    original_measures = cell_measures(sampled).clone()
    a_ref = 2.0
    sample = drivaer_dataset.DrivAerMLSample(
        prediction_mesh=sampled,
        boundary_meshes=TensorDict({}),
        reference_lengths=TensorDict({}),
        dimensional_constants=TensorDict({"A_ref": torch.tensor(a_ref)}),
        aero_coefficients=TensorDict(
            {key: torch.tensor(0.0) for key in ("Cd", "Cl", "Cs")}
        ),
    )
    traction = torch.tensor([2.0, 3.0, 4.0])
    prediction = sampled.to_point_cloud().with_data(
        point_data={
            "C_p": torch.zeros(sampled.n_points),
            "C_f": traction.repeat(sampled.n_points, 1),
        }
    )

    combined = drivaer_dataset.postprocess(pred_mesh=prediction, sample=sample)

    expected_force = traction * total_area / a_ref
    for key, axis in (("Cd", 0), ("Cl", 2), ("Cs", 1)):
        torch.testing.assert_close(
            combined.global_data["pred", key], expected_force[axis]
        )
        torch.testing.assert_close(
            combined.global_data["true", key], sample.aero_coefficients[key]
        )
    ### Disk rendering reads the represented area from the combined mesh.
    torch.testing.assert_close(cell_measures(combined), original_measures)
    torch.testing.assert_close(cell_measures(sampled), original_measures)
