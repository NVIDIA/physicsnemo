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

"""Measure conservation when GLOBE resamples an already weighted mesh."""

import importlib.util
from pathlib import Path

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures


@pytest.mark.parametrize("geometry_only", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_subsample_preserves_represented_measure(geometry_only, explicit):
    pytest.importorskip("pyvista")
    path = (
        Path(__file__).parents[3]
        / "examples/cfd/external_aerodynamics/globe/drivaer/dataset.py"
    )
    spec = importlib.util.spec_from_file_location("globe_drivaer_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
        mesh = module.DrivAerMLDataSet.subsample_mesh(
            mesh, n_cells, geometry_only=geometry_only
        )
        torch.testing.assert_close(cell_measures(mesh).sum(), original.sum())
        torch.testing.assert_close(
            cell_measures(mesh) / cell_measures(mesh).sum(),
            retained / retained.sum(),
        )
