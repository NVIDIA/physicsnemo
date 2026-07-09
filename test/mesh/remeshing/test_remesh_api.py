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

"""Device-independent tests for the public Mesh remeshing API."""

import inspect

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.remeshing import WarpRemeshOptions, remesh


def test_remesh_public_signatures():
    assert tuple(inspect.signature(remesh).parameters) == (
        "mesh",
        "n_clusters",
        "max_iterations",
        "warp_options",
    )
    assert "warp_options" in inspect.signature(Mesh.remesh).parameters


def test_remesh_rejects_wrong_options_type_on_cpu():
    source = sphere_icosahedral.load(subdivisions=2)

    with pytest.raises(TypeError, match="WarpRemeshOptions instance.*dict"):
        remesh(
            source,
            48,
            warp_options={"hash_grid_resolution": 64},
        )


@pytest.mark.parametrize(
    "warp_options",
    [None, WarpRemeshOptions(farthest_point_threshold=0)],
)
def test_remesh_runs_on_cpu(warp_options):
    source = sphere_icosahedral.load(subdivisions=2)
    output = remesh(
        source,
        48,
        max_iterations=1,
        warp_options=warp_options,
    )

    assert 3 <= output.n_points <= 48
    assert output.points.shape[1] == 3
    assert output.cells.ndim == 2 and output.cells.shape[1] == 3
    assert output.points.device.type == "cpu"
    assert output.points.dtype == source.points.dtype
    assert output.cells.dtype == torch.int64
    report = output.validate(check_manifoldness=True)
    assert report["valid"]
    assert report["is_manifold"]


def test_mesh_remesh_runs_on_cpu():
    source = sphere_icosahedral.load(subdivisions=1)
    output = source.remesh(24, max_iterations=1)

    assert 3 <= output.n_points <= 24
    assert output.points.device.type == "cpu"


def test_cpu_remesh_rejects_unsafe_geometry():
    nonfinite = sphere_icosahedral.load(subdivisions=1)
    nonfinite.points[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        remesh(nonfinite, 24)

    invalid_cells = sphere_icosahedral.load(subdivisions=1)
    invalid_cells.cells[0, 0] = invalid_cells.n_points
    with pytest.raises(ValueError, match="indices"):
        remesh(invalid_cells, 24)


def test_remesh_rejects_non_surface_mesh():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cells = torch.tensor([[0, 1], [1, 2]])
    source = Mesh(points=points, cells=cells)

    with pytest.raises(NotImplementedError, match="2D triangle surface"):
        remesh(source, 3)
