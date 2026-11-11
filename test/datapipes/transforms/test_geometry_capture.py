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

"""Sampling safeguards and force-integration measures for SMART data."""

import pytest
import torch

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.datapipes.transforms.mesh import (
    CaptureGeometry,
    MeshToDomainMesh,
    SubsampleMesh,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, point_measures


@pytest.fixture
def cfd_surface():
    """Triangles with distinct areas and fields to detect row misalignment."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [0, 3, 4], [0, 2, 5]]),
        cell_data={"pressure": torch.arange(3.0).unsqueeze(-1)},
    )


@pytest.mark.parametrize("query_count", [1, 100])
@pytest.mark.parametrize("prefix", ["", "captured."])
def test_surface_effective_areas_and_full_geometry(cfd_surface, query_count, prefix):
    """Query sampling transfers aligned inverse-inclusion areas to the targets."""
    names = {
        name: prefix + value
        for name, value in {
            "geometry_field": "geometry_points",
            "center_field": "geometry_center",
            "query_field": "query_coordinates",
        }.items()
    }
    captured = CaptureGeometry(**names)(cfd_surface)
    sampled = SubsampleMesh(n_cells=query_count)(captured)
    domain = MeshToDomainMesh(cell_data_targets=["pressure"], copy_boundary=True)(
        sampled
    )
    boundary = domain.boundaries["vehicle"]
    rows = domain.interior.point_data["pressure"].flatten().long()
    expected = cfd_surface.cell_areas[rows] * 3 / min(3, query_count)
    torch.testing.assert_close(cell_measures(boundary), expected)
    torch.testing.assert_close(point_measures(domain.interior), expected)
    torch.testing.assert_close(domain.interior.points, cfd_surface.cell_centroids[rows])
    torch.testing.assert_close(
        domain.global_data[as_nested_key(names["geometry_field"])],
        cfd_surface.cell_centroids,
    )
    torch.testing.assert_close(
        boundary.cell_data[as_nested_key(names["query_field"])],
        cfd_surface.cell_centroids[rows],
    )
    centers = cfd_surface.cell_centroids
    torch.testing.assert_close(
        domain.global_data[as_nested_key(names["center_field"])],
        (centers.amin(0) + centers.amax(0)) / 2,
    )
    assert "pressure" not in boundary.cell_data
