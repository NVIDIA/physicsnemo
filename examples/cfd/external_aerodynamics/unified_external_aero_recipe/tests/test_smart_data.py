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
from smart_data import (
    SMARTSurfaceMeshToDomainMesh,
    SMARTSurfaceReader,
    SMARTVolumeReader,
)

from physicsnemo.mesh import DomainMesh, Mesh
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
        global_data={"smart_query_seed": torch.tensor(27)},
    )


@pytest.mark.parametrize("query_count", [1, 100])
def test_surface_effective_areas_and_full_geometry(cfd_surface, query_count):
    """Query sampling transfers aligned inverse-inclusion areas to the targets."""
    domain = SMARTSurfaceMeshToDomainMesh(
        n_cells=query_count, cell_data_targets=["pressure"]
    )(cfd_surface)
    boundary = domain.boundaries["vehicle"]
    rows = domain.interior.point_data["pressure"].flatten().long()
    expected = cfd_surface.cell_areas[rows] * 3 / min(3, query_count)
    torch.testing.assert_close(cell_measures(boundary), expected)
    torch.testing.assert_close(point_measures(domain.interior), expected)
    torch.testing.assert_close(domain.interior.points, cfd_surface.cell_centroids[rows])
    torch.testing.assert_close(
        domain.boundaries["smart_geometry"].points, cfd_surface.cell_centroids
    )
    assert "pressure" not in boundary.cell_data


@pytest.mark.parametrize("field", ["surface", "volume"])
@pytest.mark.parametrize("cap", ["subsample_n_cells", "subsample_n_points"])
def test_reader_rejects_geometry_truncation(cfd_surface, tmp_path, field, cap):
    """Reject reader settings that would cap geometry before online sampling."""
    if field == "surface":
        cfd_surface.save(tmp_path / "surface.pmsh")
        reader = SMARTSurfaceReader
        kwargs = {"pattern": "*.pmsh"}
    else:
        DomainMesh(
            interior=Mesh(points=torch.zeros(2, 3)),
            boundaries={"vehicle": cfd_surface},
        ).save(tmp_path / "domain.pdmsh")
        reader = SMARTVolumeReader
        kwargs = {"pattern": "*.pdmsh", "boundary_subsample": "both"}
    with pytest.raises(ValueError, match="geometry|complete"):
        reader(path=tmp_path, **kwargs, **{cap: 1})
