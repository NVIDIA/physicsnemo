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
        global_data={"query_seed": torch.tensor(27)},
    )


@pytest.mark.parametrize("query_count", [1, 100])
def test_surface_effective_areas_and_full_geometry(cfd_surface, query_count):
    """Query sampling transfers aligned inverse-inclusion areas to the targets."""
    captured = CaptureGeometry()(cfd_surface)
    sampled = SubsampleMesh(n_cells=query_count, seed_field="query_seed")(captured)
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
        domain.global_data["geometry_points"], cfd_surface.cell_centroids
    )
    assert "pressure" not in boundary.cell_data


def test_unsigned_boundary_distance_and_physical_coordinates(device):
    """Distances stay unsigned and directions vanish at the surface."""
    from physicsnemo.datapipes.transforms.mesh import ComputeUnsignedDistance
    from physicsnemo.mesh import DomainMesh

    surface = Mesh(
        points=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
        ),
        cells=torch.tensor([[0, 1, 2]], device=device),
    )
    points = torch.tensor(
        [[0.2, 0.2, 0.5], [0.2, 0.2, -0.5], [0.2, 0.2, 0.0]], device=device
    )
    domain = DomainMesh(interior=Mesh(points=points), boundaries={"vehicle": surface})
    captured = CaptureGeometry().apply_to_domain(domain)
    result = ComputeUnsignedDistance(
        coordinate_scale=(12.0, 4.5, 3.25)
    ).apply_to_domain(captured)
    fields = result.interior.point_data
    torch.testing.assert_close(fields["query_coordinates"], points, rtol=0, atol=0)
    torch.testing.assert_close(
        fields["sdf"].flatten(), points[:, 2].abs(), rtol=0, atol=1e-6
    )
    expected = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]], device=device
    )
    torch.testing.assert_close(fields["sdf_normals"], expected, rtol=0, atol=1e-6)
