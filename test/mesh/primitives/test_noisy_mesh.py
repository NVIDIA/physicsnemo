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

"""Tests for the coordinate-noise mesh primitive."""

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.procedural import noisy_mesh


def test_noisy_mesh_invalidates_geometry_and_retains_topology() -> None:
    mesh = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
        point_data={"temperature": torch.tensor([1.0, 2.0, 3.0])},
    )
    _ = mesh.cell_areas
    _ = mesh.cell_centroids
    _ = mesh.point_normals
    topology = mesh.get_point_to_points_adjacency()

    result = noisy_mesh.load(mesh, noise_scale=0.1, seed=7)

    assert not torch.equal(result.points, mesh.points)
    torch.testing.assert_close(
        result.point_data["temperature"], mesh.point_data["temperature"]
    )
    assert list(result._cache["cell"].keys()) == []
    assert list(result._cache["point"].keys()) == []

    cached_topology = result._cache.get(("topology", "point_to_points"))
    assert cached_topology is not None
    assert cached_topology.offsets.data_ptr() == topology.offsets.data_ptr()
    assert cached_topology.indices.data_ptr() == topology.indices.data_ptr()

    # Coordinate replacement must not clear already-computed source geometry.
    assert mesh._cache.get(("cell", "areas")) is not None
    assert mesh._cache.get(("cell", "centroids")) is not None
    assert mesh._cache.get(("point", "normals")) is not None
