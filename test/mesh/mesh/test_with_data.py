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

"""Tests for immutable Mesh field-data replacement."""

import torch

from physicsnemo.mesh import Mesh


def _mesh_with_data() -> Mesh:
    return Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
        point_data={"temperature": torch.tensor([1.0, 2.0, 3.0])},
        cell_data={"pressure": torch.tensor([4.0])},
        global_data={"time": torch.tensor(5.0)},
    )


def test_with_data_replaces_only_requested_association():
    mesh = _mesh_with_data()
    result = mesh.with_data(point_data={"prediction": torch.tensor([6.0, 7.0, 8.0])})

    assert set(result.point_data.keys()) == {"prediction"}
    assert set(result.cell_data.keys()) == {"pressure"}
    assert set(result.global_data.keys()) == {"time"}

    # The source remains unchanged.
    assert set(mesh.point_data.keys()) == {"temperature"}
    assert "prediction" not in mesh.point_data


def test_with_data_preserves_geometry_and_populated_cache():
    mesh = _mesh_with_data()
    centroids = mesh.cell_centroids
    areas = mesh.cell_areas

    result = mesh.with_data(cell_data={"new_pressure": torch.tensor([9.0])})

    assert result.points.data_ptr() == mesh.points.data_ptr()
    assert result.cells.data_ptr() == mesh.cells.data_ptr()
    assert result.cell_centroids.data_ptr() == centroids.data_ptr()
    assert result.cell_areas.data_ptr() == areas.data_ptr()


def test_with_data_cache_container_does_not_alias_source():
    mesh = _mesh_with_data()
    result = mesh.with_data(point_data={})

    _ = result.cell_centroids
    assert mesh._cache.get(("cell", "centroids"), None) is None


def test_with_data_data_containers_do_not_alias_source_structure():
    mesh = _mesh_with_data()
    result = mesh.with_data()

    result.point_data["added"] = torch.zeros(mesh.n_points)
    result.cell_data["added"] = torch.zeros(mesh.n_cells)
    result.global_data["added"] = torch.tensor(0.0)

    assert "added" not in mesh.point_data
    assert "added" not in mesh.cell_data
    assert "added" not in mesh.global_data


def test_with_data_empty_mapping_clears_association():
    mesh = _mesh_with_data()
    result = mesh.with_data(cell_data={})

    assert len(result.cell_data.keys()) == 0
    assert "pressure" in mesh.cell_data


def test_with_data_preserves_expected_batch_sizes():
    mesh = _mesh_with_data()
    result = mesh.with_data(
        point_data={"x": torch.ones(mesh.n_points)},
        cell_data={"y": torch.ones(mesh.n_cells)},
        global_data={"z": torch.tensor(1.0)},
    )

    assert result.point_data.batch_size == torch.Size([mesh.n_points])
    assert result.cell_data.batch_size == torch.Size([mesh.n_cells])
    assert result.global_data.batch_size == torch.Size([])
